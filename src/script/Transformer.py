import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime
from DataProcessor import NonBinaryDataset, PastRaceFeatures, DEFAULT_DATASET_DIR
from CatBoost import DEFAULT_PREDICTION_OUTPUT_DIR
from DataProcessor import LoadLatest5Records

MAX_HORSES = 18
PAD_CAT = 0
MODEL_NAME = {
    "中山": "transformer_model.pth"
}

class HorseTransformer(nn.Module):
    def __init__(self, input_dim=9, d_model=64, nhead=4, num_layers=2, max_len=5):
        super().__init__()

        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_embed = nn.Parameter(torch.zeros(1, max_len, d_model))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

    def forward(self, x, mask=None):
        x = self.input_proj(x)
        x = x + self.pos_embed[:, :x.size(1), :]
        x = self.transformer(x, src_key_padding_mask=mask)

        if mask is None:
            return x[:, -1, :]

        # 有効長 = False の数
        lengths = (~mask).sum(dim=1).clamp(min=1)   # (B*N,)
        last_idx = lengths - 1                      # 最後の有効位置

        batch_idx = torch.arange(x.size(0), device=x.device)
        x = x[batch_idx, last_idx, :]               # (B*N, d_model)

        return x
    
class RaceModel(nn.Module):
    def __init__(self, input_dim=9, d_model=64):
        super().__init__()

        self.horse_encoder = HorseTransformer(input_dim, d_model)
        self.scorer = nn.Linear(d_model, 1, bias=False)

    def forward(
        self,
        x,
        time_mask=None,
        horse_mask=None,
        return_parts=False,
    ):
        B, N, T, F = x.shape

        # CatBoost取り出し
        cat_scores = x[:, :, 0, -1]  # (B, N)

        # reshape
        x = x.reshape(B * N, T, F)

        if time_mask is not None:
            time_mask = time_mask.reshape(B * N, T)

        x = self.horse_encoder(x, time_mask)

        nn_score = self.scorer(x).view(B, N)

        if horse_mask is not None:
            valid = ~horse_mask
            valid_f = valid.float()

            mean = (
                (nn_score * valid_f).sum(dim=1, keepdim=True)
                / valid_f.sum(dim=1, keepdim=True).clamp_min(1.0)
            )
            nn_score = nn_score - mean

        nn_score = torch.tanh(nn_score)
        #scores = cat_scores + 0.5 * nn_score
        scores = nn_score

        if horse_mask is not None:
            scores = scores.masked_fill(horse_mask, -1e9)

        if return_parts:
            return scores, cat_scores, nn_score

        return scores

def pad_ranks(ranks, max_horses=18):
    ranks = list(ranks)
    if len(ranks) < max_horses:
        pad_size = max_horses - len(ranks)
        ranks += [999] * pad_size
    return torch.tensor(ranks, dtype=torch.long)
    
def build_nn_input(features_tuple, cat_scores, max_horses=18):
    features = [np.array(f) for f in features_tuple]  # 8個

    # (8, 頭数, 5)
    x = np.stack(features, axis=0)

    # (頭数, 5, 8)
    x = np.transpose(x, (1, 2, 0))

    num_horses = x.shape[0]

    # --- CatBoost特徴追加 ---
    cat_scores = np.array(cat_scores)  # (頭数,)
    cat_feature = np.repeat(cat_scores[:, None], 5, axis=1)  # (頭数, 5)
    cat_feature = cat_feature[:, :, None]  # (頭数, 5, 1)

    x = np.concatenate([x, cat_feature], axis=2)  # (頭数, 5, 9)

    # --- 時系列mask ---
    # 差分日数（0番目特徴）で判定
    time_mask = (x[:, :, 0] == -1)  # (頭数, 5)

    # --- horse_mask ---
    horse_mask = time_mask.all(axis=1)  # (頭数,)

    # --- 頭数パディング ---
    if num_horses < max_horses:
        pad_size = max_horses - num_horses

        # x padding
        pad_x = np.zeros((pad_size, 5, x.shape[2]))
        pad_x[:, :, 8] = PAD_CAT
        x = np.concatenate([x, pad_x], axis=0)

        # mask padding（全部True = 無効）
        pad_time_mask = np.ones((pad_size, 5), dtype=bool)
        time_mask = np.concatenate([time_mask, pad_time_mask], axis=0)

        pad_horse_mask = np.ones(pad_size, dtype=bool)
        horse_mask = np.concatenate([horse_mask, pad_horse_mask], axis=0)

    return (
        torch.tensor(x, dtype=torch.float32),          # (18, 5, 9)
        torch.tensor(time_mask, dtype=torch.bool),     # (18, 5)
        torch.tensor(horse_mask, dtype=torch.bool)     # (18,)
    )

class RaceDataset(torch.utils.data.Dataset):
    def __init__(self, races):
        self.races = races

    def __getitem__(self, idx):
        x, ranks, time_mask, horse_mask = self.races[idx]
        return x, ranks, time_mask, horse_mask

    def __len__(self):
        return len(self.races)
    
def pairwise_loss(scores, ranks, horse_mask):
    """
    ベクトル化版ペアワイズロス
    scores: (B, N)
    ranks:  (B, N)  小さいほど上位（1着=1）
    horse_mask: (B, N) True=無効
    """
    B, N = scores.shape

    # 無効馬はスコアを-1e9にしておく
    scores = scores.masked_fill(horse_mask, -1e9)
    
    # ペアワイズ差分
    diff = scores[:, :, None] - scores[:, None, :]  # (B, N, N)

    # ランク条件
    rank_i = ranks[:, :, None]  # (B, N, 1)
    rank_j = ranks[:, None, :]  # (B, 1, N)
    mask = (rank_i < rank_j) & (~horse_mask[:, :, None]) & (~horse_mask[:, None, :])  # Trueならloss対象

    # logsigmoid を適用して、maskで不要部分をゼロに
    loss = -F.logsigmoid(diff / 10) * mask.float()

    # 平均
    total_loss = loss.sum() / (mask.sum() + 1e-8)
    return total_loss

def makeTrainingData(place: str):

    catboost_pred_dir = Path(DEFAULT_PREDICTION_OUTPUT_DIR) / place
    if not catboost_pred_dir.exists():
        raise FileNotFoundError(catboost_pred_dir)

    all_pred_results = list(catboost_pred_dir.glob("*.parquet"))

    temp = []

    # =========================
    # 1回目：全部集める
    # =========================

    for pred_result in all_pred_results:

        cat_scores = pd.read_parquet(pred_result)["score"].to_numpy()
        cat_scores = (cat_scores - cat_scores.mean()) / (cat_scores.std() + 1e-8)

        race_name = pred_result.stem
        dataset = Path(DEFAULT_DATASET_DIR) / place / (race_name + ".csv")

        nbd = NonBinaryDataset.from_csv(dataset)

        features_tuple = nbd.to_time_scale_dataset()

        ranks = [int(r) for r in nbd.finish_order_list]
        ranks = pad_ranks(ranks)
        ranks = (ranks - 1) / (MAX_HORSES - 1)

        temp.append((features_tuple, cat_scores, ranks))


    # =========================
    # stats作る
    # =========================

    all_values = [[] for _ in range(8)]

    for features_tuple, _, _ in temp:

        for i, feat in enumerate(features_tuple):

            for horse in feat:
                for v in horse:
                    if v != -1 and v != 0:
                        all_values[i].append(v)


    means = []
    stds = []

    for vals in all_values:
        vals = np.array(vals)
        means.append(vals.mean())
        stds.append(vals.std() + 1e-8)


    # =========================
    # 正規化関数
    # =========================

    def normalize(features_tuple):

        out = []

        for i, feat in enumerate(features_tuple):

            mean = means[i]
            std = stds[i]

            new_feat = []

            for horse in feat:

                seq = []

                for v in horse:

                    if v == -1 or v == 0:
                        seq.append(v)
                    else:
                        seq.append((v - mean) / std)

                new_feat.append(seq)

            out.append(new_feat)

        return tuple(out)


    # =========================
    # 2回目：build
    # =========================

    races = []

    for features_tuple, cat_scores, ranks in temp:

        features_tuple = normalize(features_tuple)

        x, time_mask, horse_mask = build_nn_input(
            features_tuple,
            cat_scores
        )

        races.append((x, ranks, time_mask, horse_mask))


    return races

def Train_Transformer(place):
    races = makeTrainingData(place)

    dataset = RaceDataset(races)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = RaceModel(input_dim=9).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)

    model.train()

    for epoch in range(10):
        total_loss = 0
        for x, ranks, time_mask, horse_mask in dataloader:
            x = x.to(device)
            ranks = ranks.to(device)
            time_mask = time_mask.to(device)
            horse_mask = horse_mask.to(device)

            scores = model(x, time_mask, horse_mask)
            loss = pairwise_loss(scores, ranks, horse_mask)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        scheduler.step()

        print(f"epoch {epoch}: loss={total_loss:.4f}, lr={scheduler.get_last_lr()[0]:.6f}")

    save_path = MODEL_NAME[place]

    # モデル保存
    torch.save(model.state_dict(), save_path)
    print(f"model saved to: {save_path}")

def loadModel(place, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = RaceModel(input_dim=9).to(device)

    model.load_state_dict(torch.load(MODEL_NAME[place], map_location=device))

    model.eval()
    return model

def makePredictionInputFeature(horse_name_list, horse_conditions, base, max_len=5):
    date_now_list, date_diff_list = [], []
    jw_now_list, jw_diff_list = [], []
    hw_now_list, hw_diff_list = [], []
    f3_diff_list, pci_diff_list = [], []

    past_record_list = []
    race_date_list = []
    for horse_name in horse_name_list:
        record_5, race_date = LoadLatest5Records(horse_name)
        past_record_list.append(record_5)
        race_date_list.append(race_date)


    for i in range(len(race_date_list)):
        dates = race_date_list[i] or []
        records = past_record_list[i] or []

        # =========================
        # ペア化 & ソート
        # =========================
        paired = []
        for j in range(min(len(dates), len(records))):
            try:
                dt = datetime.strptime(dates[j], "%Y%m%d")
                paired.append((dt, records[j]))
            except:
                continue

        paired = sorted(paired, key=lambda x: x[0])

        dt_list = [p[0] for p in paired]

        # =========================
        # date_now
        # =========================
        date_now = [(base - dt).days for dt in dt_list]
        date_now = date_now[:max_len]
        while len(date_now) < max_len:
            date_now.append(NonBinaryDataset.PADDING_VALUES.get("date", -1))
        date_now_list.append(date_now)

        # =========================
        # date_diff
        # =========================
        date_diff = []
        for j in range(1, len(dt_list)):
            date_diff.append((dt_list[j] - dt_list[j-1]).days)

        date_diff = date_diff[:max_len]
        while len(date_diff) < max_len:
            date_diff.append(NonBinaryDataset.PADDING_VALUES.get("date", -1))
        date_diff_list.append(date_diff)

        # =========================
        # 現在値
        # =========================
        try:
            current_jw = float(horse_conditions[i][0])
        except:
            current_jw = None

        try:
            current_hw = float(horse_conditions[i][1])
        except:
            current_hw = None

        jw_seq, hw_seq, f3_seq, pci_seq = [], [], [], []

        for _, rec in paired:
            try:
                jw_seq.append(float(rec[PastRaceFeatures.TIME_SCALE_IDX["jw"]]))
            except:
                jw_seq.append(None)

            try:
                hw_seq.append(float(rec[PastRaceFeatures.TIME_SCALE_IDX["hw"]]))
            except:
                hw_seq.append(None)

            try:
                f3_seq.append(float(rec[PastRaceFeatures.TIME_SCALE_IDX["final3f"]]))
            except:
                f3_seq.append(None)

            try:
                pci_seq.append(float(rec[PastRaceFeatures.TIME_SCALE_IDX["pci"]]))
            except:
                pci_seq.append(None)

        # =========================
        # jw_now / hw_now
        # =========================
        jw_now, hw_now = [], []

        for j in range(len(jw_seq)):
            jw_now.append(current_jw - jw_seq[j] if current_jw is not None and jw_seq[j] is not None else NonBinaryDataset.PADDING_VALUES.get("jw", 0))
            hw_now.append(current_hw - hw_seq[j] if current_hw is not None and hw_seq[j] is not None else NonBinaryDataset.PADDING_VALUES.get("hw", 0))

        jw_now = jw_now[:max_len]
        hw_now = hw_now[:max_len]

        while len(jw_now) < max_len:
            jw_now.append(NonBinaryDataset.PADDING_VALUES.get("jw", 0))
        while len(hw_now) < max_len:
            hw_now.append(NonBinaryDataset.PADDING_VALUES.get("hw", 0))

        jw_now_list.append(jw_now)
        hw_now_list.append(hw_now)

        # =========================
        # jw_diff / hw_diff / f3_diff / pci_diff
        # =========================
        jw_diff, hw_diff, f3_diff, pci_diff = [], [], [], []

        for j in range(1, len(jw_seq)):
            # jw
            if jw_seq[j] is not None and jw_seq[j-1] is not None:
                jw_diff.append(jw_seq[j] - jw_seq[j-1])
            else:
                jw_diff.append(NonBinaryDataset.PADDING_VALUES.get("jw", 0))

            # hw
            if hw_seq[j] is not None and hw_seq[j-1] is not None:
                hw_diff.append(hw_seq[j] - hw_seq[j-1])
            else:
                hw_diff.append(NonBinaryDataset.PADDING_VALUES.get("hw", 0))

            # f3
            if f3_seq[j] is not None and f3_seq[j-1] is not None:
                f3_diff.append(f3_seq[j] - f3_seq[j-1])
            else:
                f3_diff.append(NonBinaryDataset.PADDING_VALUES.get("final3f", 0))

            # pci
            if pci_seq[j] is not None and pci_seq[j-1] is not None:
                pci_diff.append(pci_seq[j] - pci_seq[j-1])
            else:
                pci_diff.append(NonBinaryDataset.PADDING_VALUES.get("pci", 0))

        # パディング
        for arr, key in [
            (jw_diff, "jw"),
            (hw_diff, "hw"),
            (f3_diff, "final3f"),
            (pci_diff, "pci"),
        ]:
            del arr[max_len:]
            while len(arr) < max_len:
                arr.append(NonBinaryDataset.PADDING_VALUES.get(key, 0))

        jw_diff_list.append(jw_diff)
        hw_diff_list.append(hw_diff)
        f3_diff_list.append(f3_diff)
        pci_diff_list.append(pci_diff)

    return (
        date_now_list, date_diff_list,
        jw_now_list, jw_diff_list,
        hw_now_list, hw_diff_list,
        f3_diff_list, pci_diff_list
    )

def Predict_Transformer(horse_name_list, horse_conditions, place, cb_score, base=None):
    if base is None:
        base = datetime.today()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = loadModel(place, device)
    feature_tuples = makePredictionInputFeature(
        horse_name_list,
        horse_conditions,
        base
    )
    x, time_mask, horse_mask = build_nn_input(feature_tuples, cb_score)
    x = x.unsqueeze(0).to(device)
    time_mask = time_mask.unsqueeze(0).to(device)
    horse_mask = horse_mask.unsqueeze(0).to(device)
    with torch.no_grad():
        scores, cat_scores, nn_scores = model(
            x,
            time_mask,
            horse_mask,
            return_parts=True
        )
        scores = scores.squeeze(0).cpu().numpy()
        cat_scores = cat_scores.squeeze(0).cpu().numpy()
        nn_scores = nn_scores.squeeze(0).cpu().numpy()
    horse_mask_np = horse_mask.squeeze(0).cpu().numpy()
    result = []
    for i, name in enumerate(horse_name_list):
        if horse_mask_np[i]:
            continue
        result.append({
            "horse_name": name,
            "score": float(scores[i])
        })
    result.sort(key=lambda x: x["score"], reverse=True)
    for i, r in enumerate(result):
        r["rank"] = i + 1
    results = [
        (r["horse_name"], r["score"], r["rank"])
        for r in result
    ]
    results_sorted = sorted(results, key=lambda x: x[2])
    print(f"{'Rank':<5} {'Horse':<20} {'Score':>10}")
    print("-" * 40)
    for name, score, rank in results_sorted:
        print(f"{rank:<5} {name:<20} {score:>10.4f}")
    print()
    print("Horse / CB / NN / Total")
    print("-" * 40)

    for i, name in enumerate(horse_name_list):

        if horse_mask_np[i]:
            continue

        print(
            f"{name:<20} "
            f"{cat_scores[i]:>8.4f} "
            f"{nn_scores[i]:>8.4f} "
            f"{scores[i]:>8.4f}"
        )
    return result

if __name__ == "__main__":
    Train_Transformer("中山")