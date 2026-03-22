import pandas as pd
import numpy as np
from pathlib import Path
from DataProcessor import PastRaceFeatures, NonBinaryDataset, LoadLatest5Records
from DataProcessor import DEFAULT_DATASET_DIR
from catboost import Pool, CatBoostRanker
from Netkeiba import GetHorseNamesFromNetkeiba
from tqdm import tqdm
import shutil

DEFAULT_PREDICTION_OUTPUT_DIR = r"..\data\catboost"
MAX_PAST = 5
MODEL_NAME = {
    "中山": "catboost_ranker_model.cbm"
}

def record5ToMergedDict(record_5):
    dict_list = []
    padded_records = list(record_5[:MAX_PAST])
    num_past_races = len(padded_records)
    while len(padded_records) < MAX_PAST:
        padded_records.append(None)
    for record in padded_records:
        if record is None:
            features_dict = {
                col: (np.nan if col in PastRaceFeatures.COLUMN_TYPES else None)
                for col in PastRaceFeatures.__dataclass_fields__.keys()
            }
        else:
            features = PastRaceFeatures.from_list(record)
            features_dict = features.to_typed_dict()
        dict_list.append(features_dict)
    merged_dict = {
        f"{k}_{i+1}": v
        for i, d in enumerate(dict_list)
        for k, v in d.items()
    }
    return merged_dict, num_past_races

class BinaryDataset(NonBinaryDataset):
    def featureVector(self, race_id:str):
        target = self.setPredictionTarget()
        feature_vec = []
        for j, record_5 in enumerate(self.past_record_list):
            if np.isnan(target[j]):
                continue
            merged_dict, num_past_races = record5ToMergedDict(record_5)
            merged_dict.update({
                "num_past_races": num_past_races,
                "race_id": race_id,
                "target": target[j]
            })
            feature_vec.append(merged_dict)
        return feature_vec
    
    def featureVectorForPredict(self):
        feature_vec = []
        for j, record_5 in enumerate(self.past_record_list):
            merged_dict, num_past_races = record5ToMergedDict(record_5)
            merged_dict.update({"num_past_races": num_past_races})
            feature_vec.append(merged_dict)
        return feature_vec, self.horse_name_list
    
    def setPredictionTarget(self):
        target = []
        for finish_order in self.finish_order_list:
            if finish_order == "0":
                target.append(np.nan)
            else:
                target.append(1.0 / int(finish_order))
        return target
    
def checkConditions(data_conditions, query_conditions):
    for dc, qc in zip(data_conditions, query_conditions):
        if qc is None:
            continue
        if dc != qc:
            return False
    return True

# データセットのロード
def loadDatasets(place, start_index, conditions):
    dataset_dir = Path(DEFAULT_DATASET_DIR) / place
    if not dataset_dir.exists():
        raise FileNotFoundError(dataset_dir)
    dataset_list = sorted(dataset_dir.glob("*.csv"))
    all_features = []
    print("Training starts at " + str(dataset_list[start_index]))
    for dataset in dataset_list[start_index:]:
        bd = BinaryDataset.from_csv(dataset)
        condition_check = checkConditions(bd.alpha_conditions, conditions)
        if condition_check == False:
            continue
        feature_vec = bd.featureVector(dataset.stem)
        all_features.extend(feature_vec)
    df = pd.DataFrame(all_features)
    print(df.shape)
    race_ids = sorted(df["race_id"].unique())
    split = int(len(race_ids) * 0.8)
    train_ids = race_ids[:split]
    valid_ids = race_ids[split:]
    train_df = df[df["race_id"].isin(train_ids)].sort_values("race_id")
    valid_df = df[df["race_id"].isin(valid_ids)].sort_values("race_id")
    train_df["race_id"] = train_df["race_id"].astype("category").cat.codes
    valid_df["race_id"] = valid_df["race_id"].astype("category").cat.codes
    X_train = train_df.drop(columns=["target", "race_id"])
    y_train = train_df["target"]
    group_train = train_df["race_id"]
    X_valid = valid_df.drop(columns=["target", "race_id"])
    y_valid = valid_df["target"]
    group_valid = valid_df["race_id"]
    cat_cols = [col for col in X_train.columns 
                if any(k in col for k in PastRaceFeatures.CATEGORY_COLUMN)]
    for col in cat_cols:
        X_train[col] = X_train[col].fillna("missing").astype(str)
        X_valid[col] = X_valid[col].fillna("missing").astype(str)
    train_pool = Pool(
        X_train,
        y_train,
        group_id=group_train,
        cat_features=cat_cols
    )
    valid_pool = Pool(
        X_valid,
        y_valid,
        group_id=group_valid,
        cat_features=cat_cols
    )
    return train_pool, valid_pool

def Train_Catboost(place, start_index, spec_conditions=NonBinaryDataset.DEFAULT_SPEC_CONDITIONS):
    train_pool, valid_pool = loadDatasets(place, start_index, spec_conditions)
    model = CatBoostRanker(
        iterations=1000,
        learning_rate=0.05,
        depth=6,
        loss_function="YetiRank",   # ランキング用
        eval_metric="NDCG",
        early_stopping_rounds=50,
        verbose=100,
        task_type="GPU",
        devices="0"
    )
    model.fit(train_pool, eval_set=valid_pool)
    model.save_model(MODEL_NAME[place])

def makePredictionInputFeature(horse_name_list):
    feature_vec = []
    valid_horses = []
    for horse_name in horse_name_list:
        record_5, _ = LoadLatest5Records(horse_name)
        if record_5 == None:
            continue
        merged_dict, num_past_races = record5ToMergedDict(record_5)
        merged_dict.update({"num_past_races": num_past_races})
        feature_vec.append(merged_dict)
        valid_horses.append(horse_name)
    x = pd.DataFrame(feature_vec)
    cat_cols = [col for col in x.columns
                if any(k in col for k in PastRaceFeatures.CATEGORY_COLUMN)]
    for col in cat_cols:
        x[col] = x[col].fillna("missing").astype(str)
    return x, valid_horses

def Predict_Catboost(horse_name_list, place):
    x, valid_horses = makePredictionInputFeature(horse_name_list)
    model = CatBoostRanker()
    model.load_model(MODEL_NAME[place])
    pred = model.predict(x)
    order = sorted(range(len(pred)), key=lambda i: pred[i], reverse=True)
    rank = [0] * len(pred)
    for r, i in enumerate(order):
        rank[i] = r + 1
    results = list(zip(valid_horses, pred, rank))
    results_sorted = sorted(results, key=lambda x: x[2])
    print(f"{'Rank':<5} {'Horse':<20} {'Score':>10}")
    print("-" * 40)
    for name, score, rank in results_sorted:
        print(f"{rank:<5} {name:<20} {score:>10.4f}")
    return list(zip(valid_horses, pred))

def MultiplePredictionForTransFormer(place, conditions):
    shutil.rmtree(Path(DEFAULT_PREDICTION_OUTPUT_DIR) / place)
    (Path(DEFAULT_PREDICTION_OUTPUT_DIR) / place).mkdir(parents=True, exist_ok=True)
    model = CatBoostRanker()
    model.load_model(MODEL_NAME[place])
    dataset_dir = Path(DEFAULT_DATASET_DIR) / place
    if not dataset_dir.exists():
        raise FileNotFoundError(dataset_dir)
    dataset_list = sorted(dataset_dir.glob("*.csv"))
    for dataset in tqdm(dataset_list, desc="Prediction"):
        bd = BinaryDataset.from_csv(dataset)
        condition_check = checkConditions(bd.alpha_conditions, conditions)
        if condition_check == False:
            continue
        feature_vec, horse_name_list = bd.featureVectorForPredict()
        x = pd.DataFrame(feature_vec)
        cat_cols = [col for col in x.columns
                    if any(k in col for k in PastRaceFeatures.CATEGORY_COLUMN)]
        for col in cat_cols:
            x[col] = x[col].fillna("missing").astype(str)
        pred = model.predict(x)
        result = list(zip(horse_name_list, pred))
        df = pd.DataFrame(result, columns=["horse_name", "score"])
        df.to_parquet(Path(DEFAULT_PREDICTION_OUTPUT_DIR) / place / (dataset.stem + ".parquet"))

if __name__ == "__main__":
    Train_Catboost("中山", 0, [None, "芝", "1800"])
    horse_name_list = GetHorseNamesFromNetkeiba("202606020711")
    Predict_Catboost(horse_name_list, "中山")
    # MultiplePredictionForTransFormer("中山", [None, "芝", "1800"])
    pass