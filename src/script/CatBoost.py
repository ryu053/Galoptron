import pandas as pd
import numpy as np
from pathlib import Path
from DataProcessor import PastRaceFeatures, NonBinaryDataset
from DataProcessor import DEFAULT_DATASET_DIR
from catboost import Pool, CatBoostRanker

MAX_PAST = 5

class BinaryDataset(NonBinaryDataset):
    def featureVector(self, race_id:str):
        target = self.setPredictionTarget()
        feature_vec = []
        for j, record_5 in enumerate(self.past_record_list):
            if np.isnan(target[j]):
                continue
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
            merged_dict.update({
                "num_past_races": num_past_races,
                "race_id": race_id,
                "target": target[j]
            })
            feature_vec.append(merged_dict)
        return feature_vec
    
    def setPredictionTarget(self):
        target = []
        for finish_order in self.finish_order_list:
            if finish_order == "0":
                target.append(np.nan)
            else:
                target.append(1.0 / int(finish_order))
        return target

# データセットのロード
def loadDatasets(place):
    dataset_dir = Path(DEFAULT_DATASET_DIR) / place
    if not dataset_dir.exists():
        raise FileNotFoundError(dataset_dir)
    dataset_list = sorted(dataset_dir.glob("*.csv"))
    all_features = []
    for dataset in dataset_list:
        bd = BinaryDataset.from_csv(dataset)
        feature_vec = bd.featureVector(dataset.stem)
        all_features.extend(feature_vec)
    df = pd.DataFrame(all_features)
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

def Train(place):
    train_pool, valid_pool = loadDatasets(place)
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
    model.save_model("catboost_ranker_model.cbm")


if __name__ == "__main__":
    Train("中山")