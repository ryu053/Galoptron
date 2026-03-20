import pandas as pd
from pathlib import Path
from DataProcessor import PastRaceFeatures, NonBinaryDataset
from DataProcessor import DEFAULT_DATASET_DIR

class BinaryDataset(NonBinaryDataset):
    def featureVector(self, race_id:str):
        target = self.setPredictionTarget()
        feature_vec = []
        for j, record_5 in enumerate(self.past_record_list):
            dict_list = []
            for record in record_5:
                features = PastRaceFeatures.from_list(record)
                features_dict = features.to_typed_dict()
                dict_list.append(features_dict)
            merged_dict = {
                f"{k}_{i+1}": v
                for i, d in enumerate(dict_list)
                for k, v in d.items()
            }
            merged_dict.update({"race_id": race_id, "target": target[j]})
            feature_vec.append(merged_dict)
        return feature_vec
    
    def setPredictionTarget(self):
        target = []
        for finish_order in self.finish_order_list:
            if finish_order == "1":
                target.append(1)
            else:
                target.append(0)
        return target


if __name__ == "__main__":
    bd = BinaryDataset.from_csv(DEFAULT_DATASET_DIR + r"\中山\20251228_11.csv")
    bd.featureVector("20251228_11")