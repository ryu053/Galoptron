from CatBoost import Train_Catboost, Predict_Catboost, MultiplePredictionForTransFormer
from Transformer import Train_Transformer, Predict_Transformer
from Netkeiba import GetHorseDataFromNetkeiba
from datetime import datetime

def Full_Predict(race_id, place, train_start_idx, alpha_condition, date=None):
    if date is None:
        date = datetime.today()
    else:
        date = datetime.strptime(date, "%Y%m%d")

    # モデルのアップデート
    Train_Catboost(place, train_start_idx, alpha_condition)
    MultiplePredictionForTransFormer(place, alpha_condition)
    Train_Transformer(place)

    # 出馬表のロード
    horse_names, horse_conditions = GetHorseDataFromNetkeiba(race_id)

    # 予測
    cb_result = Predict_Catboost(horse_names, place)
    cb_score = [p for _, p in cb_result]
    Predict_Transformer(horse_names, horse_conditions, place, cb_score, date)

if __name__ == "__main__":
    Full_Predict(
        "202606020711",
        "中山",
        0,
        [None, "芝", "1800"],
        "20260321")
