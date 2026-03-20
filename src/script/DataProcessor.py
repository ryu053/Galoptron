from pathlib import Path
from dataclasses import dataclass, asdict
import csv
from tqdm import tqdm

"""
外部IF一覧
- UpdateDataset(): 共通データセット全体のアップデート
"""

DEFAULT_JVOUT_DIR = r"..\data\jvout"
DEFAULT_RECORD_DIR = r"..\data\record"
DEFAULT_SCHEDULE_DIR = r"..\data\schedule"
DEFAULT_DATASET_DIR = r"..\data\dataset"
PLACE_NAMES = {
  '札幌': '札',
  '函館': '函',
  '福島': '福',
  '新潟': '新',
  '東京': '東',
  '中山': '中',
  '中京': '名',
  '京都': '京',
  '阪神': '阪',
  '小倉': '小'
}

# 特徴量として必要なデータをまとめたクラス
@dataclass
class PastRaceFeatures:
    surface: str        # 芝 or ダート
    place: str          # 競馬場
    distance: str       # 距離
    condition: str      # 馬場状態
    time: str           # 時間
    jw: str             # ジョッキー重量
    hw: str             # 馬体重
    wc: str             # ウェイト変化
    final3f: str        # 上がり3ハロン

    COLUMN_TYPES = {
        "distance": int,
        "time": float,
        "jw": float,
        "hw": int,
        "wc": int,
        "final3f": float,
    }

    def to_list(self):
        return [
            self.surface,
            self.place,
            self.distance,
            self.condition,
            self.time,
            self.jw,
            self.hw,
            self.wc,
            self.final3f
        ]
    
    def to_typed_dict(self) -> dict:
        d = asdict(self)
        for col, typ in self.COLUMN_TYPES.items():
            if col in d:
                val = d[col]
                if val in ("", "---", "計不"):
                    d[col] = None
                    continue
                try:
                    d[col] = typ(val)
                except ValueError:
                    d[col] = None
        return d
    
    @classmethod
    def from_line(cls, jvout_line:str):
        parts = jvout_line.strip().split(",")
        return cls(
            surface=parts[7],
            place=parts[3],
            distance=parts[8],
            condition=parts[9],
            time=parts[22],
            jw=parts[19],
            hw=parts[30],
            wc=parts[31],
            final3f=parts[29]
        )
    
    @classmethod
    def from_list(cls, past_one_record):
        return cls(*past_one_record)
    
    
# ノンバイナリデータセット
@dataclass
class NonBinaryDataset:
    horse_name_list: list       # 頭数n 次元
    finish_order_list: list     # 頭数n 次元
    past_record_list: list      # 頭数n × 過去レース数5 × 特徴量数m 次元

    def to_csv(self, file_name):
        Path(file_name).parent.mkdir(parents=True, exist_ok=True)
        with open(file_name, "w", encoding="cp932") as f:
            for i, horse_name in enumerate(self.horse_name_list):
                f.write(horse_name + "," + self.finish_order_list[i] + "\n")
                for line in self.past_record_list[i]:
                    print(*line, sep=",", file=f)
                f.write("\n")

    @classmethod
    def from_csv(cls, file_name):
        horse_name_list = []
        finish_order_list = []
        past_record_list = []
        with open(file_name, "r", encoding="cp932") as f:
            lines = [line.strip() for line in f]
        i = 0
        while i < len(lines):
            if not lines[i]:
                i += 1
                continue
            parts = lines[i].split(",")
            horse_name = parts[0]
            finish_order = parts[1]
            horse_name_list.append(horse_name)
            finish_order_list.append(finish_order)
            i += 1
            past_records = []
            while i < len(lines) and lines[i]:
                past_records.append(lines[i].split(","))
                i += 1
            past_record_list.append(past_records)
        return cls(horse_name_list, finish_order_list, past_record_list)

# 1レース単位の開催情報とjvoutのファイルを紐づけるクラス
@dataclass
class RaceSchedule:
    date: str
    place: str
    race_num: str
    jvout_path: str

    def to_list(self):
        return [
            self.date,
            self.place,
            self.race_num,
            self.jvout_path
        ]

# 馬ごとのレコードを作成する関数
def makeRecordsFromCsv(path):
    horse_name_list = set()
    with open(path, "r", encoding="cp932") as f:
        lines = f.readlines()
    for line in lines:
        parts = line.strip().split(",")
        horse_name = parts[15]
        horse_name_list.add(horse_name)
        horse_name_initial = horse_name[0] if horse_name else ""
        if (not horse_name_initial == ""):
            initial_folder = Path(DEFAULT_RECORD_DIR) / horse_name_initial
            if not initial_folder.exists():
                initial_folder.mkdir(parents=True, exist_ok=True)
            record_path = initial_folder / f"{horse_name}.csv"
            with open(record_path, "a", encoding="cp932") as rf:
                rf.write(line.strip() + "\n")
    record_list = []
    for horse_name in horse_name_list:
        horse_name_initial = horse_name[0]
        record_path = Path(DEFAULT_RECORD_DIR) / horse_name_initial / f"{horse_name}.csv"
        record_list.append(record_path)
    for record_path in record_list:
        with open(record_path, "r", encoding="cp932") as rf:
            rf_lines = rf.readlines()
        unique_sorted = sorted(
            set(line.strip() for line in rf_lines),
            key=lambda x: tuple(map(int, x.split(",")[:3])),
            reverse=True
        )
        with open(record_path, "w", encoding="cp932") as rf:
            for line in unique_sorted:
                rf.write(line.strip() + "\n")

# レコードの作成・更新
def updateRecords():
    if not Path(DEFAULT_RECORD_DIR).exists():
        Path(DEFAULT_RECORD_DIR).mkdir(parents=True, exist_ok=True)
    jvout_list = list(Path(DEFAULT_JVOUT_DIR).glob(r"*.csv"))
    for jvout in tqdm(jvout_list, desc="Update Records"):
        makeRecordsFromCsv(jvout)

# レースの開催日程とjvoutのファイルを紐づけたファイルを作成する関数
def updateRaceSchedules():
    if not Path(DEFAULT_SCHEDULE_DIR).exists():
        Path(DEFAULT_SCHEDULE_DIR).mkdir(parents=True, exist_ok=True)
    jvout_list = list(Path(DEFAULT_JVOUT_DIR).glob(r"*.csv"))
    for jvout in tqdm(jvout_list, desc="Update RaceSchedules"):
        output_key = None
        for place in PLACE_NAMES.keys():
            if jvout.name[3] == PLACE_NAMES[place]:
                output_key = place
                break
        if output_key is None:
            print(f"Unknown place in file name: {jvout.name}")
            continue
        race_set = set()
        with open(jvout, "r", encoding="cp932") as f:
            lines = f.readlines()
        for line in lines:
            parts = line.strip().split(",")
            if int(parts[0]) < 50:
                date = "20" + parts[0].zfill(2) + parts[1].zfill(2) + parts[2].zfill(2)
            else:
                date = "19" + parts[0].zfill(2) + parts[1].zfill(2) + parts[2].zfill(2)
            raceSchedule = RaceSchedule(
                date=date,
                place=parts[3],
                race_num=parts[4],
                jvout_path=str(jvout)
            ).to_list()
            race_set.add(tuple(raceSchedule))
        output_path = Path(DEFAULT_SCHEDULE_DIR) / f"{output_key}.csv"
        with open(output_path, "a", newline="", encoding="cp932") as f:
            writer = csv.writer(f)
            writer.writerows(race_set)
    schedule_files = list(Path(DEFAULT_SCHEDULE_DIR).glob(r"*.csv"))
    for schedule_file in tqdm(schedule_files, desc="→ Sorting"):
        with open(schedule_file, "r", encoding="cp932") as f:
            lines = f.readlines()
        unique_sorted = sorted(
            set(line.strip() for line in lines),
            key=lambda x: (
                int(x.split(",")[0]),  # 日付
                int(x.split(",")[2])   # レース番号
            ),
            reverse=True
        )
        with open(schedule_file, "w", encoding="cp932") as f:
            for line in unique_sorted:
                f.write(line.strip() + "\n")

# 過去5レース分の特徴量を取得する関数
def extractFeaturesFromRecord(horse_name, date:tuple):
    results = []
    horse_name_initial = horse_name[0]
    record_path = Path(DEFAULT_RECORD_DIR) / horse_name_initial / f"{horse_name}.csv"
    with open(record_path, "r", encoding="cp932") as rf:
        for line in rf:
            parts = line.strip().split(",")
            record_date = tuple(map(int, parts[:3]))
            if record_date >= date:
                continue
            result = PastRaceFeatures.from_line(line)
            results.append(result.to_list())
            if len(results) >= 5:
                break
    return results

# 会場と日付、レース番号を指定して、データセットに変換する関数
def createNonBinaryDatasetForRace(place:str, date:str, race_num:str, jvout_path:str):
    if not place in PLACE_NAMES.keys():
        print(f"Unknown place: {place}")
        return
    jv_date = tuple(map(str, [date[2:4], date[4:6], date[6:8]]))
    jv_date_int = tuple(map(int, jv_date))
    horse_name_list = []
    finish_order_list = []
    is_extracted = False
    with open(jvout_path, "r", encoding="cp932") as f:
        for line in f:
            parts = line.strip().split(",")
            if (parts[0] == jv_date[0] and parts[1] == jv_date[1] and parts[2] == jv_date[2] and parts[4] == race_num):
                is_extracted = True
                horse_name_list.append(parts[15])
                finish_order_list.append(parts[21])
            elif is_extracted == False:
                continue
            else:
                break
    past_record_list = []
    for horse_name in horse_name_list:
        past_record = extractFeaturesFromRecord(horse_name, jv_date_int)
        past_record_list.append(past_record)
    output_file_name = Path(DEFAULT_DATASET_DIR) / place / (date + "_" + race_num + ".csv")
    NonBinaryDataset(
        horse_name_list=horse_name_list,
        finish_order_list=finish_order_list,
        past_record_list=past_record_list
    ).to_csv(output_file_name)

# ノンバイナリデータセットのアップデート
def updateNonBinaryDataset():
    Path(DEFAULT_DATASET_DIR).mkdir(parents=True, exist_ok=True)
    schedule_files = list(Path(DEFAULT_SCHEDULE_DIR).glob(r"*.csv"))
    size = len(schedule_files)
    print("Update Non-Binary Dataset:")
    for i, file in enumerate(schedule_files):
        with open(file, "r", encoding="cp932") as f:
            lines = f.readlines()
            for line in tqdm(lines, desc=f"({i+1}/{size})"):
                parts = line.strip().split(",")
                place = parts[1]
                date = parts[0]
                race_num = parts[2]
                jvout_path = parts[3]
                createNonBinaryDatasetForRace(place, date, race_num, jvout_path)

# データセット全体のアップデート
def UpdateDataset():
    updateRecords()
    updateRaceSchedules()
    updateNonBinaryDataset()

if __name__ == "__main__":
    UpdateDataset()
    pass

