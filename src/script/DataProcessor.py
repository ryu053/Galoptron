from pathlib import Path
from dataclasses import dataclass, asdict
import csv
from tqdm import tqdm
import numpy as np
from datetime import datetime

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
    p1: str
    p2: str
    p3: str
    p4: str
    pci: str

    COLUMN_TYPES = {
        "distance": int,
        "time": float,
        "jw": float,
        "hw": int,
        "wc": int,
        "final3f": float,
        "p1": int,
        "p2": int,
        "p3": int,
        "p4": int,
        "pci": float
    }

    CATEGORY_COLUMN = [
        "surface",
        "place",
        "condition"
    ]

    MISSING_VALUES = ("", "---", "計不", None)

    TIME_SCALE_IDX = {
        "jw": 5,
        "hw": 6,
        "final3f": 8,
        "pci": 13
    }

    def to_list(self):
        return list(asdict(self).values())

    def to_typed_dict(self) -> dict:
        d = asdict(self)
        for col, val in d.items():
            if val in self.MISSING_VALUES:
                if col in self.COLUMN_TYPES:
                    d[col] = np.nan
                else:
                    d[col] = None
                continue
            if col in self.COLUMN_TYPES:
                typ = self.COLUMN_TYPES[col]
                try:
                    d[col] = typ(val)
                except (ValueError, TypeError):
                    d[col] = np.nan
            else:
                d[col] = val
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
            final3f=parts[29],
            p1=parts[24],
            p2=parts[25],
            p3=parts[26],
            p4=parts[27],
            pci=parts[13]
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
    alpha_conditions: list      # レースコンディション[クラス名, 芝・ダ, 距離]
    horse_conditions: list      # 馬のコンディション
    race_date_list: list        # 頭数n ×　過去レース数5
    dataset_race_date: str

    DEFAULT_SPEC_CONDITIONS = [None, None, None]
    PADDING_VALUES = {
        "date": -1,
        "jw": 0,
        "hw": 0,
        "final3f": 0,
        "pci": 0
    }

    def to_csv(self, file_name:Path):
        Path(file_name).parent.mkdir(parents=True, exist_ok=True)
        with open(file_name, "w", encoding="cp932") as f:
            print(*self.alpha_conditions, sep=",", file=f)
            for i, horse_name in enumerate(self.horse_name_list):
                f.write(horse_name + "," + self.finish_order_list[i] + ",")
                print(*self.horse_conditions[i], sep=",", file=f)
                for j, line in enumerate(self.past_record_list[i]):
                    print(*line, sep=",", file=f, end=",")
                    print(self.race_date_list[i][j], file=f)
                f.write("\n")
    
    def to_time_scale_dataset(self, max_len=5):
        base = datetime.strptime(self.dataset_race_date, "%Y%m%d")

        date_now_list, date_diff_list = [], []
        jw_now_list, jw_diff_list = [], []
        hw_now_list, hw_diff_list = [], []
        f3_diff_list, pci_diff_list = [], []

        for i in range(len(self.race_date_list)):
            dates = self.race_date_list[i] or []
            records = self.past_record_list[i] or []

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
                date_now.append(self.PADDING_VALUES.get("date", -1))
            date_now_list.append(date_now)

            # =========================
            # date_diff
            # =========================
            date_diff = []
            for j in range(1, len(dt_list)):
                date_diff.append((dt_list[j] - dt_list[j-1]).days)

            date_diff = date_diff[:max_len]
            while len(date_diff) < max_len:
                date_diff.append(self.PADDING_VALUES.get("date", -1))
            date_diff_list.append(date_diff)

            # =========================
            # 現在値
            # =========================
            try:
                current_jw = float(self.horse_conditions[i][0])
            except:
                current_jw = None

            try:
                current_hw = float(self.horse_conditions[i][1])
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
                jw_now.append(current_jw - jw_seq[j] if current_jw is not None and jw_seq[j] is not None else self.PADDING_VALUES.get("jw", 0))
                hw_now.append(current_hw - hw_seq[j] if current_hw is not None and hw_seq[j] is not None else self.PADDING_VALUES.get("hw", 0))

            jw_now = jw_now[:max_len]
            hw_now = hw_now[:max_len]

            while len(jw_now) < max_len:
                jw_now.append(self.PADDING_VALUES.get("jw", 0))
            while len(hw_now) < max_len:
                hw_now.append(self.PADDING_VALUES.get("hw", 0))

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
                    jw_diff.append(self.PADDING_VALUES.get("jw", 0))

                # hw
                if hw_seq[j] is not None and hw_seq[j-1] is not None:
                    hw_diff.append(hw_seq[j] - hw_seq[j-1])
                else:
                    hw_diff.append(self.PADDING_VALUES.get("hw", 0))

                # f3
                if f3_seq[j] is not None and f3_seq[j-1] is not None:
                    f3_diff.append(f3_seq[j] - f3_seq[j-1])
                else:
                    f3_diff.append(self.PADDING_VALUES.get("final3f", 0))

                # pci
                if pci_seq[j] is not None and pci_seq[j-1] is not None:
                    pci_diff.append(pci_seq[j] - pci_seq[j-1])
                else:
                    pci_diff.append(self.PADDING_VALUES.get("pci", 0))

            # パディング
            for arr, key in [
                (jw_diff, "jw"),
                (hw_diff, "hw"),
                (f3_diff, "final3f"),
                (pci_diff, "pci"),
            ]:
                del arr[max_len:]
                while len(arr) < max_len:
                    arr.append(self.PADDING_VALUES.get(key, 0))

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

    @classmethod
    def from_csv(cls, file_name):
        alpha_conditions = []
        horse_name_list = []
        finish_order_list = []
        horse_conditions = []
        past_record_list = []
        race_date_list = []
        with open(file_name, "r", encoding="cp932") as f:
            lines = [line.strip() for line in f]
        alpha_conditions = lines[0].split(",")
        i = 1
        while i < len(lines):
            if not lines[i]:
                i += 1
                continue
            parts = lines[i].split(",")
            horse_name = parts[0]
            finish_order = parts[1]
            horse_cond = parts[2:]
            horse_name_list.append(horse_name)
            finish_order_list.append(finish_order)
            horse_conditions.append(horse_cond)
            i += 1
            past_records = []
            race_dates = []
            while i < len(lines) and lines[i]:
                row = lines[i].split(",")
                past_records.append(row[:-1])
                race_dates.append(row[-1])
                i += 1
            past_record_list.append(past_records)
            race_date_list.append(race_dates)
        dataset_race_date = Path(file_name).name.split("_")[0]
        return cls(
            horse_name_list,
            finish_order_list,
            past_record_list,
            alpha_conditions,
            horse_conditions,
            race_date_list,
            dataset_race_date
        )
    
# 時系列データクラス
@dataclass
class TimeScaleData:
    delta_days: int
    delta_jw: float
    delta_hw: int
    delta_final3f: float
    delta_pci: float


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
    record_dates = []
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
            if int(parts[0]) < 50:
                record_date = "20" + parts[0].zfill(2) + parts[1].zfill(2) + parts[2].zfill(2)
            else:
                record_date = "19" + parts[0].zfill(2) + parts[1].zfill(2) + parts[2].zfill(2)
            record_dates.append(record_date)
            if len(results) >= 5:
                break
    return results, record_dates

# 会場と日付、レース番号を指定して、データセットに変換する関数
def createNonBinaryDatasetForRace(place:str, date:str, race_num:str, jvout_path:str):
    if not place in PLACE_NAMES.keys():
        print(f"Unknown place: {place}")
        return
    jv_date = tuple(map(str, [date[2:4], date[4:6], date[6:8]]))
    jv_date_int = tuple(map(int, jv_date))
    horse_name_list = []
    finish_order_list = []
    alpha_conditions = []
    horse_conditions = []
    race_date_list = []
    is_extracted = False
    lines_cnt = 0
    with open(jvout_path, "r", encoding="cp932") as f:
        for line in f:
            parts = line.strip().split(",")
            if (parts[0] == jv_date[0] and parts[1] == jv_date[1] and parts[2] == jv_date[2] and parts[4] == race_num):
                if lines_cnt == 0:
                    alpha_conditions.append(parts[6])
                    alpha_conditions.append(parts[7])
                    alpha_conditions.append(parts[8])
                    lines_cnt += 1
                is_extracted = True
                horse_name_list.append(parts[15])
                finish_order_list.append(parts[21])
                horse_conditions.append([parts[19], parts[30]])
            elif is_extracted == False:
                continue
            else:
                break
    past_record_list = []
    for horse_name in horse_name_list:
        past_record, record_dates = extractFeaturesFromRecord(horse_name, jv_date_int)
        past_record_list.append(past_record)
        race_date_list.append(record_dates)
    output_file_name = Path(DEFAULT_DATASET_DIR) / place / (date + "_" + race_num + ".csv")
    NonBinaryDataset(
        horse_name_list=horse_name_list,
        finish_order_list=finish_order_list,
        past_record_list=past_record_list,
        alpha_conditions=alpha_conditions,
        horse_conditions=horse_conditions,
        race_date_list=race_date_list,
        dataset_race_date=date
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

# 最新のレコードを読み込む
def LoadLatest5Records(horse_name: str) -> tuple:
    horse_name_initial = horse_name[0]
    record_path = Path(DEFAULT_RECORD_DIR) / horse_name_initial / (horse_name + ".csv")
    result = []
    race_dates = []
    try:
        with open(record_path, "r", encoding="cp932") as f:
            for line in f:
                result.append(PastRaceFeatures.from_line(line).to_list())
                parts = line.strip().split(",")
                if int(parts[0]) < 50:
                    record_date = "20" + parts[0].zfill(2) + parts[1].zfill(2) + parts[2].zfill(2)
                else:
                    record_date = "19" + parts[0].zfill(2) + parts[1].zfill(2) + parts[2].zfill(2)
                race_dates.append(record_date)
    except FileNotFoundError:
        print(f"Warning: File not found -> {record_path}")
        return None, None
    except Exception as e:
        print(f"Warning: Failed to read file -> {record_path} ({e})")
        return None, None
    return result, race_dates

# データセット全体のアップデート
def UpdateDataset():
    # updateRecords()
    # updateRaceSchedules()
    updateNonBinaryDataset()

if __name__ == "__main__":
    #UpdateDataset()
    nbd = NonBinaryDataset.from_csv(r"..\data\dataset\中山\20251207_1.csv")
    for l in nbd.to_time_scale_dataset():
        print(l)
    pass

