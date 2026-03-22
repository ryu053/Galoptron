import re
import requests
from bs4 import BeautifulSoup

session = requests.Session()

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
    "Referer": "https://race.netkeiba.com/"
}

def getHorseData(url: str):
    url = url.split("&rf=")[0]

    try:
        res = session.get(url, headers=HEADERS, timeout=10)
        res.raise_for_status()
        res.encoding = res.apparent_encoding
    except Exception as e:
        print(f"Warning: Failed to fetch page -> {url} ({e})")
        return None, None

    soup = BeautifulSoup(res.text, "html.parser")

    horse_names = []
    horse_conditions = []

    rows = soup.select("tr.HorseList")

    for row in rows:
        # 馬名
        a = row.select_one("span.HorseName > a")
        if a is None:
            continue
        horse_name = a.text.strip()

        # 斤量
        weight_carried = None
        barei_td = row.select_one("td.Barei")
        if barei_td is not None:
            tds = row.find_all("td", recursive=False)
            try:
                idx = tds.index(barei_td)
                if idx + 1 < len(tds):
                    text = tds[idx + 1].get_text(strip=True)
                    try:
                        weight_carried = float(text)
                    except ValueError:
                        weight_carried = text
            except ValueError:
                pass

        # 馬体重
        horse_weight = None
        weight_td = row.select_one("td.Weight")
        if weight_td is not None:
            weight_text = weight_td.get_text(" ", strip=True)
            m = re.search(r"\d+", weight_text)
            if m:
                horse_weight = int(m.group())

        horse_names.append(horse_name)
        horse_conditions.append([weight_carried, horse_weight])

    return horse_names, horse_conditions


def getHorseNames(url: str):
    horse_names, _ = getHorseData(url)
    return horse_names


def GetHorseNamesFromNetkeiba(race_id):
    url = f"https://race.netkeiba.com/race/shutuba.html?race_id={race_id}&rf=race_list"
    return getHorseNames(url)


def GetHorseDataFromNetkeiba(race_id):
    url = f"https://race.netkeiba.com/race/shutuba.html?race_id={race_id}&rf=race_list"
    return getHorseData(url)


if __name__ == "__main__":
    horse_names, horse_conditions = GetHorseDataFromNetkeiba("202606020711")

    print(horse_names)
    print(horse_conditions)