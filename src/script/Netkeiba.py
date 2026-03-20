import requests
from bs4 import BeautifulSoup

session = requests.Session()

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
    "Referer": "https://race.netkeiba.com/"
}

def getHorseNames(url: str):
    url = url.split("&rf=")[0]

    try:
        res = session.get(url, headers=HEADERS, timeout=10)
        res.raise_for_status()

        # ★これ追加
        res.encoding = res.apparent_encoding

    except Exception as e:
        print(f"Warning: Failed to fetch page -> {url} ({e})")
        return None

    soup = BeautifulSoup(res.text, "html.parser")

    horse_names = []
    horse_tags = soup.select("span.HorseName > a")

    for a in horse_tags:
        horse_names.append(a.text.strip())

    return horse_names

def GetHorseNamesFromNetkeiba(race_id):
    url = f"https://race.netkeiba.com/race/shutuba.html?race_id={race_id}&rf=race_list"
    return getHorseNames(url)

if __name__ == "__main__":
    print(GetHorseNamesFromNetkeiba("202606020711"))