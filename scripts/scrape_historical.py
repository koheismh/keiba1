"""過去レースデータのスクレイピングスクリプト。

netkeiba.comの過去レース結果ページからデータを取得し、
data/raw/ にJSON形式で保存する。

サーバー負荷を抑えるため、リクエスト間に2〜3秒のスリープを入れる。

使用方法:
    python3 scripts/scrape_historical.py --year 2024 --months 1 2 3
    python3 scripts/scrape_historical.py --year 2024 --months 1 --venues 05 06
"""

import argparse
import json
import logging
import random
import re
import sys
import time
from datetime import date
from pathlib import Path

import requests
from bs4 import BeautifulSoup

# プロジェクトルートをパスに追加
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.models import BetType, HorseEntry, PayoutInfo, RaceData, RaceResult, TrackCondition

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# リクエスト間のスリープ（秒）- サーバーに迷惑をかけないため
MIN_SLEEP = 1.0
MAX_SLEEP = 1.5

# netkeiba.comの場所コード
VENUE_CODES = {
    "01": "札幌",
    "02": "函館",
    "03": "福島",
    "04": "新潟",
    "05": "東京",
    "06": "中山",
    "07": "中京",
    "08": "京都",
    "09": "阪神",
    "10": "小倉",
}


def create_session() -> requests.Session:
    """HTTP セッションを作成する。"""
    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            ),
            "Accept-Language": "ja,en;q=0.9",
        }
    )
    return session


def polite_sleep():
    """サーバーに優しいスリープ。"""
    sleep_time = random.uniform(MIN_SLEEP, MAX_SLEEP)
    time.sleep(sleep_time)


def fetch_page(session: requests.Session, url: str) -> str | None:
    """ページを取得する（リトライ付き）。"""
    for attempt in range(3):
        try:
            response = session.get(url, timeout=30)
            response.raise_for_status()
            response.encoding = "EUC-JP"
            return response.text
        except requests.RequestException as e:
            logger.warning(f"リクエスト失敗 (試行{attempt+1}/3): {url} - {e}")
            if attempt < 2:
                time.sleep(5 * (attempt + 1))
    return None


def get_race_ids_for_date(session: requests.Session, year: int, month: int, day: int) -> list[str]:
    """指定日のレースID一覧を取得する。"""
    date_str = f"{year:04d}{month:02d}{day:02d}"
    url = f"https://db.netkeiba.com/race/list/{date_str}/"
    
    html = fetch_page(session, url)
    if not html:
        return []
    
    soup = BeautifulSoup(html, "html.parser")
    race_ids = []
    
    for link in soup.select("a[href*='/race/']"):
        href = link.get("href", "")
        if isinstance(href, str):
            match = re.search(r"/race/(\d{12})/", href)
            if match:
                race_id = match.group(1)
                if race_id not in race_ids:
                    race_ids.append(race_id)
    
    return race_ids


def get_race_ids_for_month(session: requests.Session, year: int, month: int) -> list[str]:
    """指定月の全レースID一覧を取得する。

    netkeiba.comのカレンダーページから開催日を特定し、
    各日のレースIDを収集する。
    """
    # カレンダーページから開催日を取得
    url = f"https://db.netkeiba.com/race/list/?pid=race_top&date={year:04d}{month:02d}"
    html = fetch_page(session, url)
    
    all_race_ids = []
    
    if html:
        soup = BeautifulSoup(html, "html.parser")
        # 開催日リンクを探す
        dates_found = set()
        for link in soup.select("a[href*='/race/list/']"):
            href = link.get("href", "")
            if isinstance(href, str):
                match = re.search(r"/race/list/(\d{8})/", href)
                if match:
                    date_str = match.group(1)
                    if date_str.startswith(f"{year:04d}{month:02d}"):
                        dates_found.add(date_str)
        
        if dates_found:
            logger.info(f"  {year}年{month}月: {len(dates_found)}日の開催を検出")
            for date_str in sorted(dates_found):
                polite_sleep()
                day_ids = get_race_ids_for_date_str(session, date_str)
                all_race_ids.extend(day_ids)
                if day_ids:
                    logger.info(f"    {date_str}: {len(day_ids)}レース")
            return all_race_ids
    
    # フォールバック: 土日を中心に直接アクセス
    import calendar
    cal = calendar.monthcalendar(year, month)
    
    for week in cal:
        for day_idx in [5, 6, 0]:  # 土、日、月（祝日の場合）
            day = week[day_idx]
            if day == 0:
                continue
            polite_sleep()
            day_ids = get_race_ids_for_date(session, year, month, day)
            if day_ids:
                logger.info(f"    {year}/{month:02d}/{day:02d}: {len(day_ids)}レース")
                all_race_ids.extend(day_ids)
    
    return all_race_ids


def get_race_ids_for_date_str(session: requests.Session, date_str: str) -> list[str]:
    """日付文字列（YYYYMMDD）からレースID一覧を取得する。"""
    url = f"https://db.netkeiba.com/race/list/{date_str}/"
    html = fetch_page(session, url)
    if not html:
        return []
    
    soup = BeautifulSoup(html, "html.parser")
    race_ids = []
    
    for link in soup.select("a[href*='/race/']"):
        href = link.get("href", "")
        if isinstance(href, str):
            match = re.search(r"/race/(\d{12})/", href)
            if match:
                race_id = match.group(1)
                if race_id not in race_ids:
                    race_ids.append(race_id)
    
    return race_ids


def parse_race_page(html: str, race_id: str) -> dict | None:
    """レース結果ページをパースする。"""
    try:
        soup = BeautifulSoup(html, "html.parser")
        
        # レース名
        race_name_elem = soup.select_one(".racedata h1") or soup.select_one("h1")
        race_name = race_name_elem.get_text(strip=True) if race_name_elem else "不明"
        # 余分な情報を除去
        race_name = re.sub(r"\s+", " ", race_name).strip()
        
        # レース情報（距離、馬場状態、天候など）
        race_info = parse_race_info(soup)
        
        # 出走馬と結果
        entries, results = parse_result_table(soup)
        
        if not entries or not results:
            return None
        
        # 払い戻し情報
        payouts = parse_payouts(soup)
        
        # 日付の推定（race_idから）
        # race_id: YYYYVVKKDDRRNN (年4桁、場所2桁、開催2桁、日目2桁、レース番号2桁)
        year = int(race_id[:4])
        # 日付はレースIDから正確に特定できないので、ページから取得を試みる
        race_date_str = extract_race_date(soup, race_id)
        
        return {
            "race_id": race_id,
            "race_name": race_name,
            "race_date": race_date_str,
            "post_time": race_info.get("post_time"),
            "venue": race_info.get("venue", "不明"),
            "course_type": race_info.get("course_type", "芝"),
            "distance": race_info.get("distance", 0),
            "track_condition": race_info.get("track_condition", "良"),
            "weather": race_info.get("weather"),
            "entries": entries,
            "results": results,
            "payouts": payouts,
        }
    except Exception as e:
        logger.error(f"パースエラー (race_id={race_id}): {e}")
        return None


def extract_race_date(soup: BeautifulSoup, race_id: str) -> str:
    """レースの日付を抽出する。"""
    # ページ内の日付情報を探す
    # db.netkeiba.comでは<p class="smalltxt">に日付がある場合がある
    date_elem = soup.select_one("p.smalltxt")
    if date_elem:
        text = date_elem.get_text()
        match = re.search(r"(\d{4})年(\d{1,2})月(\d{1,2})日", text)
        if match:
            y, m, d = int(match.group(1)), int(match.group(2)), int(match.group(3))
            return f"{y:04d}-{m:02d}-{d:02d}"
    
    # diary_snap_cutから日付を探す
    for elem in soup.select("[class*='date'], .race_otherdata"):
        text = elem.get_text()
        match = re.search(r"(\d{4})年(\d{1,2})月(\d{1,2})日", text)
        if match:
            y, m, d = int(match.group(1)), int(match.group(2)), int(match.group(3))
            return f"{y:04d}-{m:02d}-{d:02d}"
    
    # race_idから推定（不正確だが最終手段）
    year = race_id[:4]
    return f"{year}-01-01"


def parse_race_info(soup: BeautifulSoup) -> dict:
    """レースの基本情報を解析する。"""
    info = {
        "venue": "不明",
        "course_type": "芝",
        "distance": 0,
        "track_condition": "良",
        "weather": None,
        "post_time": None,
    }
    
    # db.netkeiba.comでは <diary_snap_cut> や <p class="smalltxt"> に情報がある
    # smalltxtから会場情報を取得
    smalltxt = soup.select_one("p.smalltxt")
    if smalltxt:
        text = smalltxt.get_text()
        # "2024年01月06日 1回中山1日目" のような形式
        for code, name in VENUE_CODES.items():
            if name in text:
                info["venue"] = name
                break
    
    # レースのコース情報はspanに含まれる
    # "ダ右1200m / 天候 : 晴 / ダート : 良 / 発走 : 10:05"
    race_data_text = ""
    for span in soup.select("span"):
        text = span.get_text()
        if "m" in text and ("芝" in text or "ダ" in text or "障" in text):
            race_data_text = text
            break
    
    if not race_data_text:
        # diary_snap_cutの直下テキストを探す
        for elem in soup.select("dl.racedata, .mainrace_data, div"):
            text = elem.get_text()
            if re.search(r"(芝|ダ)\S*\d{3,4}m", text):
                race_data_text = text
                break
    
    if race_data_text:
        # コースタイプと距離
        # パターン: "芝右1200m", "ダ右1200m", "芝右 外1200m", "障芝3000m" など
        # スペースや他の文字を含む場合があるので柔軟にマッチ
        course_match = re.search(r"(芝|ダ|障).{0,10}?(\d{3,4})m", race_data_text)
        if not course_match:
            # フォールバック: 距離だけでも拾う
            dist_match = re.search(r"(\d{3,4})m", race_data_text)
            if dist_match:
                info["distance"] = int(dist_match.group(1))
                # コースタイプの推定
                if "ダ" in race_data_text:
                    info["course_type"] = "ダート"
                elif "障" in race_data_text:
                    info["course_type"] = "障害"
                else:
                    info["course_type"] = "芝"
        else:
            course_type = course_match.group(1)
            info["course_type"] = "ダート" if course_type == "ダ" else ("障害" if course_type == "障" else "芝")
            info["distance"] = int(course_match.group(2))
        
        # 馬場状態
        condition_match = re.search(r"(?:芝|ダート|障)\s*[:：]\s*(良|稍重|重|不良)", race_data_text)
        if condition_match:
            info["track_condition"] = condition_match.group(1)
        else:
            # フォールバック
            for cond in ["不良", "重", "稍重", "良"]:
                if cond in race_data_text:
                    info["track_condition"] = cond
                    break
        
        # 天候
        weather_match = re.search(r"天候\s*[:：]\s*(晴|曇|雨|小雨|雪|小雪)", race_data_text)
        if weather_match:
            info["weather"] = weather_match.group(1)
        
        # 発走時刻
        time_match = re.search(r"発走\s*[:：]\s*(\d{1,2}):(\d{2})", race_data_text)
        if time_match:
            info["post_time"] = f"{int(time_match.group(1)):02d}:{time_match.group(2)}"
    
    return info


def parse_result_table(soup: BeautifulSoup) -> tuple[list[dict], list[dict]]:
    """レース結果テーブルをパースする。"""
    entries = []
    results = []
    
    # 結果テーブルを探す
    table = soup.select_one("table.race_table_01")
    if not table:
        return entries, results
    
    # ヘッダーから列インデックスを特定
    header_row = table.select_one("tr")
    headers = [th.get_text(strip=True) for th in header_row.select("th")] if header_row else []
    
    # デフォルトインデックス（db.netkeiba.comの標準レイアウト）
    col_finish = 0
    col_gate = 1
    col_horse_num = 2
    col_horse_name = 3
    col_jockey = 6
    col_odds = 16
    col_weight = 18
    
    # ヘッダーからインデックスを動的に決定
    for i, h in enumerate(headers):
        if h == "着順":
            col_finish = i
        elif h == "枠番":
            col_gate = i
        elif h == "馬番":
            col_horse_num = i
        elif h == "馬名":
            col_horse_name = i
        elif h == "騎手":
            col_jockey = i
        elif h == "単勝":
            col_odds = i
        elif h == "馬体重":
            col_weight = i
    
    rows = table.select("tr")[1:]  # ヘッダー行をスキップ
    
    for row in rows:
        tds = row.select("td")
        if len(tds) < 7:
            continue
        
        try:
            # 着順
            finish_text = tds[col_finish].get_text(strip=True)
            if not finish_text.isdigit():
                continue  # 取消・除外等はスキップ
            finish_position = int(finish_text)
            
            # 枠番
            gate_text = tds[col_gate].get_text(strip=True)
            gate_number = int(gate_text) if gate_text.isdigit() else 0
            
            # 馬番
            horse_num_text = tds[col_horse_num].get_text(strip=True)
            horse_number = int(horse_num_text) if horse_num_text.isdigit() else 0
            
            # 馬名
            horse_name_elem = tds[col_horse_name].select_one("a")
            horse_name = horse_name_elem.get_text(strip=True) if horse_name_elem else tds[col_horse_name].get_text(strip=True)
            
            # 騎手名
            jockey_elem = tds[col_jockey].select_one("a")
            jockey_name = jockey_elem.get_text(strip=True) if jockey_elem else tds[col_jockey].get_text(strip=True)
            
            # 馬体重
            weight = None
            weight_change = None
            if col_weight < len(tds):
                weight_text = tds[col_weight].get_text(strip=True)
                weight_match = re.match(r"(\d+)\(([+-]?\d+)\)", weight_text)
                if weight_match:
                    weight = int(weight_match.group(1))
                    weight_change = int(weight_match.group(2))
            
            # 単勝オッズ
            win_odds = None
            if col_odds < len(tds):
                odds_text = tds[col_odds].get_text(strip=True)
                try:
                    win_odds = float(odds_text)
                except ValueError:
                    pass
            
            if horse_number > 0:
                entries.append({
                    "horse_name": horse_name,
                    "jockey_name": jockey_name,
                    "gate_number": gate_number,
                    "horse_number": horse_number,
                    "weight": weight,
                    "weight_change": weight_change,
                    "win_odds": win_odds,
                })
                results.append({
                    "horse_number": horse_number,
                    "finish_position": finish_position,
                })
        except (ValueError, IndexError) as e:
            continue
    
    return entries, results


def parse_payouts(soup: BeautifulSoup) -> list[dict]:
    """払い戻し情報をパースする。

    db.netkeiba.comでは複勝・ワイド等で複数の組み合わせが
    <br/>で区切られて1行に入っているため、分割して処理する。
    """
    payouts = []
    
    # 払い戻しテーブルを探す
    payout_tables = soup.select("table.pay_table_01")
    
    bet_type_map = {
        "単勝": "単勝",
        "複勝": "複勝",
        "枠連": None,  # 枠連はスキップ（システム未対応）
        "馬連": "馬連",
        "馬単": "馬単",
        "ワイド": "ワイド",
        "三連複": "三連複",
        "三連単": "三連単",
    }
    
    for table in payout_tables:
        rows = table.select("tr")
        
        for row in rows:
            th = row.select_one("th")
            tds = row.select("td")
            
            if not th or len(tds) < 2:
                continue
            
            bet_type_text = th.get_text(strip=True)
            current_bet_type = bet_type_map.get(bet_type_text)
            
            if current_bet_type is None:
                continue
            
            # <br/>で区切られた複数の組み合わせを分割
            combo_td = tds[0]
            payout_td = tds[1]
            
            # br要素で分割してテキストを取得
            combo_parts = _split_by_br(combo_td)
            payout_parts = _split_by_br(payout_td)
            
            # 各組み合わせと払い戻しをペアで処理
            for i in range(min(len(combo_parts), len(payout_parts))):
                combo_text = combo_parts[i].strip()
                payout_text = payout_parts[i].strip()
                
                if not combo_text or not payout_text:
                    continue
                
                try:
                    # 組み合わせの馬番を抽出
                    combo_text = combo_text.replace("→", "-").replace("－", "-").replace("―", "-")
                    combo_numbers = re.findall(r"\d+", combo_text)
                    if not combo_numbers:
                        continue
                    combination = [int(n) for n in combo_numbers]
                    
                    # 払い戻し金額
                    payout_text = payout_text.replace(",", "").replace("円", "").replace("¥", "")
                    payout_digits = re.sub(r"[^\d]", "", payout_text)
                    payout_amount = int(payout_digits) if payout_digits else 0
                    
                    if payout_amount > 0:
                        payouts.append({
                            "bet_type": current_bet_type,
                            "combination": combination,
                            "payout": payout_amount,
                        })
                except (ValueError, IndexError):
                    continue
    
    return payouts


def _split_by_br(td_element) -> list[str]:
    """td要素内のテキストを<br/>で分割する。"""
    # br要素を改行文字に置換してからテキスト取得
    for br in td_element.find_all("br"):
        br.replace_with("\n")
    text = td_element.get_text()
    parts = text.split("\n")
    return [p.strip() for p in parts if p.strip()]


def scrape_race(session: requests.Session, race_id: str) -> dict | None:
    """1レースのデータを取得・パースする。"""
    url = f"https://db.netkeiba.com/race/{race_id}/"
    html = fetch_page(session, url)
    if not html:
        return None
    return parse_race_page(html, race_id)


def save_races_by_date(races: list[dict], output_dir: Path) -> None:
    """レースデータを日付ごとにJSONファイルとして保存する。"""
    # 日付でグループ化
    by_date: dict[str, list[dict]] = {}
    for race in races:
        race_date = race["race_date"]
        if race_date not in by_date:
            by_date[race_date] = []
        by_date[race_date].append(race)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for date_str, date_races in sorted(by_date.items()):
        file_path = output_dir / f"{date_str}.json"
        
        # 既存ファイルがあればマージ
        existing = []
        if file_path.exists():
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    existing = json.load(f)
            except (json.JSONDecodeError, OSError):
                existing = []
        
        # 既存データとマージ（race_idで重複排除）
        existing_ids = {r["race_id"] for r in existing}
        for race in date_races:
            if race["race_id"] not in existing_ids:
                existing.append(race)
        
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(existing, f, ensure_ascii=False, indent=2)
        
        logger.info(f"保存: {file_path} ({len(existing)}レース)")


def main():
    parser = argparse.ArgumentParser(description="過去レースデータのスクレイピング")
    parser.add_argument("--year", type=int, required=True, help="対象年")
    parser.add_argument("--months", type=int, nargs="+", required=True, help="対象月（複数指定可）")
    parser.add_argument("--output", type=str, default="data/raw", help="出力ディレクトリ")
    parser.add_argument("--limit", type=int, default=0, help="取得レース数の上限（0=無制限）")
    args = parser.parse_args()
    
    output_dir = Path(args.output)
    session = create_session()
    
    all_races = []
    total_fetched = 0
    skipped = 0
    
    # 既存データのrace_idを収集（スキップ判定用）
    existing_race_ids = set()
    if output_dir.exists():
        for json_file in output_dir.glob("*.json"):
            try:
                with open(json_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, list):
                    for race in data:
                        if "race_id" in race:
                            existing_race_ids.add(race["race_id"])
            except (json.JSONDecodeError, OSError):
                pass
    
    logger.info(f"スクレイピング開始: {args.year}年 {args.months}月")
    logger.info(f"出力先: {output_dir}")
    logger.info(f"リクエスト間隔: {MIN_SLEEP}〜{MAX_SLEEP}秒")
    if existing_race_ids:
        logger.info(f"既存データ: {len(existing_race_ids)}レース（スキップ対象）")
    logger.info("")
    
    for month in args.months:
        logger.info(f"=== {args.year}年{month}月 ===")
        
        polite_sleep()
        race_ids = get_race_ids_for_month(session, args.year, month)
        
        if not race_ids:
            logger.warning(f"  レースが見つかりませんでした")
            continue
        
        logger.info(f"  合計 {len(race_ids)} レースを検出")
        
        for i, race_id in enumerate(race_ids):
            if args.limit > 0 and total_fetched >= args.limit:
                logger.info(f"上限 ({args.limit}) に達したため終了")
                break
            
            # 既存データはスキップ
            if race_id in existing_race_ids:
                skipped += 1
                continue
            
            polite_sleep()
            
            race_data = scrape_race(session, race_id)
            if race_data:
                all_races.append(race_data)
                existing_race_ids.add(race_id)
                total_fetched += 1
                if total_fetched % 10 == 0:
                    logger.info(f"  進捗: {total_fetched}レース取得済み "
                               f"(現在: {i+1}/{len(race_ids)}, スキップ: {skipped})")
                    # 10レースごとに中間保存
                    save_races_by_date(all_races, output_dir)
            else:
                logger.warning(f"  取得失敗: {race_id}")
        
        if args.limit > 0 and total_fetched >= args.limit:
            break
    
    # 最終保存
    if all_races:
        save_races_by_date(all_races, output_dir)
    
    logger.info("")
    logger.info(f"=== 完了 ===")
    logger.info(f"取得レース数: {total_fetched}")
    logger.info(f"スキップ数: {skipped}")
    logger.info(f"保存先: {output_dir}")


if __name__ == "__main__":
    main()
