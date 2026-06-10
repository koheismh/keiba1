"""当日レースデータ取得モジュール。

netkeiba.comからレース当日の情報（出走馬、騎手、枠順、オッズ、馬体重、
天候、馬場状態）を取得し、構造化データとして返す。
"""

import logging
import time
from dataclasses import dataclass, field
from datetime import date

import requests
from bs4 import BeautifulSoup

from src.data.models import HorseEntry, RaceData, TrackCondition
from src.exceptions import DataFetchError

logger = logging.getLogger(__name__)


@dataclass
class FetchResult:
    """データ取得結果。

    成功したレースデータと、取得に失敗したレースの情報を保持する。
    """

    races: list[RaceData] = field(default_factory=list)
    failed_races: list[tuple[str, str]] = field(default_factory=list)
    # failed_races: (race_id or description, error message)


@dataclass
class OddsData:
    """直前オッズデータ。

    レース直前のオッズと馬体重変動情報を保持する。
    """

    race_id: str
    odds: dict[int, float] = field(default_factory=dict)
    # horse_number -> win_odds
    weight_changes: dict[int, int] = field(default_factory=dict)
    # horse_number -> weight_change (kg)


class RaceDataFetcher:
    """レース当日のデータを外部ソースから取得するクラス。

    netkeiba.comからレース情報をスクレイピングし、構造化データとして返す。
    データ取得失敗時はリトライ（最大3回）を行い、それでも失敗した場合は
    エラー情報を含むレスポンスを返す。
    """

    DEFAULT_TIMEOUT = 30  # seconds
    MAX_RETRIES = 3
    RETRY_DELAY = 1.0  # seconds between retries

    def __init__(
        self,
        base_url: str = "https://race.netkeiba.com",
        timeout: int = 30,
        data_store: "DataStore | None" = None,
    ) -> None:
        """初期化。

        Args:
            base_url: データ取得先のベースURL。
            timeout: HTTPリクエストのタイムアウト秒数。
            data_store: データ永続化ストア。指定時は取得データを自動保存し、
                保存済みデータがある場合はスクレイピングをスキップする。
        """
        from src.data.data_store import DataStore

        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.data_store = data_store
        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/120.0.0.0 Safari/537.36"
                ),
            }
        )

    def fetch_race_day(self, race_date: date) -> FetchResult:
        """指定日の全レース情報を取得する。

        data_storeが設定されている場合、まず保存済みデータを確認し、
        存在すればスクレイピングをスキップする。取得成功時はデータを
        自動的にディスクに保存する。

        Args:
            race_date: 取得対象のレース開催日。

        Returns:
            FetchResult: 成功したレースデータと失敗情報を含む結果。
        """
        # キャッシュからの読み込みを試行
        if self.data_store is not None and self.data_store.has_race_date(race_date):
            logger.info(f"保存済みデータを使用: {race_date}")
            cached_races = self.data_store.load_race_date(race_date)
            if cached_races:
                return FetchResult(races=cached_races)

        result = FetchResult()
        date_str = race_date.strftime("%Y%m%d")

        # レース一覧ページを取得
        race_list_url = f"{self.base_url}/top/race_list.html?kaisai_date={date_str}"
        try:
            race_ids = self._fetch_race_ids(race_list_url)
        except DataFetchError as e:
            logger.error(f"レース一覧の取得に失敗: {e}")
            result.failed_races.append((date_str, str(e)))
            return result

        # 各レースの詳細情報を取得
        for race_id in race_ids:
            try:
                race_data = self._fetch_single_race(race_id, race_date)
                result.races.append(race_data)
            except DataFetchError as e:
                logger.warning(f"レース {race_id} の取得に失敗: {e}")
                result.failed_races.append((race_id, str(e)))

        # 取得成功したデータを保存
        if self.data_store is not None and result.races:
            self.data_store.save_races(result.races, race_date)

        return result

    def fetch_realtime_odds(self, race_id: str) -> OddsData:
        """直前オッズおよび馬体重変動情報を取得する。

        レース発走直前のオッズと馬体重変動データを取得する。
        取得失敗時はリトライ（最大3回）を行い、それでも失敗した場合は
        DataFetchErrorを送出する。

        Args:
            race_id: 対象レースのID。

        Returns:
            OddsData: オッズと馬体重変動情報。

        Raises:
            DataFetchError: リトライ後も取得に失敗した場合。
        """
        odds_url = f"{self.base_url}/odds/index.html?race_id={race_id}"
        html = self._request_with_retry(odds_url, race_id)
        return self._parse_odds_page(race_id, html)

    def _fetch_race_ids(self, url: str) -> list[str]:
        """レース一覧ページからレースIDリストを取得する。

        Args:
            url: レース一覧ページのURL。

        Returns:
            レースIDのリスト。

        Raises:
            DataFetchError: 取得に失敗した場合。
        """
        html = self._request_with_retry(url, "race_list")
        return self._parse_race_list(html)

    def _fetch_single_race(self, race_id: str, race_date: date) -> RaceData:
        """単一レースの詳細情報を取得する。

        Args:
            race_id: レースID。
            race_date: レース開催日。

        Returns:
            RaceData: パース済みのレースデータ。

        Raises:
            DataFetchError: 取得またはパースに失敗した場合。
        """
        race_url = f"{self.base_url}/race/{race_id}"
        html = self._request_with_retry(race_url, race_id)
        return self._parse_race_page(race_id, race_date, html)

    def _request_with_retry(self, url: str, context_id: str) -> str:
        """リトライ付きHTTPリクエストを実行する。

        最大MAX_RETRIES回のリトライを行い、すべて失敗した場合は
        DataFetchErrorを送出する。タイムアウトはself.timeout秒。

        Args:
            url: リクエスト先URL。
            context_id: エラー報告用の識別子（レースIDなど）。

        Returns:
            レスポンスのHTMLテキスト。

        Raises:
            DataFetchError: 全リトライ失敗時。
        """
        last_error: Exception | None = None

        for attempt in range(1, self.MAX_RETRIES + 1):
            try:
                response = self.session.get(url, timeout=self.timeout)
                response.raise_for_status()
                response.encoding = response.apparent_encoding
                return response.text
            except requests.Timeout as e:
                last_error = e
                logger.warning(
                    f"タイムアウト (試行 {attempt}/{self.MAX_RETRIES}): {url}"
                )
            except requests.ConnectionError as e:
                last_error = e
                logger.warning(
                    f"接続エラー (試行 {attempt}/{self.MAX_RETRIES}): {url}"
                )
            except requests.HTTPError as e:
                last_error = e
                logger.warning(
                    f"HTTPエラー (試行 {attempt}/{self.MAX_RETRIES}): "
                    f"{e.response.status_code} {url}"
                )
            except requests.RequestException as e:
                last_error = e
                logger.warning(
                    f"リクエストエラー (試行 {attempt}/{self.MAX_RETRIES}): {url}"
                )

            if attempt < self.MAX_RETRIES:
                time.sleep(self.RETRY_DELAY * attempt)

        raise DataFetchError(
            race_id=context_id,
            message=f"全{self.MAX_RETRIES}回のリトライが失敗: {last_error}",
        )

    def _parse_race_list(self, html: str) -> list[str]:
        """レース一覧HTMLからレースIDリストをパースする。

        Args:
            html: レース一覧ページのHTML。

        Returns:
            レースIDのリスト。
        """
        soup = BeautifulSoup(html, "html.parser")
        race_ids: list[str] = []

        # netkeiba.comのレース一覧からレースIDを抽出
        for link in soup.select("a[href*='/race/']"):
            href = link.get("href", "")
            if isinstance(href, str) and "/race/" in href:
                # URLからレースIDを抽出 (例: /race/202401010101/)
                parts = href.rstrip("/").split("/")
                if parts and parts[-1].isdigit() and len(parts[-1]) >= 10:
                    race_id = parts[-1]
                    if race_id not in race_ids:
                        race_ids.append(race_id)

        return race_ids

    def _parse_race_page(
        self, race_id: str, race_date: date, html: str
    ) -> RaceData:
        """レース詳細HTMLをパースしてRaceDataを生成する。

        Args:
            race_id: レースID。
            race_date: レース開催日。
            html: レース詳細ページのHTML。

        Returns:
            RaceData: パース済みのレースデータ。

        Raises:
            DataFetchError: パースに失敗した場合。
        """
        try:
            soup = BeautifulSoup(html, "html.parser")

            # レース名を取得
            race_name_elem = soup.select_one(".RaceName")
            race_name = (
                race_name_elem.get_text(strip=True) if race_name_elem else "不明"
            )

            # レース情報（コース、距離、馬場状態、天候）を取得
            race_info = self._extract_race_info(soup)

            # 出走馬情報を取得
            entries = self._extract_entries(soup)

            return RaceData(
                race_id=race_id,
                race_name=race_name,
                race_date=race_date,
                post_time=None,
                venue=race_info.get("venue", "不明"),
                course_type=race_info.get("course_type", "芝"),
                distance=race_info.get("distance", 0),
                track_condition=race_info.get(
                    "track_condition", TrackCondition.FIRM
                ),
                weather=race_info.get("weather"),
                entries=entries,
                results=None,
                payouts=None,
            )
        except DataFetchError:
            raise
        except Exception as e:
            raise DataFetchError(
                race_id=race_id,
                message=f"レースページのパースに失敗: {e}",
            )

    def _extract_race_info(self, soup: BeautifulSoup) -> dict:
        """レース基本情報（コース種別、距離、馬場状態、天候）を抽出する。

        Args:
            soup: パース済みのBeautifulSoupオブジェクト。

        Returns:
            レース情報の辞書。
        """
        info: dict = {
            "venue": "不明",
            "course_type": "芝",
            "distance": 0,
            "track_condition": TrackCondition.FIRM,
            "weather": None,
        }

        # レース詳細情報の解析
        race_data_elem = soup.select_one(".RaceData01")
        if race_data_elem:
            text = race_data_elem.get_text()
            # コースタイプと距離
            if "ダ" in text:
                info["course_type"] = "ダート"
            elif "芝" in text:
                info["course_type"] = "芝"

            # 距離の数値抽出
            import re

            distance_match = re.search(r"(\d{3,4})m", text)
            if distance_match:
                info["distance"] = int(distance_match.group(1))

        # 馬場状態
        race_data2_elem = soup.select_one(".RaceData02")
        if race_data2_elem:
            text = race_data2_elem.get_text()
            condition_map = {
                "不良": TrackCondition.SOFT,
                "重": TrackCondition.YIELDING,
                "稍重": TrackCondition.GOOD,
                "良": TrackCondition.FIRM,
            }
            for condition_text, condition_enum in condition_map.items():
                if condition_text in text:
                    info["track_condition"] = condition_enum
                    break

            # 天候
            weather_map = ["晴", "曇", "雨", "小雨", "雪", "小雪"]
            for w in weather_map:
                if w in text:
                    info["weather"] = w
                    break

        return info

    def _extract_entries(self, soup: BeautifulSoup) -> list[HorseEntry]:
        """出走馬テーブルから全出走馬情報を抽出する。

        Args:
            soup: パース済みのBeautifulSoupオブジェクト。

        Returns:
            出走馬情報のリスト。
        """
        entries: list[HorseEntry] = []
        rows = soup.select("table.Shutuba_Table tr.HorseList")

        for row in rows:
            try:
                entry = self._parse_horse_row(row)
                if entry:
                    entries.append(entry)
            except (ValueError, IndexError):
                continue

        return entries

    def _parse_horse_row(self, row: BeautifulSoup) -> HorseEntry | None:
        """出走馬テーブルの1行をパースする。

        Args:
            row: テーブル行のBeautifulSoupオブジェクト。

        Returns:
            HorseEntry or None: パース成功時は馬情報、失敗時はNone。
        """
        tds = row.select("td")
        if len(tds) < 5:
            return None

        # 枠番
        gate_text = tds[0].get_text(strip=True)
        gate_number = int(gate_text) if gate_text.isdigit() else 0

        # 馬番
        horse_num_text = tds[1].get_text(strip=True)
        horse_number = int(horse_num_text) if horse_num_text.isdigit() else 0

        # 馬名
        horse_name_elem = row.select_one(".HorseName a")
        horse_name = (
            horse_name_elem.get_text(strip=True)
            if horse_name_elem
            else "不明"
        )

        # 騎手名
        jockey_elem = row.select_one(".Jockey a")
        jockey_name = (
            jockey_elem.get_text(strip=True) if jockey_elem else "不明"
        )

        # オッズ
        odds_elem = row.select_one(".Odds")
        win_odds: float | None = None
        if odds_elem:
            try:
                win_odds = float(odds_elem.get_text(strip=True))
            except ValueError:
                pass

        # 馬体重
        weight: int | None = None
        weight_change: int | None = None
        weight_elem = row.select_one(".Weight")
        if weight_elem:
            import re

            weight_text = weight_elem.get_text(strip=True)
            weight_match = re.match(r"(\d+)\(([+-]?\d+)\)", weight_text)
            if weight_match:
                weight = int(weight_match.group(1))
                weight_change = int(weight_match.group(2))

        return HorseEntry(
            horse_name=horse_name,
            jockey_name=jockey_name,
            gate_number=gate_number,
            horse_number=horse_number,
            weight=weight,
            weight_change=weight_change,
            win_odds=win_odds,
        )

    def _parse_odds_page(self, race_id: str, html: str) -> OddsData:
        """オッズページHTMLをパースしてOddsDataを生成する。

        Args:
            race_id: レースID。
            html: オッズページのHTML。

        Returns:
            OddsData: パース済みのオッズデータ。
        """
        soup = BeautifulSoup(html, "html.parser")
        odds_data = OddsData(race_id=race_id)

        # オッズテーブルをパース
        odds_rows = soup.select("table.Odds_Table tr")
        for row in odds_rows:
            tds = row.select("td")
            if len(tds) >= 2:
                try:
                    num_text = tds[0].get_text(strip=True)
                    odds_text = tds[1].get_text(strip=True)
                    if num_text.isdigit():
                        horse_number = int(num_text)
                        odds_data.odds[horse_number] = float(odds_text)
                except (ValueError, IndexError):
                    continue

        # 馬体重変動テーブルをパース
        weight_rows = soup.select("table.Weight_Table tr")
        for row in weight_rows:
            tds = row.select("td")
            if len(tds) >= 2:
                try:
                    import re

                    num_text = tds[0].get_text(strip=True)
                    change_text = tds[1].get_text(strip=True)
                    if num_text.isdigit():
                        horse_number = int(num_text)
                        change_match = re.search(r"([+-]?\d+)", change_text)
                        if change_match:
                            odds_data.weight_changes[horse_number] = int(
                                change_match.group(1)
                            )
                except (ValueError, IndexError):
                    continue

        return odds_data
