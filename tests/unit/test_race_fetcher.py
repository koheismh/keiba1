"""RaceDataFetcherのユニットテスト。

外部HTTPリクエストはモックを使用してテストする。
リトライロジック、タイムアウト設定、エラーハンドリングを検証する。
"""

from datetime import date
from unittest.mock import MagicMock, patch

import pytest
import requests

from src.data.race_fetcher import FetchResult, OddsData, RaceDataFetcher
from src.exceptions import DataFetchError


class TestRaceDataFetcherInit:
    """初期化のテスト。"""

    def test_default_initialization(self):
        fetcher = RaceDataFetcher()
        assert fetcher.base_url == "https://race.netkeiba.com"
        assert fetcher.timeout == 30
        assert fetcher.MAX_RETRIES == 3

    def test_custom_initialization(self):
        fetcher = RaceDataFetcher(
            base_url="https://custom.example.com/",
            timeout=15,
        )
        assert fetcher.base_url == "https://custom.example.com"
        assert fetcher.timeout == 15

    def test_trailing_slash_removed(self):
        fetcher = RaceDataFetcher(base_url="https://example.com///")
        assert fetcher.base_url == "https://example.com"


class TestRequestWithRetry:
    """リトライロジックのテスト。"""

    def test_success_on_first_attempt(self):
        fetcher = RaceDataFetcher()
        mock_response = MagicMock()
        mock_response.text = "<html>OK</html>"
        mock_response.apparent_encoding = "utf-8"
        mock_response.raise_for_status = MagicMock()

        with patch.object(fetcher.session, "get", return_value=mock_response):
            result = fetcher._request_with_retry("http://test.com", "test_id")
            assert result == "<html>OK</html>"

    def test_success_after_retries(self):
        fetcher = RaceDataFetcher()
        fetcher.RETRY_DELAY = 0  # テストではウェイトなし

        mock_response = MagicMock()
        mock_response.text = "<html>OK</html>"
        mock_response.apparent_encoding = "utf-8"
        mock_response.raise_for_status = MagicMock()

        with patch.object(
            fetcher.session,
            "get",
            side_effect=[
                requests.Timeout("timeout"),
                requests.ConnectionError("connection error"),
                mock_response,
            ],
        ):
            with patch("src.data.race_fetcher.time.sleep"):
                result = fetcher._request_with_retry(
                    "http://test.com", "test_id"
                )
                assert result == "<html>OK</html>"

    def test_failure_after_max_retries(self):
        fetcher = RaceDataFetcher()
        fetcher.RETRY_DELAY = 0

        with patch.object(
            fetcher.session,
            "get",
            side_effect=requests.Timeout("timeout"),
        ):
            with patch("src.data.race_fetcher.time.sleep"):
                with pytest.raises(DataFetchError) as exc_info:
                    fetcher._request_with_retry(
                        "http://test.com", "race123"
                    )
                assert "race123" in str(exc_info.value)
                assert exc_info.value.race_id == "race123"

    def test_timeout_setting_used(self):
        fetcher = RaceDataFetcher(timeout=15)
        mock_response = MagicMock()
        mock_response.text = "<html></html>"
        mock_response.apparent_encoding = "utf-8"
        mock_response.raise_for_status = MagicMock()

        with patch.object(
            fetcher.session, "get", return_value=mock_response
        ) as mock_get:
            fetcher._request_with_retry("http://test.com", "test")
            mock_get.assert_called_once_with("http://test.com", timeout=15)

    def test_http_error_triggers_retry(self):
        fetcher = RaceDataFetcher()
        fetcher.RETRY_DELAY = 0

        http_error = requests.HTTPError(response=MagicMock(status_code=503))

        mock_response = MagicMock()
        mock_response.text = "<html>OK</html>"
        mock_response.apparent_encoding = "utf-8"
        mock_response.raise_for_status = MagicMock()

        with patch.object(
            fetcher.session,
            "get",
            side_effect=[http_error, mock_response],
        ):
            with patch("src.data.race_fetcher.time.sleep"):
                result = fetcher._request_with_retry(
                    "http://test.com", "test_id"
                )
                assert result == "<html>OK</html>"

    def test_default_timeout_30_seconds_in_request(self):
        """デフォルトタイムアウト30秒がHTTPリクエストに使用されることを検証。"""
        fetcher = RaceDataFetcher()  # デフォルト timeout=30
        mock_response = MagicMock()
        mock_response.text = "<html></html>"
        mock_response.apparent_encoding = "utf-8"
        mock_response.raise_for_status = MagicMock()

        with patch.object(
            fetcher.session, "get", return_value=mock_response
        ) as mock_get:
            fetcher._request_with_retry("http://test.com", "test")
            mock_get.assert_called_once_with("http://test.com", timeout=30)

    def test_connection_error_triggers_retry(self):
        """ネットワーク接続エラーでリトライされることを検証。"""
        fetcher = RaceDataFetcher()
        fetcher.RETRY_DELAY = 0

        mock_response = MagicMock()
        mock_response.text = "<html>OK</html>"
        mock_response.apparent_encoding = "utf-8"
        mock_response.raise_for_status = MagicMock()

        with patch.object(
            fetcher.session,
            "get",
            side_effect=[
                requests.ConnectionError("DNS resolution failed"),
                mock_response,
            ],
        ):
            with patch("src.data.race_fetcher.time.sleep"):
                result = fetcher._request_with_retry(
                    "http://test.com", "test_id"
                )
                assert result == "<html>OK</html>"

    def test_all_retries_exhausted_with_connection_error(self):
        """全リトライ失敗時にDataFetchErrorが送出されることを検証。"""
        fetcher = RaceDataFetcher()
        fetcher.RETRY_DELAY = 0

        with patch.object(
            fetcher.session,
            "get",
            side_effect=requests.ConnectionError("network unreachable"),
        ):
            with patch("src.data.race_fetcher.time.sleep"):
                with pytest.raises(DataFetchError) as exc_info:
                    fetcher._request_with_retry(
                        "http://test.com", "race456"
                    )
                assert exc_info.value.race_id == "race456"
                assert "全3回のリトライが失敗" in str(exc_info.value)


class TestFetchRaceDay:
    """fetch_race_dayメソッドのテスト。"""

    def test_returns_fetch_result(self):
        fetcher = RaceDataFetcher()
        with patch.object(
            fetcher, "_fetch_race_ids", return_value=[]
        ):
            result = fetcher.fetch_race_day(date(2024, 1, 1))
            assert isinstance(result, FetchResult)
            assert result.races == []
            assert result.failed_races == []

    def test_race_list_failure_returns_error(self):
        fetcher = RaceDataFetcher()
        with patch.object(
            fetcher,
            "_fetch_race_ids",
            side_effect=DataFetchError("race_list", "取得失敗"),
        ):
            result = fetcher.fetch_race_day(date(2024, 1, 1))
            assert result.races == []
            assert len(result.failed_races) == 1
            assert "20240101" in result.failed_races[0][0]

    def test_partial_failure_continues(self):
        fetcher = RaceDataFetcher()
        mock_race_data = MagicMock()

        with patch.object(
            fetcher, "_fetch_race_ids", return_value=["race1", "race2", "race3"]
        ):
            with patch.object(
                fetcher,
                "_fetch_single_race",
                side_effect=[
                    mock_race_data,
                    DataFetchError("race2", "取得失敗"),
                    mock_race_data,
                ],
            ):
                result = fetcher.fetch_race_day(date(2024, 1, 1))
                assert len(result.races) == 2
                assert len(result.failed_races) == 1
                assert result.failed_races[0][0] == "race2"


class TestFetchRealtimeOdds:
    """fetch_realtime_oddsメソッドのテスト。"""

    def test_returns_odds_data(self):
        fetcher = RaceDataFetcher()
        html = """
        <html>
        <table class="Odds_Table">
            <tr><td>1</td><td>3.5</td></tr>
            <tr><td>2</td><td>8.2</td></tr>
        </table>
        <table class="Weight_Table">
            <tr><td>1</td><td>+2</td></tr>
            <tr><td>2</td><td>-4</td></tr>
        </table>
        </html>
        """
        with patch.object(
            fetcher, "_request_with_retry", return_value=html
        ):
            result = fetcher.fetch_realtime_odds("race123")
            assert isinstance(result, OddsData)
            assert result.race_id == "race123"
            assert result.odds[1] == 3.5
            assert result.odds[2] == 8.2
            assert result.weight_changes[1] == 2
            assert result.weight_changes[2] == -4

    def test_raises_on_fetch_failure(self):
        fetcher = RaceDataFetcher()
        with patch.object(
            fetcher,
            "_request_with_retry",
            side_effect=DataFetchError("race123", "取得失敗"),
        ):
            with pytest.raises(DataFetchError):
                fetcher.fetch_realtime_odds("race123")

    def test_correct_url_constructed(self):
        """fetch_realtime_oddsが正しいURLを構築することを検証。"""
        fetcher = RaceDataFetcher(base_url="https://race.netkeiba.com")
        html = "<html><table class='Odds_Table'></table></html>"
        with patch.object(
            fetcher, "_request_with_retry", return_value=html
        ) as mock_req:
            fetcher.fetch_realtime_odds("202401010101")
            mock_req.assert_called_once_with(
                "https://race.netkeiba.com/odds/index.html?race_id=202401010101",
                "202401010101",
            )

    def test_empty_odds_table(self):
        """オッズテーブルが空でもエラーにならないことを検証。"""
        fetcher = RaceDataFetcher()
        html = "<html><body>No odds data</body></html>"
        with patch.object(
            fetcher, "_request_with_retry", return_value=html
        ):
            result = fetcher.fetch_realtime_odds("race123")
            assert isinstance(result, OddsData)
            assert result.odds == {}
            assert result.weight_changes == {}


class TestParseRaceList:
    """_parse_race_listメソッドのテスト。"""

    def test_extracts_race_ids(self):
        fetcher = RaceDataFetcher()
        html = """
        <html>
        <a href="/race/202401010101/">1R</a>
        <a href="/race/202401010102/">2R</a>
        <a href="/race/202401010103/">3R</a>
        </html>
        """
        ids = fetcher._parse_race_list(html)
        assert ids == ["202401010101", "202401010102", "202401010103"]

    def test_deduplicates_race_ids(self):
        fetcher = RaceDataFetcher()
        html = """
        <html>
        <a href="/race/202401010101/">1R</a>
        <a href="/race/202401010101/">1R again</a>
        </html>
        """
        ids = fetcher._parse_race_list(html)
        assert ids == ["202401010101"]

    def test_empty_page(self):
        fetcher = RaceDataFetcher()
        html = "<html><body></body></html>"
        ids = fetcher._parse_race_list(html)
        assert ids == []


class TestParseRacePage:
    """_parse_race_pageメソッドのテスト。"""

    def test_parses_race_data(self):
        fetcher = RaceDataFetcher()
        html = """
        <html>
        <span class="RaceName">テストレース</span>
        <div class="RaceData01">芝1600m</div>
        <div class="RaceData02">天候:晴 馬場:良</div>
        <table class="Shutuba_Table">
            <tr class="HorseList">
                <td>1</td>
                <td>1</td>
                <td><span class="HorseName"><a>テスト馬</a></span></td>
                <td><span class="Jockey"><a>テスト騎手</a></span></td>
                <td><span class="Odds">5.0</span></td>
                <td><span class="Weight">480(+2)</span></td>
            </tr>
        </table>
        </html>
        """
        result = fetcher._parse_race_page("race123", date(2024, 1, 1), html)
        assert result.race_id == "race123"
        assert result.race_name == "テストレース"
        assert result.course_type == "芝"
        assert result.distance == 1600

    def test_raises_on_parse_error(self):
        fetcher = RaceDataFetcher()
        # None is not valid HTML input for BeautifulSoup but won't cause parse failure
        # An exception during attribute access would cause DataFetchError
        with patch(
            "src.data.race_fetcher.BeautifulSoup",
            side_effect=Exception("parse error"),
        ):
            with pytest.raises(DataFetchError):
                fetcher._parse_race_page("race123", date(2024, 1, 1), "bad")


class TestFetchResultDataclass:
    """FetchResult データクラスのテスト。"""

    def test_default_values(self):
        result = FetchResult()
        assert result.races == []
        assert result.failed_races == []

    def test_with_data(self):
        result = FetchResult(
            races=[MagicMock()],
            failed_races=[("race1", "error")],
        )
        assert len(result.races) == 1
        assert len(result.failed_races) == 1


class TestOddsDataDataclass:
    """OddsData データクラスのテスト。"""

    def test_default_values(self):
        data = OddsData(race_id="test")
        assert data.race_id == "test"
        assert data.odds == {}
        assert data.weight_changes == {}

    def test_with_data(self):
        data = OddsData(
            race_id="race123",
            odds={1: 3.5, 2: 8.0},
            weight_changes={1: 2, 2: -4},
        )
        assert data.odds[1] == 3.5
        assert data.weight_changes[2] == -4
