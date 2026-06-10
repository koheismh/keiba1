"""DataStore のユニットテスト"""

from datetime import date, time

import pytest

from src.data.data_store import DataStore
from src.data.models import (
    BetType,
    HorseEntry,
    PayoutInfo,
    RaceData,
    RaceResult,
    TrackCondition,
)


@pytest.fixture
def store(tmp_path) -> DataStore:
    """一時ディレクトリを使用するDataStoreを返す。"""
    return DataStore(data_dir=tmp_path / "raw")


@pytest.fixture
def sample_race() -> RaceData:
    """テスト用のレースデータを返す。"""
    return RaceData(
        race_id="202401010101",
        race_name="テストレース",
        race_date=date(2024, 1, 15),
        post_time=time(15, 30),
        venue="東京",
        course_type="芝",
        distance=2000,
        track_condition=TrackCondition.FIRM,
        weather="晴",
        entries=[
            HorseEntry(
                horse_name="テスト馬1",
                jockey_name="テスト騎手1",
                gate_number=1,
                horse_number=1,
                weight=480,
                weight_change=2,
                win_odds=3.5,
            ),
            HorseEntry(
                horse_name="テスト馬2",
                jockey_name="テスト騎手2",
                gate_number=2,
                horse_number=2,
                weight=460,
                weight_change=-4,
                win_odds=5.0,
            ),
        ],
        results=[
            RaceResult(horse_number=1, finish_position=1),
            RaceResult(horse_number=2, finish_position=2),
        ],
        payouts=[
            PayoutInfo(bet_type=BetType.WIN, combination=(1,), payout=350),
        ],
    )


@pytest.fixture
def sample_race_no_results() -> RaceData:
    """結果なしのレースデータを返す。"""
    return RaceData(
        race_id="202401010102",
        race_name="テストレース2",
        race_date=date(2024, 1, 15),
        post_time=None,
        venue="中山",
        course_type="ダート",
        distance=1200,
        track_condition=TrackCondition.YIELDING,
        weather=None,
        entries=[
            HorseEntry(
                horse_name="テスト馬3",
                jockey_name="テスト騎手3",
                gate_number=3,
                horse_number=3,
                weight=None,
                weight_change=None,
                win_odds=None,
            ),
            HorseEntry(
                horse_name="テスト馬4",
                jockey_name="テスト騎手4",
                gate_number=4,
                horse_number=4,
                weight=500,
                weight_change=0,
                win_odds=12.0,
            ),
        ],
        results=None,
        payouts=None,
    )


class TestSaveAndLoad:
    """保存と読み込みのテスト。"""

    def test_save_races(self, store: DataStore, sample_race: RaceData) -> None:
        path = store.save_races([sample_race], date(2024, 1, 15))
        assert path.exists()
        assert path.name == "2024-01-15.json"

    def test_load_after_save(self, store: DataStore, sample_race: RaceData) -> None:
        store.save_races([sample_race], date(2024, 1, 15))
        loaded = store.load_race_date(date(2024, 1, 15))
        assert len(loaded) == 1
        assert loaded[0].race_id == sample_race.race_id
        assert loaded[0].race_name == sample_race.race_name
        assert loaded[0].race_date == sample_race.race_date
        assert loaded[0].venue == sample_race.venue
        assert loaded[0].distance == sample_race.distance
        assert loaded[0].track_condition == sample_race.track_condition
        assert loaded[0].weather == sample_race.weather

    def test_entries_preserved(self, store: DataStore, sample_race: RaceData) -> None:
        store.save_races([sample_race], date(2024, 1, 15))
        loaded = store.load_race_date(date(2024, 1, 15))
        assert len(loaded[0].entries) == 2
        entry = loaded[0].entries[0]
        assert entry.horse_name == "テスト馬1"
        assert entry.jockey_name == "テスト騎手1"
        assert entry.gate_number == 1
        assert entry.horse_number == 1
        assert entry.weight == 480
        assert entry.weight_change == 2
        assert entry.win_odds == 3.5

    def test_results_preserved(self, store: DataStore, sample_race: RaceData) -> None:
        store.save_races([sample_race], date(2024, 1, 15))
        loaded = store.load_race_date(date(2024, 1, 15))
        assert loaded[0].results is not None
        assert len(loaded[0].results) == 2
        assert loaded[0].results[0].horse_number == 1
        assert loaded[0].results[0].finish_position == 1

    def test_payouts_preserved(self, store: DataStore, sample_race: RaceData) -> None:
        store.save_races([sample_race], date(2024, 1, 15))
        loaded = store.load_race_date(date(2024, 1, 15))
        assert loaded[0].payouts is not None
        assert len(loaded[0].payouts) == 1
        assert loaded[0].payouts[0].bet_type == BetType.WIN
        assert loaded[0].payouts[0].combination == (1,)
        assert loaded[0].payouts[0].payout == 350

    def test_post_time_preserved(self, store: DataStore, sample_race: RaceData) -> None:
        store.save_races([sample_race], date(2024, 1, 15))
        loaded = store.load_race_date(date(2024, 1, 15))
        assert loaded[0].post_time == time(15, 30)

    def test_none_fields_preserved(
        self, store: DataStore, sample_race_no_results: RaceData
    ) -> None:
        store.save_races([sample_race_no_results], date(2024, 1, 15))
        loaded = store.load_race_date(date(2024, 1, 15))
        assert loaded[0].post_time is None
        assert loaded[0].weather is None
        assert loaded[0].results is None
        assert loaded[0].payouts is None

    def test_multiple_races(
        self,
        store: DataStore,
        sample_race: RaceData,
        sample_race_no_results: RaceData,
    ) -> None:
        store.save_races([sample_race, sample_race_no_results], date(2024, 1, 15))
        loaded = store.load_race_date(date(2024, 1, 15))
        assert len(loaded) == 2


class TestCacheChecks:
    """キャッシュ判定のテスト。"""

    def test_has_race_date_false(self, store: DataStore) -> None:
        assert store.has_race_date(date(2024, 1, 15)) is False

    def test_has_race_date_true(self, store: DataStore, sample_race: RaceData) -> None:
        store.save_races([sample_race], date(2024, 1, 15))
        assert store.has_race_date(date(2024, 1, 15)) is True

    def test_has_race_false(self, store: DataStore) -> None:
        assert store.has_race(race_id="202401010101", race_date=date(2024, 1, 15)) is False

    def test_has_race_true(self, store: DataStore, sample_race: RaceData) -> None:
        store.save_races([sample_race], date(2024, 1, 15))
        assert store.has_race(race_id="202401010101", race_date=date(2024, 1, 15)) is True

    def test_has_race_wrong_id(self, store: DataStore, sample_race: RaceData) -> None:
        store.save_races([sample_race], date(2024, 1, 15))
        assert store.has_race(race_id="999999999999", race_date=date(2024, 1, 15)) is False


class TestSaveSingleRace:
    """単一レース追加保存のテスト。"""

    def test_add_to_empty(
        self, store: DataStore, sample_race: RaceData
    ) -> None:
        store.save_single_race(sample_race)
        loaded = store.load_race_date(date(2024, 1, 15))
        assert len(loaded) == 1

    def test_add_second_race(
        self,
        store: DataStore,
        sample_race: RaceData,
        sample_race_no_results: RaceData,
    ) -> None:
        store.save_single_race(sample_race)
        store.save_single_race(sample_race_no_results)
        loaded = store.load_race_date(date(2024, 1, 15))
        assert len(loaded) == 2

    def test_update_existing_race(
        self, store: DataStore, sample_race: RaceData
    ) -> None:
        store.save_single_race(sample_race)
        # 同じrace_idで異なるデータを保存
        updated = RaceData(
            race_id=sample_race.race_id,
            race_name="更新済みレース",
            race_date=sample_race.race_date,
            post_time=sample_race.post_time,
            venue=sample_race.venue,
            course_type=sample_race.course_type,
            distance=sample_race.distance,
            track_condition=sample_race.track_condition,
            weather=sample_race.weather,
            entries=sample_race.entries,
            results=sample_race.results,
            payouts=sample_race.payouts,
        )
        store.save_single_race(updated)
        loaded = store.load_race_date(date(2024, 1, 15))
        assert len(loaded) == 1
        assert loaded[0].race_name == "更新済みレース"


class TestGetStoredDates:
    """保存済み日付一覧のテスト。"""

    def test_empty(self, store: DataStore) -> None:
        assert store.get_stored_dates() == []

    def test_multiple_dates(
        self, store: DataStore, sample_race: RaceData
    ) -> None:
        race_jan = sample_race
        race_feb = RaceData(
            race_id="202402010101",
            race_name="2月レース",
            race_date=date(2024, 2, 1),
            post_time=None,
            venue="中山",
            course_type="芝",
            distance=1600,
            track_condition=TrackCondition.GOOD,
            weather=None,
            entries=sample_race.entries,
            results=None,
            payouts=None,
        )
        store.save_races([race_jan], date(2024, 1, 15))
        store.save_races([race_feb], date(2024, 2, 1))

        dates = store.get_stored_dates()
        assert dates == [date(2024, 1, 15), date(2024, 2, 1)]


class TestLoadNonexistent:
    """存在しないデータの読み込みテスト。"""

    def test_load_empty(self, store: DataStore) -> None:
        loaded = store.load_race_date(date(2024, 12, 31))
        assert loaded == []
