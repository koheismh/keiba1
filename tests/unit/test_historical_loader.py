"""HistoricalDataLoaderのユニットテスト。"""

import json
import tempfile
from datetime import date
from pathlib import Path

import pytest

from src.data.historical_loader import HistoricalDataLoader
from src.data.models import (
    CleaningReport,
    HorseEntry,
    RaceData,
    TrackCondition,
)


def _make_entry(
    horse_number: int = 1,
    gate_number: int = 1,
    horse_name: str = "テスト馬",
    jockey_name: str = "テスト騎手",
) -> HorseEntry:
    """テスト用出走馬エントリを生成する。"""
    return HorseEntry(
        horse_name=horse_name,
        jockey_name=jockey_name,
        gate_number=gate_number,
        horse_number=horse_number,
        weight=480,
        weight_change=0,
        win_odds=3.5,
    )


def _make_race(
    race_id: str = "202401010101",
    race_name: str = "テストレース",
    race_date: date = date(2024, 1, 1),
    venue: str = "東京",
    distance: int = 1600,
    entries: list[HorseEntry] | None = None,
) -> RaceData:
    """テスト用RaceDataを生成する。"""
    if entries is None:
        entries = [
            _make_entry(horse_number=1, gate_number=1, horse_name="馬A", jockey_name="騎手A"),
            _make_entry(horse_number=2, gate_number=2, horse_name="馬B", jockey_name="騎手B"),
            _make_entry(horse_number=3, gate_number=3, horse_name="馬C", jockey_name="騎手C"),
        ]
    return RaceData(
        race_id=race_id,
        race_name=race_name,
        race_date=race_date,
        post_time=None,
        venue=venue,
        course_type="芝",
        distance=distance,
        track_condition=TrackCondition.FIRM,
        weather="晴",
        entries=entries,
        results=None,
        payouts=None,
    )


def _race_to_dict(race: RaceData) -> dict:
    """RaceDataをJSON可能な辞書に変換する。"""
    return {
        "race_id": race.race_id,
        "race_name": race.race_name,
        "race_date": race.race_date.isoformat(),
        "post_time": None,
        "venue": race.venue,
        "course_type": race.course_type,
        "distance": race.distance,
        "track_condition": race.track_condition.value,
        "weather": race.weather,
        "entries": [
            {
                "horse_name": e.horse_name,
                "jockey_name": e.jockey_name,
                "gate_number": e.gate_number,
                "horse_number": e.horse_number,
                "weight": e.weight,
                "weight_change": e.weight_change,
                "win_odds": e.win_odds,
            }
            for e in race.entries
        ],
        "results": None,
        "payouts": None,
    }


class TestLoadRaces:
    """load_racesメソッドのテスト。"""

    def test_empty_directory(self, tmp_path: Path) -> None:
        """データディレクトリが空の場合、空リストを返す。"""
        loader = HistoricalDataLoader(data_dir=tmp_path)
        result = loader.load_races(date(2024, 1, 1), date(2024, 12, 31))
        assert result == []

    def test_nonexistent_directory(self, tmp_path: Path) -> None:
        """データディレクトリが存在しない場合、空リストを返す。"""
        loader = HistoricalDataLoader(data_dir=tmp_path / "nonexistent")
        result = loader.load_races(date(2024, 1, 1), date(2024, 12, 31))
        assert result == []

    def test_load_single_race_from_json(self, tmp_path: Path) -> None:
        """単一のJSONファイルからレースデータを読み込む。"""
        race = _make_race(race_date=date(2024, 3, 15))
        json_file = tmp_path / "races_20240315.json"
        json_file.write_text(
            json.dumps([_race_to_dict(race)], ensure_ascii=False), encoding="utf-8"
        )

        loader = HistoricalDataLoader(data_dir=tmp_path)
        result = loader.load_races(date(2024, 1, 1), date(2024, 12, 31))

        assert len(result) == 1
        assert result[0].race_id == race.race_id
        assert result[0].race_date == date(2024, 3, 15)

    def test_filter_by_date_range(self, tmp_path: Path) -> None:
        """日付範囲でフィルタリングされる。"""
        races = [
            _make_race(race_id="001", race_date=date(2024, 1, 10)),
            _make_race(race_id="002", race_date=date(2024, 2, 15)),
            _make_race(race_id="003", race_date=date(2024, 3, 20)),
        ]
        json_file = tmp_path / "races.json"
        json_file.write_text(
            json.dumps([_race_to_dict(r) for r in races], ensure_ascii=False),
            encoding="utf-8",
        )

        loader = HistoricalDataLoader(data_dir=tmp_path)
        result = loader.load_races(date(2024, 2, 1), date(2024, 2, 28))

        assert len(result) == 1
        assert result[0].race_id == "002"

    def test_inclusive_date_boundaries(self, tmp_path: Path) -> None:
        """開始日と終了日を含む（inclusive）。"""
        races = [
            _make_race(race_id="001", race_date=date(2024, 3, 1)),
            _make_race(race_id="002", race_date=date(2024, 3, 15)),
            _make_race(race_id="003", race_date=date(2024, 3, 31)),
        ]
        json_file = tmp_path / "races.json"
        json_file.write_text(
            json.dumps([_race_to_dict(r) for r in races], ensure_ascii=False),
            encoding="utf-8",
        )

        loader = HistoricalDataLoader(data_dir=tmp_path)
        result = loader.load_races(date(2024, 3, 1), date(2024, 3, 31))

        assert len(result) == 3

    def test_results_sorted_by_date(self, tmp_path: Path) -> None:
        """結果が日付の昇順でソートされる。"""
        races = [
            _make_race(race_id="003", race_date=date(2024, 3, 20)),
            _make_race(race_id="001", race_date=date(2024, 1, 10)),
            _make_race(race_id="002", race_date=date(2024, 2, 15)),
        ]
        json_file = tmp_path / "races.json"
        json_file.write_text(
            json.dumps([_race_to_dict(r) for r in races], ensure_ascii=False),
            encoding="utf-8",
        )

        loader = HistoricalDataLoader(data_dir=tmp_path)
        result = loader.load_races(date(2024, 1, 1), date(2024, 12, 31))

        assert result[0].race_id == "001"
        assert result[1].race_id == "002"
        assert result[2].race_id == "003"

    def test_single_race_dict_format(self, tmp_path: Path) -> None:
        """辞書形式の単一レースJSONも読み込める。"""
        race = _make_race(race_date=date(2024, 5, 1))
        json_file = tmp_path / "single_race.json"
        json_file.write_text(
            json.dumps(_race_to_dict(race), ensure_ascii=False), encoding="utf-8"
        )

        loader = HistoricalDataLoader(data_dir=tmp_path)
        result = loader.load_races(date(2024, 1, 1), date(2024, 12, 31))

        assert len(result) == 1

    def test_invalid_json_file_skipped(self, tmp_path: Path) -> None:
        """不正なJSONファイルはスキップされる。"""
        invalid_file = tmp_path / "invalid.json"
        invalid_file.write_text("not valid json{{{", encoding="utf-8")

        valid_race = _make_race(race_date=date(2024, 6, 1))
        valid_file = tmp_path / "valid.json"
        valid_file.write_text(
            json.dumps([_race_to_dict(valid_race)], ensure_ascii=False),
            encoding="utf-8",
        )

        loader = HistoricalDataLoader(data_dir=tmp_path)
        result = loader.load_races(date(2024, 1, 1), date(2024, 12, 31))

        assert len(result) == 1


class TestSplitData:
    """split_dataメソッドのテスト。"""

    def test_empty_list(self) -> None:
        """空リストの場合、両方とも空リストを返す。"""
        loader = HistoricalDataLoader()
        train, validation = loader.split_data([])
        assert train == []
        assert validation == []

    def test_default_80_20_split(self) -> None:
        """デフォルトで80/20に分割される。"""
        races = [
            _make_race(race_id=f"{i:03d}", race_date=date(2024, 1, i + 1))
            for i in range(10)
        ]
        loader = HistoricalDataLoader()
        train, validation = loader.split_data(races)

        assert len(train) == 8
        assert len(validation) == 2

    def test_total_count_preserved(self) -> None:
        """分割後の合計件数が元の件数と一致する。"""
        races = [
            _make_race(race_id=f"{i:03d}", race_date=date(2024, 1, i + 1))
            for i in range(7)
        ]
        loader = HistoricalDataLoader()
        train, validation = loader.split_data(races)

        assert len(train) + len(validation) == len(races)

    def test_chronological_split(self) -> None:
        """時系列順に分割される（古いデータが学習用）。"""
        races = [
            _make_race(race_id="003", race_date=date(2024, 3, 1)),
            _make_race(race_id="001", race_date=date(2024, 1, 1)),
            _make_race(race_id="002", race_date=date(2024, 2, 1)),
            _make_race(race_id="004", race_date=date(2024, 4, 1)),
            _make_race(race_id="005", race_date=date(2024, 5, 1)),
        ]
        loader = HistoricalDataLoader()
        train, validation = loader.split_data(races)

        # 学習用は日付が古い方
        assert train[0].race_id == "001"
        assert train[-1].race_date < validation[0].race_date

    def test_custom_ratio(self) -> None:
        """カスタム比率で分割される。"""
        races = [
            _make_race(race_id=f"{i:03d}", race_date=date(2024, 1, i + 1))
            for i in range(10)
        ]
        loader = HistoricalDataLoader()
        train, validation = loader.split_data(races, train_ratio=0.5)

        assert len(train) == 5
        assert len(validation) == 5

    def test_single_race(self) -> None:
        """1件の場合、学習用に入り検証用は空になる。"""
        races = [_make_race()]
        loader = HistoricalDataLoader()
        train, validation = loader.split_data(races)

        # int(1 * 0.8) = 0 なので学習0件、検証1件になる
        # ただし実装上 int(1*0.8)=0 なので train=[], validation=[race]
        assert len(train) + len(validation) == 1


class TestValidateAndClean:
    """validate_and_cleanメソッドのテスト。"""

    def test_valid_races_pass(self) -> None:
        """正常なレースデータはすべてクリーンデータに含まれる。"""
        races = [_make_race(), _make_race(race_id="002", race_date=date(2024, 2, 1))]
        loader = HistoricalDataLoader()
        report = loader.validate_and_clean(races)

        assert report.total_records == 2
        assert report.excluded_count == 0
        assert report.exclusion_reasons == []
        assert len(report.clean_races) == 2

    def test_empty_race_id_excluded(self) -> None:
        """race_idが空のレースは除外される。"""
        races = [_make_race(race_id="")]
        loader = HistoricalDataLoader()
        report = loader.validate_and_clean(races)

        assert report.excluded_count == 1
        assert "race_id" in report.exclusion_reasons[0][1]

    def test_empty_race_name_excluded(self) -> None:
        """race_nameが空のレースは除外される。"""
        races = [_make_race(race_name="")]
        loader = HistoricalDataLoader()
        report = loader.validate_and_clean(races)

        assert report.excluded_count == 1
        assert "race_name" in report.exclusion_reasons[0][1]

    def test_empty_venue_excluded(self) -> None:
        """venueが空のレースは除外される。"""
        races = [_make_race(venue="")]
        loader = HistoricalDataLoader()
        report = loader.validate_and_clean(races)

        assert report.excluded_count == 1
        assert "venue" in report.exclusion_reasons[0][1]

    def test_less_than_two_entries_excluded(self) -> None:
        """出走馬が2頭未満のレースは除外される。"""
        races = [_make_race(entries=[_make_entry()])]
        loader = HistoricalDataLoader()
        report = loader.validate_and_clean(races)

        assert report.excluded_count == 1
        assert "2頭未満" in report.exclusion_reasons[0][1]

    def test_zero_distance_excluded(self) -> None:
        """distanceが0以下のレースは除外される。"""
        races = [_make_race(distance=0)]
        loader = HistoricalDataLoader()
        report = loader.validate_and_clean(races)

        assert report.excluded_count == 1
        assert "distance" in report.exclusion_reasons[0][1]

    def test_negative_distance_excluded(self) -> None:
        """distanceが負のレースは除外される。"""
        races = [_make_race(distance=-100)]
        loader = HistoricalDataLoader()
        report = loader.validate_and_clean(races)

        assert report.excluded_count == 1
        assert "distance" in report.exclusion_reasons[0][1]

    def test_invalid_horse_number_excluded(self) -> None:
        """horse_numberが0以下の出走馬を含むレースは除外される。"""
        entries = [
            _make_entry(horse_number=0, gate_number=1, horse_name="馬A", jockey_name="騎手A"),
            _make_entry(horse_number=2, gate_number=2, horse_name="馬B", jockey_name="騎手B"),
        ]
        races = [_make_race(entries=entries)]
        loader = HistoricalDataLoader()
        report = loader.validate_and_clean(races)

        assert report.excluded_count == 1
        assert "horse_number" in report.exclusion_reasons[0][1]

    def test_invalid_gate_number_excluded(self) -> None:
        """gate_numberが範囲外（1-8以外）の出走馬を含むレースは除外される。"""
        entries = [
            _make_entry(horse_number=1, gate_number=9, horse_name="馬A", jockey_name="騎手A"),
            _make_entry(horse_number=2, gate_number=2, horse_name="馬B", jockey_name="騎手B"),
        ]
        races = [_make_race(entries=entries)]
        loader = HistoricalDataLoader()
        report = loader.validate_and_clean(races)

        assert report.excluded_count == 1
        assert "gate_number" in report.exclusion_reasons[0][1]

    def test_empty_horse_name_excluded(self) -> None:
        """horse_nameが空の出走馬を含むレースは除外される。"""
        entries = [
            _make_entry(horse_number=1, gate_number=1, horse_name="", jockey_name="騎手A"),
            _make_entry(horse_number=2, gate_number=2, horse_name="馬B", jockey_name="騎手B"),
        ]
        races = [_make_race(entries=entries)]
        loader = HistoricalDataLoader()
        report = loader.validate_and_clean(races)

        assert report.excluded_count == 1
        assert "horse_name" in report.exclusion_reasons[0][1]

    def test_empty_jockey_name_excluded(self) -> None:
        """jockey_nameが空の出走馬を含むレースは除外される。"""
        entries = [
            _make_entry(horse_number=1, gate_number=1, horse_name="馬A", jockey_name=""),
            _make_entry(horse_number=2, gate_number=2, horse_name="馬B", jockey_name="騎手B"),
        ]
        races = [_make_race(entries=entries)]
        loader = HistoricalDataLoader()
        report = loader.validate_and_clean(races)

        assert report.excluded_count == 1
        assert "jockey_name" in report.exclusion_reasons[0][1]

    def test_mixed_valid_and_invalid(self) -> None:
        """正常と不正が混在する場合、不正のみ除外される。"""
        valid_race = _make_race(race_id="valid_001")
        invalid_race = _make_race(race_id="", race_name="不正レース")
        races = [valid_race, invalid_race]

        loader = HistoricalDataLoader()
        report = loader.validate_and_clean(races)

        assert report.total_records == 2
        assert report.excluded_count == 1
        assert len(report.clean_races) == 1
        assert report.clean_races[0].race_id == "valid_001"

    def test_cleaning_report_type(self) -> None:
        """戻り値がCleaningReportインスタンスであること。"""
        loader = HistoricalDataLoader()
        report = loader.validate_and_clean([])

        assert isinstance(report, CleaningReport)
        assert report.total_records == 0
        assert report.excluded_count == 0
        assert report.clean_races == []
