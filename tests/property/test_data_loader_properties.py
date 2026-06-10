"""データ読み込みのプロパティベーステスト。

Feature: horse-race-predictor, Property 1: 日付範囲フィルタリング
Feature: horse-race-predictor, Property 3: 不正データ除外の完全性

Validates: Requirements 1.1, 1.3
"""

import json
import tempfile
from datetime import date
from pathlib import Path

from hypothesis import given, settings
from hypothesis import strategies as st

from src.data.historical_loader import HistoricalDataLoader
from src.data.models import HorseEntry, RaceData, TrackCondition


# --- Strategies ---

def valid_horse_entry_strategy() -> st.SearchStrategy[HorseEntry]:
    """有効なHorseEntryを生成するストラテジ。"""
    return st.builds(
        HorseEntry,
        horse_name=st.text(min_size=1, max_size=10, alphabet=st.characters(categories=("L",))),
        jockey_name=st.text(min_size=1, max_size=10, alphabet=st.characters(categories=("L",))),
        gate_number=st.integers(min_value=1, max_value=8),
        horse_number=st.integers(min_value=1, max_value=18),
        weight=st.one_of(st.none(), st.integers(min_value=400, max_value=600)),
        weight_change=st.one_of(st.none(), st.integers(min_value=-20, max_value=20)),
        win_odds=st.one_of(st.none(), st.floats(min_value=1.0, max_value=500.0)),
    )


def valid_race_data_strategy() -> st.SearchStrategy[RaceData]:
    """有効なRaceDataを生成するストラテジ。"""
    return st.builds(
        RaceData,
        race_id=st.text(min_size=1, max_size=12, alphabet=st.characters(categories=("L", "N"))),
        race_name=st.text(min_size=1, max_size=20, alphabet=st.characters(categories=("L",))),
        race_date=st.dates(min_value=date(2020, 1, 1), max_value=date(2024, 12, 31)),
        post_time=st.none(),
        venue=st.text(min_size=1, max_size=10, alphabet=st.characters(categories=("L",))),
        course_type=st.sampled_from(["芝", "ダート"]),
        distance=st.integers(min_value=1000, max_value=3600),
        track_condition=st.sampled_from(list(TrackCondition)),
        weather=st.none(),
        entries=st.lists(valid_horse_entry_strategy(), min_size=2, max_size=18),
        results=st.none(),
        payouts=st.none(),
    )


def invalid_race_data_strategy() -> st.SearchStrategy[RaceData]:
    """不正なRaceDataを生成するストラテジ。

    以下のいずれかの不正を含むデータを生成:
    - empty race_id
    - empty venue
    - entries < 2
    - invalid horse_number (0 or negative)
    - invalid gate_number (0 or > 8)
    - empty horse_name/jockey_name
    - non-positive distance
    """
    # 不正な出走馬エントリのストラテジ群
    invalid_horse_number_entry = st.builds(
        HorseEntry,
        horse_name=st.text(min_size=1, max_size=10, alphabet=st.characters(categories=("L",))),
        jockey_name=st.text(min_size=1, max_size=10, alphabet=st.characters(categories=("L",))),
        gate_number=st.integers(min_value=1, max_value=8),
        horse_number=st.integers(max_value=0),  # 0以下
        weight=st.none(),
        weight_change=st.none(),
        win_odds=st.none(),
    )

    invalid_gate_number_entry = st.builds(
        HorseEntry,
        horse_name=st.text(min_size=1, max_size=10, alphabet=st.characters(categories=("L",))),
        jockey_name=st.text(min_size=1, max_size=10, alphabet=st.characters(categories=("L",))),
        gate_number=st.one_of(
            st.integers(max_value=0),  # 0以下
            st.integers(min_value=9, max_value=99),  # 9以上
        ),
        horse_number=st.integers(min_value=1, max_value=18),
        weight=st.none(),
        weight_change=st.none(),
        win_odds=st.none(),
    )

    empty_horse_name_entry = st.builds(
        HorseEntry,
        horse_name=st.just(""),  # 空の馬名
        jockey_name=st.text(min_size=1, max_size=10, alphabet=st.characters(categories=("L",))),
        gate_number=st.integers(min_value=1, max_value=8),
        horse_number=st.integers(min_value=1, max_value=18),
        weight=st.none(),
        weight_change=st.none(),
        win_odds=st.none(),
    )

    empty_jockey_name_entry = st.builds(
        HorseEntry,
        horse_name=st.text(min_size=1, max_size=10, alphabet=st.characters(categories=("L",))),
        jockey_name=st.just(""),  # 空の騎手名
        gate_number=st.integers(min_value=1, max_value=8),
        horse_number=st.integers(min_value=1, max_value=18),
        weight=st.none(),
        weight_change=st.none(),
        win_odds=st.none(),
    )

    # 不正エントリを含むレースデータ（2頭以上必要なので、不正エントリ + 有効エントリ）
    def make_entries_with_invalid(invalid_entry_strategy):
        return st.lists(valid_horse_entry_strategy(), min_size=1, max_size=5).flatmap(
            lambda valid_entries: invalid_entry_strategy.map(
                lambda invalid: [invalid] + valid_entries
            )
        )

    # 各種不正パターンのRaceData
    empty_race_id = st.builds(
        RaceData,
        race_id=st.just(""),  # 空のrace_id
        race_name=st.text(min_size=1, max_size=20, alphabet=st.characters(categories=("L",))),
        race_date=st.dates(min_value=date(2020, 1, 1), max_value=date(2024, 12, 31)),
        post_time=st.none(),
        venue=st.text(min_size=1, max_size=10, alphabet=st.characters(categories=("L",))),
        course_type=st.sampled_from(["芝", "ダート"]),
        distance=st.integers(min_value=1000, max_value=3600),
        track_condition=st.sampled_from(list(TrackCondition)),
        weather=st.none(),
        entries=st.lists(valid_horse_entry_strategy(), min_size=2, max_size=8),
        results=st.none(),
        payouts=st.none(),
    )

    empty_venue = st.builds(
        RaceData,
        race_id=st.text(min_size=1, max_size=12, alphabet=st.characters(categories=("L", "N"))),
        race_name=st.text(min_size=1, max_size=20, alphabet=st.characters(categories=("L",))),
        race_date=st.dates(min_value=date(2020, 1, 1), max_value=date(2024, 12, 31)),
        post_time=st.none(),
        venue=st.just(""),  # 空のvenue
        course_type=st.sampled_from(["芝", "ダート"]),
        distance=st.integers(min_value=1000, max_value=3600),
        track_condition=st.sampled_from(list(TrackCondition)),
        weather=st.none(),
        entries=st.lists(valid_horse_entry_strategy(), min_size=2, max_size=8),
        results=st.none(),
        payouts=st.none(),
    )

    too_few_entries = st.builds(
        RaceData,
        race_id=st.text(min_size=1, max_size=12, alphabet=st.characters(categories=("L", "N"))),
        race_name=st.text(min_size=1, max_size=20, alphabet=st.characters(categories=("L",))),
        race_date=st.dates(min_value=date(2020, 1, 1), max_value=date(2024, 12, 31)),
        post_time=st.none(),
        venue=st.text(min_size=1, max_size=10, alphabet=st.characters(categories=("L",))),
        course_type=st.sampled_from(["芝", "ダート"]),
        distance=st.integers(min_value=1000, max_value=3600),
        track_condition=st.sampled_from(list(TrackCondition)),
        weather=st.none(),
        entries=st.lists(valid_horse_entry_strategy(), min_size=0, max_size=1),  # 2頭未満
        results=st.none(),
        payouts=st.none(),
    )

    non_positive_distance = st.builds(
        RaceData,
        race_id=st.text(min_size=1, max_size=12, alphabet=st.characters(categories=("L", "N"))),
        race_name=st.text(min_size=1, max_size=20, alphabet=st.characters(categories=("L",))),
        race_date=st.dates(min_value=date(2020, 1, 1), max_value=date(2024, 12, 31)),
        post_time=st.none(),
        venue=st.text(min_size=1, max_size=10, alphabet=st.characters(categories=("L",))),
        course_type=st.sampled_from(["芝", "ダート"]),
        distance=st.integers(max_value=0),  # 0以下
        track_condition=st.sampled_from(list(TrackCondition)),
        weather=st.none(),
        entries=st.lists(valid_horse_entry_strategy(), min_size=2, max_size=8),
        results=st.none(),
        payouts=st.none(),
    )

    with_invalid_horse_number = st.builds(
        RaceData,
        race_id=st.text(min_size=1, max_size=12, alphabet=st.characters(categories=("L", "N"))),
        race_name=st.text(min_size=1, max_size=20, alphabet=st.characters(categories=("L",))),
        race_date=st.dates(min_value=date(2020, 1, 1), max_value=date(2024, 12, 31)),
        post_time=st.none(),
        venue=st.text(min_size=1, max_size=10, alphabet=st.characters(categories=("L",))),
        course_type=st.sampled_from(["芝", "ダート"]),
        distance=st.integers(min_value=1000, max_value=3600),
        track_condition=st.sampled_from(list(TrackCondition)),
        weather=st.none(),
        entries=make_entries_with_invalid(invalid_horse_number_entry),
        results=st.none(),
        payouts=st.none(),
    )

    with_invalid_gate_number = st.builds(
        RaceData,
        race_id=st.text(min_size=1, max_size=12, alphabet=st.characters(categories=("L", "N"))),
        race_name=st.text(min_size=1, max_size=20, alphabet=st.characters(categories=("L",))),
        race_date=st.dates(min_value=date(2020, 1, 1), max_value=date(2024, 12, 31)),
        post_time=st.none(),
        venue=st.text(min_size=1, max_size=10, alphabet=st.characters(categories=("L",))),
        course_type=st.sampled_from(["芝", "ダート"]),
        distance=st.integers(min_value=1000, max_value=3600),
        track_condition=st.sampled_from(list(TrackCondition)),
        weather=st.none(),
        entries=make_entries_with_invalid(invalid_gate_number_entry),
        results=st.none(),
        payouts=st.none(),
    )

    with_empty_horse_name = st.builds(
        RaceData,
        race_id=st.text(min_size=1, max_size=12, alphabet=st.characters(categories=("L", "N"))),
        race_name=st.text(min_size=1, max_size=20, alphabet=st.characters(categories=("L",))),
        race_date=st.dates(min_value=date(2020, 1, 1), max_value=date(2024, 12, 31)),
        post_time=st.none(),
        venue=st.text(min_size=1, max_size=10, alphabet=st.characters(categories=("L",))),
        course_type=st.sampled_from(["芝", "ダート"]),
        distance=st.integers(min_value=1000, max_value=3600),
        track_condition=st.sampled_from(list(TrackCondition)),
        weather=st.none(),
        entries=make_entries_with_invalid(empty_horse_name_entry),
        results=st.none(),
        payouts=st.none(),
    )

    with_empty_jockey_name = st.builds(
        RaceData,
        race_id=st.text(min_size=1, max_size=12, alphabet=st.characters(categories=("L", "N"))),
        race_name=st.text(min_size=1, max_size=20, alphabet=st.characters(categories=("L",))),
        race_date=st.dates(min_value=date(2020, 1, 1), max_value=date(2024, 12, 31)),
        post_time=st.none(),
        venue=st.text(min_size=1, max_size=10, alphabet=st.characters(categories=("L",))),
        course_type=st.sampled_from(["芝", "ダート"]),
        distance=st.integers(min_value=1000, max_value=3600),
        track_condition=st.sampled_from(list(TrackCondition)),
        weather=st.none(),
        entries=make_entries_with_invalid(empty_jockey_name_entry),
        results=st.none(),
        payouts=st.none(),
    )

    return st.one_of(
        empty_race_id,
        empty_venue,
        too_few_entries,
        non_positive_distance,
        with_invalid_horse_number,
        with_invalid_gate_number,
        with_empty_horse_name,
        with_empty_jockey_name,
    )


class TestDataCleaningCompleteness:
    """Property 3: 不正データ除外の完全性

    Feature: horse-race-predictor, Property 3: 不正データ除外の完全性

    For any 不正データ（フォーマット不正または必須フィールド欠損）を含むデータセットに対して、
    クリーニング後のデータには不正レコードが含まれず、
    除外件数がクリーニングレポートの除外件数と一致すること。

    Validates: Requirements 1.3
    """

    @settings(max_examples=100)
    @given(
        valid_races=st.lists(valid_race_data_strategy(), min_size=0, max_size=5),
        invalid_races=st.lists(invalid_race_data_strategy(), min_size=1, max_size=5),
    )
    def test_no_invalid_records_remain_after_cleaning(
        self,
        valid_races: list[RaceData],
        invalid_races: list[RaceData],
    ) -> None:
        """クリーニング後データに不正レコードが含まれないことを検証する。

        Validates: Requirements 1.3
        """
        loader = HistoricalDataLoader()
        all_races = valid_races + invalid_races

        report = loader.validate_and_clean(all_races)

        # クリーン済みデータの各レースがバリデーションを通過することを確認
        for race in report.clean_races:
            reason = loader._validate_race(race)
            assert reason is None, (
                f"クリーニング後データに不正レコードが残っている: "
                f"race_id={race.race_id}, reason={reason}"
            )

    @settings(max_examples=100)
    @given(
        valid_races=st.lists(valid_race_data_strategy(), min_size=0, max_size=5),
        invalid_races=st.lists(invalid_race_data_strategy(), min_size=1, max_size=5),
    )
    def test_excluded_count_matches_report(
        self,
        valid_races: list[RaceData],
        invalid_races: list[RaceData],
    ) -> None:
        """除外件数 == total_records - len(clean_races) であることを検証する。

        Validates: Requirements 1.3
        """
        loader = HistoricalDataLoader()
        all_races = valid_races + invalid_races

        report = loader.validate_and_clean(all_races)

        assert report.excluded_count == report.total_records - len(report.clean_races), (
            f"除外件数の不整合: excluded_count={report.excluded_count}, "
            f"total_records={report.total_records}, "
            f"clean_races={len(report.clean_races)}"
        )

    @settings(max_examples=100)
    @given(
        valid_races=st.lists(valid_race_data_strategy(), min_size=0, max_size=5),
        invalid_races=st.lists(invalid_race_data_strategy(), min_size=1, max_size=5),
    )
    def test_excluded_count_equals_exclusion_reasons_count(
        self,
        valid_races: list[RaceData],
        invalid_races: list[RaceData],
    ) -> None:
        """除外件数 == len(exclusion_reasons) であることを検証する。

        Validates: Requirements 1.3
        """
        loader = HistoricalDataLoader()
        all_races = valid_races + invalid_races

        report = loader.validate_and_clean(all_races)

        assert report.excluded_count == len(report.exclusion_reasons), (
            f"除外件数とexclusion_reasonsの件数が不一致: "
            f"excluded_count={report.excluded_count}, "
            f"len(exclusion_reasons)={len(report.exclusion_reasons)}"
        )

    @settings(max_examples=100)
    @given(
        valid_races=st.lists(valid_race_data_strategy(), min_size=0, max_size=5),
        invalid_races=st.lists(invalid_race_data_strategy(), min_size=1, max_size=5),
    )
    def test_total_records_equals_original_input_count(
        self,
        valid_races: list[RaceData],
        invalid_races: list[RaceData],
    ) -> None:
        """total_records == 元の入力レース数であることを検証する。

        Validates: Requirements 1.3
        """
        loader = HistoricalDataLoader()
        all_races = valid_races + invalid_races

        report = loader.validate_and_clean(all_races)

        assert report.total_records == len(all_races), (
            f"total_recordsが入力件数と不一致: "
            f"total_records={report.total_records}, "
            f"input_count={len(all_races)}"
        )



# --- Strategies for Property 1 ---

# 日付範囲: 2020-01-01 〜 2025-12-31 の範囲でテスト
_date_strategy = st.dates(
    min_value=date(2020, 1, 1),
    max_value=date(2025, 12, 31),
)


def ordered_date_pair():
    """start_date <= end_date を保証する日付ペアを生成する。"""
    return st.tuples(_date_strategy, _date_strategy).map(
        lambda pair: (min(pair), max(pair))
    )


def race_json_strategy():
    """有効なレースJSONデータを生成するstrategy。

    HistoricalDataLoaderが正常にパースできる最小限のレースデータを生成する。
    """
    return st.builds(
        lambda race_id, race_date, horse1_num, horse2_num: {
            "race_id": race_id,
            "race_name": f"テストレース{race_id}",
            "race_date": race_date.isoformat(),
            "venue": "東京",
            "course_type": "芝",
            "distance": 2000,
            "track_condition": "良",
            "entries": [
                {
                    "horse_name": f"馬A_{race_id}",
                    "jockey_name": "騎手A",
                    "gate_number": 1,
                    "horse_number": horse1_num,
                    "weight": 480,
                    "weight_change": 0,
                    "win_odds": 3.5,
                },
                {
                    "horse_name": f"馬B_{race_id}",
                    "jockey_name": "騎手B",
                    "gate_number": 2,
                    "horse_number": horse2_num,
                    "weight": 460,
                    "weight_change": -2,
                    "win_odds": 5.0,
                },
            ],
            "results": None,
            "payouts": None,
        },
        race_id=st.text(
            alphabet="abcdefghijklmnopqrstuvwxyz0123456789",
            min_size=4,
            max_size=10,
        ),
        race_date=st.dates(
            min_value=date(2020, 1, 1),
            max_value=date(2025, 12, 31),
        ),
        horse1_num=st.just(1),
        horse2_num=st.just(2),
    )


class TestDateRangeFiltering:
    """Property 1: 日付範囲フィルタリング

    Feature: horse-race-predictor, Property 1: 日付範囲フィルタリング

    For any 有効な日付範囲（start_date, end_date）とレースデータセットに対して、
    Historical_Data_Loaderが返すすべてのレースの開催日は指定された範囲内
    （start_date ≤ race_date ≤ end_date）に収まること。

    Validates: Requirements 1.1
    """

    @settings(max_examples=100)
    @given(
        date_pair=ordered_date_pair(),
        races_data=st.lists(race_json_strategy(), min_size=1, max_size=20),
    )
    def test_all_returned_races_within_date_range(
        self,
        date_pair: tuple[date, date],
        races_data: list[dict],
    ) -> None:
        """load_racesが返すすべてのレースの開催日は
        指定された日付範囲内（start_date ≤ race_date ≤ end_date）に収まること。

        Validates: Requirements 1.1
        """
        start_date, end_date = date_pair

        # 一時ディレクトリにJSONファイルを作成
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            json_file = tmp_path / "test_races.json"
            with open(json_file, "w", encoding="utf-8") as f:
                json.dump(races_data, f, ensure_ascii=False)

            # HistoricalDataLoaderでデータを読み込む
            loader = HistoricalDataLoader(data_dir=tmp_path)
            result = loader.load_races(start_date=start_date, end_date=end_date)

            # プロパティ: 返されたすべてのレースの日付が範囲内であること
            for race in result:
                assert start_date <= race.race_date <= end_date, (
                    f"Race {race.race_id} has date {race.race_date} "
                    f"which is outside range [{start_date}, {end_date}]"
                )
