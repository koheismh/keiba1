"""FeatureEngineer のユニットテスト"""

from datetime import date

import numpy as np
import pytest

from src.data.models import (
    FeatureVector,
    HorseEntry,
    RaceData,
    RaceResult,
    TrackCondition,
)
from src.features.engineer import FEATURE_NAMES, FeatureEngineer


def _make_entry(
    horse_name: str = "テスト馬",
    jockey_name: str = "テスト騎手",
    gate_number: int = 3,
    horse_number: int = 5,
    weight: int | None = 480,
    weight_change: int | None = 2,
    win_odds: float | None = 5.0,
) -> HorseEntry:
    return HorseEntry(
        horse_name=horse_name,
        jockey_name=jockey_name,
        gate_number=gate_number,
        horse_number=horse_number,
        weight=weight,
        weight_change=weight_change,
        win_odds=win_odds,
    )


def _make_race(
    race_id: str = "202401010101",
    venue: str = "東京",
    distance: int = 1600,
    track_condition: TrackCondition = TrackCondition.FIRM,
    entries: list[HorseEntry] | None = None,
    results: list[RaceResult] | None = None,
) -> RaceData:
    if entries is None:
        entries = [_make_entry()]
    return RaceData(
        race_id=race_id,
        race_name="テストレース",
        race_date=date(2024, 1, 1),
        post_time=None,
        venue=venue,
        course_type="芝",
        distance=distance,
        track_condition=track_condition,
        weather=None,
        entries=entries,
        results=results,
        payouts=None,
    )


class TestFeatureEngineerInit:
    """初期化テスト"""

    def test_init_with_none(self):
        fe = FeatureEngineer(historical_races=None)
        assert fe._historical_races == []

    def test_init_with_empty_list(self):
        fe = FeatureEngineer(historical_races=[])
        assert fe._historical_races == []

    def test_init_with_races(self):
        races = [_make_race()]
        fe = FeatureEngineer(historical_races=races)
        assert fe._historical_races == races


class TestGetFeatureNames:
    """get_feature_names テスト"""

    def test_returns_correct_names(self):
        fe = FeatureEngineer()
        names = fe.get_feature_names()
        assert names == [
            "past_performance",
            "jockey_performance",
            "course_aptitude",
            "distance_aptitude",
            "track_condition_aptitude",
            "gate_number",
            "weight_change",
            "class_performance",
        ]

    def test_returns_new_list(self):
        """返却されるリストは内部状態と独立していること"""
        fe = FeatureEngineer()
        names1 = fe.get_feature_names()
        names2 = fe.get_feature_names()
        assert names1 == names2
        assert names1 is not names2


class TestExtractFeaturesNewHorse:
    """過去データなし（新馬）の場合のテスト"""

    def test_new_horse_defaults(self):
        fe = FeatureEngineer(historical_races=[])
        horse = _make_entry(gate_number=4, weight_change=0)
        race = _make_race()

        fv = fe.extract_features(race, horse)

        assert isinstance(fv, FeatureVector)
        assert fv.feature_names == FEATURE_NAMES
        assert len(fv.values) == 8
        # past_performance = 0.0
        assert fv.values[0] == 0.0
        # jockey_performance = 0.0
        assert fv.values[1] == 0.0
        # course_aptitude = 0.0
        assert fv.values[2] == 0.0
        # distance_aptitude = 0.0
        assert fv.values[3] == 0.0
        # track_condition_aptitude = 0.0
        assert fv.values[4] == 0.0
        # gate_number = 4/8 = 0.5
        assert fv.values[5] == pytest.approx(0.5)
        # weight_change = 0/20 = 0.0
        assert fv.values[6] == 0.0
        # class_performance = 0.5 (default)
        assert fv.values[7] == 0.5


class TestGateNumber:
    """枠番正規化テスト"""

    def test_gate_1(self):
        fe = FeatureEngineer()
        horse = _make_entry(gate_number=1)
        race = _make_race()
        fv = fe.extract_features(race, horse)
        assert fv.values[5] == pytest.approx(1 / 8.0)

    def test_gate_8(self):
        fe = FeatureEngineer()
        horse = _make_entry(gate_number=8)
        race = _make_race()
        fv = fe.extract_features(race, horse)
        assert fv.values[5] == pytest.approx(1.0)


class TestWeightChange:
    """馬体重変動正規化テスト"""

    def test_none_weight_change(self):
        fe = FeatureEngineer()
        horse = _make_entry(weight_change=None)
        race = _make_race()
        fv = fe.extract_features(race, horse)
        assert fv.values[6] == 0.0

    def test_positive_weight_change(self):
        fe = FeatureEngineer()
        horse = _make_entry(weight_change=10)
        race = _make_race()
        fv = fe.extract_features(race, horse)
        assert fv.values[6] == pytest.approx(0.5)

    def test_negative_weight_change(self):
        fe = FeatureEngineer()
        horse = _make_entry(weight_change=-10)
        race = _make_race()
        fv = fe.extract_features(race, horse)
        assert fv.values[6] == pytest.approx(-0.5)

    def test_capped_at_positive_1(self):
        fe = FeatureEngineer()
        horse = _make_entry(weight_change=30)
        race = _make_race()
        fv = fe.extract_features(race, horse)
        assert fv.values[6] == 1.0

    def test_capped_at_negative_1(self):
        fe = FeatureEngineer()
        horse = _make_entry(weight_change=-30)
        race = _make_race()
        fv = fe.extract_features(race, horse)
        assert fv.values[6] == -1.0


class TestPastPerformance:
    """過去成績テスト"""

    def test_horse_with_wins(self):
        horse = _make_entry(horse_name="勝馬", horse_number=1)
        # 3 races, 2 wins
        historical = [
            _make_race(
                race_id="r1",
                entries=[_make_entry(horse_name="勝馬", horse_number=1)],
                results=[RaceResult(horse_number=1, finish_position=1)],
            ),
            _make_race(
                race_id="r2",
                entries=[_make_entry(horse_name="勝馬", horse_number=1)],
                results=[RaceResult(horse_number=1, finish_position=3)],
            ),
            _make_race(
                race_id="r3",
                entries=[_make_entry(horse_name="勝馬", horse_number=1)],
                results=[RaceResult(horse_number=1, finish_position=1)],
            ),
        ]
        fe = FeatureEngineer(historical_races=historical)
        race = _make_race(entries=[horse])
        fv = fe.extract_features(race, horse)
        # Win rate = 2/3
        assert fv.values[0] == pytest.approx(2 / 3)


class TestJockeyPerformance:
    """騎手成績テスト"""

    def test_jockey_win_rate(self):
        # Jockey rode in 4 races, won 1
        entries_j = _make_entry(jockey_name="名騎手", horse_number=2)
        historical = [
            _make_race(
                race_id="r1",
                entries=[_make_entry(jockey_name="名騎手", horse_number=2)],
                results=[RaceResult(horse_number=2, finish_position=1)],
            ),
            _make_race(
                race_id="r2",
                entries=[_make_entry(jockey_name="名騎手", horse_number=2)],
                results=[RaceResult(horse_number=2, finish_position=5)],
            ),
            _make_race(
                race_id="r3",
                entries=[_make_entry(jockey_name="名騎手", horse_number=2)],
                results=[RaceResult(horse_number=2, finish_position=2)],
            ),
            _make_race(
                race_id="r4",
                entries=[_make_entry(jockey_name="名騎手", horse_number=2)],
                results=[RaceResult(horse_number=2, finish_position=3)],
            ),
        ]
        fe = FeatureEngineer(historical_races=historical)
        horse = _make_entry(jockey_name="名騎手", horse_number=1)
        race = _make_race(entries=[horse])
        fv = fe.extract_features(race, horse)
        # Win rate = 1/4
        assert fv.values[1] == pytest.approx(0.25)


class TestCourseAptitude:
    """コース適性テスト"""

    def test_same_venue_win_rate(self):
        historical = [
            _make_race(
                race_id="r1",
                venue="東京",
                entries=[_make_entry(horse_name="東京馬", horse_number=1)],
                results=[RaceResult(horse_number=1, finish_position=1)],
            ),
            _make_race(
                race_id="r2",
                venue="東京",
                entries=[_make_entry(horse_name="東京馬", horse_number=1)],
                results=[RaceResult(horse_number=1, finish_position=2)],
            ),
            _make_race(
                race_id="r3",
                venue="中山",
                entries=[_make_entry(horse_name="東京馬", horse_number=1)],
                results=[RaceResult(horse_number=1, finish_position=1)],
            ),
        ]
        fe = FeatureEngineer(historical_races=historical)
        horse = _make_entry(horse_name="東京馬", horse_number=1)
        race = _make_race(venue="東京", entries=[horse])
        fv = fe.extract_features(race, horse)
        # 東京での勝率 = 1/2
        assert fv.values[2] == pytest.approx(0.5)


class TestDistanceAptitude:
    """距離適性テスト"""

    def test_similar_distance_win_rate(self):
        historical = [
            _make_race(
                race_id="r1",
                distance=1600,
                entries=[_make_entry(horse_name="マイル馬", horse_number=1)],
                results=[RaceResult(horse_number=1, finish_position=1)],
            ),
            _make_race(
                race_id="r2",
                distance=1800,  # within ±200m of 1600
                entries=[_make_entry(horse_name="マイル馬", horse_number=1)],
                results=[RaceResult(horse_number=1, finish_position=3)],
            ),
            _make_race(
                race_id="r3",
                distance=2400,  # outside ±200m
                entries=[_make_entry(horse_name="マイル馬", horse_number=1)],
                results=[RaceResult(horse_number=1, finish_position=1)],
            ),
        ]
        fe = FeatureEngineer(historical_races=historical)
        horse = _make_entry(horse_name="マイル馬", horse_number=1)
        race = _make_race(distance=1600, entries=[horse])
        fv = fe.extract_features(race, horse)
        # Similar distances (1600, 1800): win rate = 1/2
        assert fv.values[3] == pytest.approx(0.5)


class TestTrackConditionAptitude:
    """馬場状態適性テスト"""

    def test_same_condition_win_rate(self):
        historical = [
            _make_race(
                race_id="r1",
                track_condition=TrackCondition.SOFT,
                entries=[_make_entry(horse_name="重馬場馬", horse_number=1)],
                results=[RaceResult(horse_number=1, finish_position=1)],
            ),
            _make_race(
                race_id="r2",
                track_condition=TrackCondition.SOFT,
                entries=[_make_entry(horse_name="重馬場馬", horse_number=1)],
                results=[RaceResult(horse_number=1, finish_position=1)],
            ),
            _make_race(
                race_id="r3",
                track_condition=TrackCondition.FIRM,
                entries=[_make_entry(horse_name="重馬場馬", horse_number=1)],
                results=[RaceResult(horse_number=1, finish_position=5)],
            ),
        ]
        fe = FeatureEngineer(historical_races=historical)
        horse = _make_entry(horse_name="重馬場馬", horse_number=1)
        race = _make_race(track_condition=TrackCondition.SOFT, entries=[horse])
        fv = fe.extract_features(race, horse)
        # SOFT condition win rate = 2/2 = 1.0
        assert fv.values[4] == pytest.approx(1.0)


class TestClassPerformance:
    """クラス実績テスト"""

    def test_recent_performance(self):
        # Horse with recent finishes: 2, 3, 1, 5, 4
        historical = [
            _make_race(
                race_id=f"r{i}",
                entries=[_make_entry(horse_name="実績馬", horse_number=1)],
                results=[RaceResult(horse_number=1, finish_position=pos)],
            )
            for i, pos in enumerate([2, 3, 1, 5, 4], start=1)
        ]
        fe = FeatureEngineer(historical_races=historical)
        horse = _make_entry(horse_name="実績馬", horse_number=1)
        race = _make_race(entries=[horse])
        fv = fe.extract_features(race, horse)
        # Average position = (2+3+1+5+4)/5 = 3.0
        # Normalized = (3.0 - 1) / (18 - 1) = 2.0 / 17 ≈ 0.1176
        assert fv.values[7] == pytest.approx(2.0 / 17.0)

    def test_uses_only_last_5_races(self):
        """7レース分のデータがあっても直近5レースのみ使用すること"""
        # Positions: 10, 12, 15, 1, 2, 3, 1 → last 5: 15, 1, 2, 3, 1
        positions = [10, 12, 15, 1, 2, 3, 1]
        historical = [
            _make_race(
                race_id=f"r{i}",
                entries=[_make_entry(horse_name="ベテラン馬", horse_number=1)],
                results=[RaceResult(horse_number=1, finish_position=pos)],
            )
            for i, pos in enumerate(positions, start=1)
        ]
        fe = FeatureEngineer(historical_races=historical)
        horse = _make_entry(horse_name="ベテラン馬", horse_number=1)
        race = _make_race(entries=[horse])
        fv = fe.extract_features(race, horse)
        # Last 5 positions: 15, 1, 2, 3, 1 → avg = 22/5 = 4.4
        # Normalized = (4.4 - 1) / (18 - 1) = 3.4 / 17
        assert fv.values[7] == pytest.approx(3.4 / 17.0)

    def test_no_history_returns_default(self):
        fe = FeatureEngineer(historical_races=[])
        horse = _make_entry(horse_name="新馬")
        race = _make_race(entries=[horse])
        fv = fe.extract_features(race, horse)
        assert fv.values[7] == 0.5


class TestDistanceAptitudeBoundary:
    """距離適性の±200m境界テスト"""

    def test_exactly_200m_difference_is_included(self):
        """距離差がちょうど200mの場合は含まれる"""
        historical = [
            _make_race(
                race_id="r1",
                distance=1800,  # exactly 200m from 1600
                entries=[_make_entry(horse_name="境界馬", horse_number=1)],
                results=[RaceResult(horse_number=1, finish_position=1)],
            ),
        ]
        fe = FeatureEngineer(historical_races=historical)
        horse = _make_entry(horse_name="境界馬", horse_number=1)
        race = _make_race(distance=1600, entries=[horse])
        fv = fe.extract_features(race, horse)
        # 1800 is within ±200m of 1600 → win rate = 1/1 = 1.0
        assert fv.values[3] == pytest.approx(1.0)

    def test_201m_difference_is_excluded(self):
        """距離差が201mの場合は含まれない"""
        historical = [
            _make_race(
                race_id="r1",
                distance=1801,  # 201m from 1600 - outside range
                entries=[_make_entry(horse_name="遠距離馬", horse_number=1)],
                results=[RaceResult(horse_number=1, finish_position=1)],
            ),
        ]
        fe = FeatureEngineer(historical_races=historical)
        horse = _make_entry(horse_name="遠距離馬", horse_number=1)
        race = _make_race(distance=1600, entries=[horse])
        fv = fe.extract_features(race, horse)
        # No matching races within ±200m → fallback 0.0
        assert fv.values[3] == 0.0


class TestFeatureVectorStructure:
    """FeatureVector構造テスト"""

    def test_values_dtype_is_float64(self):
        """特徴量の値がfloat64であること"""
        fe = FeatureEngineer()
        horse = _make_entry()
        race = _make_race()
        fv = fe.extract_features(race, horse)
        assert fv.values.dtype == np.float64

    def test_feature_count_matches_names(self):
        """値の数と特徴量名の数が一致すること"""
        fe = FeatureEngineer()
        horse = _make_entry()
        race = _make_race()
        fv = fe.extract_features(race, horse)
        assert len(fv.values) == len(fv.feature_names)


class TestHistoricalDataEdgeCases:
    """過去データのエッジケーステスト"""

    def test_race_with_no_results_is_ignored(self):
        """results=Noneのレースは無視される"""
        historical = [
            _make_race(
                race_id="r1",
                entries=[_make_entry(horse_name="テスト馬X", horse_number=1)],
                results=None,  # No results available
            ),
        ]
        fe = FeatureEngineer(historical_races=historical)
        horse = _make_entry(horse_name="テスト馬X", horse_number=1)
        race = _make_race(entries=[horse])
        fv = fe.extract_features(race, horse)
        # No usable results → all history-based features are defaults
        assert fv.values[0] == 0.0  # past_performance
        assert fv.values[7] == 0.5  # class_performance default

    def test_multiple_horses_in_race_matches_correctly(self):
        """複数の馬がいるレースで正しい馬の結果を取得する"""
        entries = [
            _make_entry(horse_name="馬A", horse_number=1),
            _make_entry(horse_name="馬B", horse_number=2),
            _make_entry(horse_name="馬C", horse_number=3),
        ]
        results = [
            RaceResult(horse_number=1, finish_position=3),
            RaceResult(horse_number=2, finish_position=1),
            RaceResult(horse_number=3, finish_position=2),
        ]
        historical = [_make_race(race_id="r1", entries=entries, results=results)]
        fe = FeatureEngineer(historical_races=historical)

        # 馬B won the race
        horse_b = _make_entry(horse_name="馬B", horse_number=2)
        race = _make_race(entries=[horse_b])
        fv = fe.extract_features(race, horse_b)
        assert fv.values[0] == pytest.approx(1.0)  # 1 win / 1 race

        # 馬A did not win
        horse_a = _make_entry(horse_name="馬A", horse_number=1)
        race = _make_race(entries=[horse_a])
        fv = fe.extract_features(race, horse_a)
        assert fv.values[0] == pytest.approx(0.0)  # 0 wins / 1 race
