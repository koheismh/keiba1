"""RaceEvaluator のユニットテスト"""

from datetime import date

import numpy as np
import pytest

from src.data.models import HorseEntry, RaceData, RaceEvaluation, TrackCondition
from src.evaluation.race_evaluator import RaceEvaluator


def _make_entry(
    horse_number: int,
    win_odds: float | None = 5.0,
    weight: int | None = 480,
) -> HorseEntry:
    """テスト用の出走馬エントリーを生成する"""
    return HorseEntry(
        horse_name=f"Horse{horse_number}",
        jockey_name=f"Jockey{horse_number}",
        gate_number=min(horse_number, 8),
        horse_number=horse_number,
        weight=weight,
        weight_change=0,
        win_odds=win_odds,
    )


def _make_race(
    num_entries: int = 10,
    odds_available: bool = True,
    weight_available: bool = True,
) -> RaceData:
    """テスト用のレースデータを生成する"""
    entries = [
        _make_entry(
            i + 1,
            win_odds=5.0 if odds_available else None,
            weight=480 if weight_available else None,
        )
        for i in range(num_entries)
    ]
    return RaceData(
        race_id="202401010101",
        race_name="テストレース",
        race_date=date(2024, 1, 1),
        post_time=None,
        venue="東京",
        course_type="芝",
        distance=2000,
        track_condition=TrackCondition.FIRM,
        weather="晴",
        entries=entries,
        results=None,
        payouts=None,
    )


class TestRaceEvaluatorInit:
    """初期化テスト"""

    def test_default_threshold(self) -> None:
        evaluator = RaceEvaluator()
        assert evaluator.confidence_threshold == 50

    def test_custom_threshold(self) -> None:
        evaluator = RaceEvaluator(confidence_threshold=70)
        assert evaluator.confidence_threshold == 70


class TestRaceEvaluatorEvaluate:
    """evaluate メソッドのテスト"""

    def test_score_range_0_to_100(self) -> None:
        """信頼度スコアが0〜100の範囲であること"""
        evaluator = RaceEvaluator()
        race = _make_race()
        predictions = np.array([0.1] * 10)
        result = evaluator.evaluate(race, predictions)
        assert 0 <= result.confidence_score <= 100

    def test_clear_favorite_high_confidence(self) -> None:
        """明確な本命馬がいるレースは信頼度が高い"""
        evaluator = RaceEvaluator(confidence_threshold=30)
        race = _make_race(num_entries=5)
        # 1頭が支配的な確率を持つ分布
        predictions = np.array([0.7, 0.1, 0.1, 0.05, 0.05])
        result = evaluator.evaluate(race, predictions)
        assert result.confidence_score > 40
        assert result.should_bet is True

    def test_uniform_distribution_low_confidence(self) -> None:
        """均等分布のレースは信頼度が低い"""
        evaluator = RaceEvaluator(confidence_threshold=60)
        race = _make_race(num_entries=10)
        # 均等分布
        predictions = np.array([0.1] * 10)
        result = evaluator.evaluate(race, predictions)
        assert result.confidence_score < 60
        assert result.should_bet is False

    def test_should_bet_true_when_score_above_threshold(self) -> None:
        """スコアが閾値以上のとき should_bet=True"""
        evaluator = RaceEvaluator(confidence_threshold=20)
        race = _make_race(num_entries=3)
        predictions = np.array([0.8, 0.1, 0.1])
        result = evaluator.evaluate(race, predictions)
        assert result.should_bet is True
        assert result.skip_reason is None

    def test_should_bet_false_when_score_below_threshold(self) -> None:
        """スコアが閾値未満のとき should_bet=False"""
        evaluator = RaceEvaluator(confidence_threshold=90)
        race = _make_race(num_entries=10)
        predictions = np.array([0.1] * 10)
        result = evaluator.evaluate(race, predictions)
        assert result.should_bet is False
        assert result.skip_reason is not None

    def test_skip_reason_included_when_not_betting(self) -> None:
        """見送り時には理由が含まれる"""
        evaluator = RaceEvaluator(confidence_threshold=90)
        race = _make_race(num_entries=10)
        predictions = np.array([0.1] * 10)
        result = evaluator.evaluate(race, predictions)
        assert result.skip_reason is not None
        assert "見送り" in result.skip_reason

    def test_factors_dict_contains_all_keys(self) -> None:
        """factors辞書に全キーが含まれること"""
        evaluator = RaceEvaluator()
        race = _make_race()
        predictions = np.array([0.1] * 10)
        result = evaluator.evaluate(race, predictions)
        assert "unpredictability" in result.factors
        assert "strength_gap" in result.factors
        assert "data_completeness" in result.factors

    def test_factors_values_in_range(self) -> None:
        """factors値が0-1の範囲であること"""
        evaluator = RaceEvaluator()
        race = _make_race()
        predictions = np.array([0.5, 0.2, 0.15, 0.1, 0.05])
        result = evaluator.evaluate(race, predictions)
        for value in result.factors.values():
            assert 0.0 <= value <= 1.0

    def test_race_id_preserved(self) -> None:
        """race_idが正しく引き継がれること"""
        evaluator = RaceEvaluator()
        race = _make_race()
        predictions = np.array([0.1] * 10)
        result = evaluator.evaluate(race, predictions)
        assert result.race_id == race.race_id

    def test_data_completeness_zero_when_no_data(self) -> None:
        """オッズも馬体重もない場合data_completenessが0"""
        evaluator = RaceEvaluator()
        race = _make_race(odds_available=False, weight_available=False)
        predictions = np.array([0.1] * 10)
        result = evaluator.evaluate(race, predictions)
        assert result.factors["data_completeness"] == 0.0

    def test_data_completeness_full_when_all_data(self) -> None:
        """全データがある場合data_completenessが1.0"""
        evaluator = RaceEvaluator()
        race = _make_race(odds_available=True, weight_available=True)
        predictions = np.array([0.1] * 10)
        result = evaluator.evaluate(race, predictions)
        assert result.factors["data_completeness"] == 1.0


class TestRaceEvaluatorEvaluateRaceDay:
    """evaluate_race_day メソッドのテスト"""

    def test_empty_list(self) -> None:
        """空リストの場合そのまま返す"""
        evaluator = RaceEvaluator()
        result = evaluator.evaluate_race_day([])
        assert result == []

    def test_all_below_threshold_marks_all_as_skip(self) -> None:
        """全レースが閾値以下の場合、全レースが見送りになる"""
        evaluator = RaceEvaluator(confidence_threshold=80)
        evaluations = [
            RaceEvaluation(
                race_id=f"race{i}",
                confidence_score=30,
                should_bet=False,
                skip_reason="見送り: 総合スコアが閾値未満",
                factors={
                    "unpredictability": 0.8,
                    "strength_gap": 0.2,
                    "data_completeness": 0.5,
                },
            )
            for i in range(3)
        ]
        result = evaluator.evaluate_race_day(evaluations)
        assert all(not e.should_bet for e in result)
        assert all("全レース見送り" in (e.skip_reason or "") for e in result)

    def test_some_above_threshold_returns_unchanged(self) -> None:
        """一部が閾値以上の場合、変更なしで返す"""
        evaluator = RaceEvaluator(confidence_threshold=50)
        evaluations = [
            RaceEvaluation(
                race_id="race1",
                confidence_score=60,
                should_bet=True,
                skip_reason=None,
                factors={
                    "unpredictability": 0.3,
                    "strength_gap": 0.7,
                    "data_completeness": 0.9,
                },
            ),
            RaceEvaluation(
                race_id="race2",
                confidence_score=30,
                should_bet=False,
                skip_reason="見送り: 荒れやすさが高い",
                factors={
                    "unpredictability": 0.9,
                    "strength_gap": 0.1,
                    "data_completeness": 0.5,
                },
            ),
        ]
        result = evaluator.evaluate_race_day(evaluations)
        assert result[0].should_bet is True
        assert result[1].should_bet is False

    def test_preserves_race_ids(self) -> None:
        """レースIDが保持されること"""
        evaluator = RaceEvaluator()
        evaluations = [
            RaceEvaluation(
                race_id=f"race{i}",
                confidence_score=20,
                should_bet=False,
                skip_reason="見送り: 総合スコアが閾値未満",
                factors={
                    "unpredictability": 0.8,
                    "strength_gap": 0.1,
                    "data_completeness": 0.3,
                },
            )
            for i in range(5)
        ]
        result = evaluator.evaluate_race_day(evaluations)
        for i, e in enumerate(result):
            assert e.race_id == f"race{i}"
