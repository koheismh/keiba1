"""レース評価・見送り判定のプロパティベーステスト。

Feature: horse-race-predictor, Property 6: 信頼度スコアの範囲と閾値判定

Validates: Requirements 3.1, 3.2, 3.3, 3.4
"""

from datetime import date

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

from src.data.models import HorseEntry, RaceData, TrackCondition
from src.evaluation.race_evaluator import RaceEvaluator


# --- Strategies ---


def horse_entry_strategy() -> st.SearchStrategy[HorseEntry]:
    """ランダムなHorseEntryを生成するストラテジ。

    オッズや馬体重の有無をランダムに変化させ、
    data_completeness要素に多様性を持たせる。
    """
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


def race_data_strategy() -> st.SearchStrategy[RaceData]:
    """ランダムなRaceDataを生成するストラテジ。

    出走馬数を2〜18頭の範囲で変化させ、
    様々なレース構成での信頼度スコア算出をテストする。
    """
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
        entries=st.lists(horse_entry_strategy(), min_size=2, max_size=18),
        results=st.none(),
        payouts=st.none(),
    )


def prediction_array_strategy(n_horses: int) -> st.SearchStrategy[np.ndarray]:
    """有効な確率分布（合計≈1.0、各値∈(0,1]）を生成するストラテジ。

    Dirichlet分布的なアプローチで確率を生成し、合計が1.0になるよう正規化する。
    """
    return st.lists(
        st.floats(min_value=0.01, max_value=10.0, allow_nan=False, allow_infinity=False),
        min_size=n_horses,
        max_size=n_horses,
    ).map(lambda values: np.array(values) / sum(values))


@st.composite
def race_and_predictions_strategy(draw: st.DrawFn) -> tuple[RaceData, np.ndarray]:
    """RaceDataとそれに対応する確率分布をペアで生成するストラテジ。

    出走馬数に合致した確率分布配列を生成する。
    """
    race = draw(race_data_strategy())
    n_horses = len(race.entries)
    predictions = draw(prediction_array_strategy(n_horses))
    return race, predictions


# --- Property Tests ---


class TestConfidenceScoreRangeAndThreshold:
    """Property 6: 信頼度スコアの範囲と閾値判定

    Feature: horse-race-predictor, Property 6: 信頼度スコアの範囲と閾値判定

    For any レース入力に対して、Race_Evaluatorが算出する信頼度スコアは[0, 100]の
    整数であること。かつ、for any 信頼度スコアと閾値の組み合わせに対して、
    スコア ≥ 閾値のとき投資判定がTrue、スコア < 閾値のとき投資判定がFalseであること。

    Validates: Requirements 3.1, 3.2, 3.3, 3.4
    """

    @settings(max_examples=100)
    @given(data=race_and_predictions_strategy())
    def test_confidence_score_is_integer_in_range(
        self,
        data: tuple[RaceData, np.ndarray],
    ) -> None:
        """信頼度スコアが[0, 100]の整数であることを検証する。

        Validates: Requirements 3.1
        """
        race, predictions = data
        evaluator = RaceEvaluator(confidence_threshold=50)

        evaluation = evaluator.evaluate(race, predictions)

        # スコアがint型であること
        assert isinstance(evaluation.confidence_score, int), (
            f"信頼度スコアが整数ではない: "
            f"type={type(evaluation.confidence_score)}, "
            f"value={evaluation.confidence_score}"
        )
        # スコアが[0, 100]の範囲内であること
        assert 0 <= evaluation.confidence_score <= 100, (
            f"信頼度スコアが[0, 100]の範囲外: "
            f"score={evaluation.confidence_score}"
        )

    @settings(max_examples=100)
    @given(
        data=race_and_predictions_strategy(),
        threshold=st.integers(min_value=0, max_value=100),
    )
    def test_threshold_judgment_correctness(
        self,
        data: tuple[RaceData, np.ndarray],
        threshold: int,
    ) -> None:
        """スコア ≥ 閾値のとき投資判定がTrue、スコア < 閾値のとき投資判定がFalse
        であることを検証する。

        Validates: Requirements 3.2, 3.3
        """
        race, predictions = data
        evaluator = RaceEvaluator(confidence_threshold=threshold)

        evaluation = evaluator.evaluate(race, predictions)

        expected_should_bet = evaluation.confidence_score >= threshold
        assert evaluation.should_bet == expected_should_bet, (
            f"閾値判定が不正: "
            f"score={evaluation.confidence_score}, "
            f"threshold={threshold}, "
            f"should_bet={evaluation.should_bet}, "
            f"expected={expected_should_bet}"
        )

    @settings(max_examples=100)
    @given(data=race_and_predictions_strategy())
    def test_factors_dict_contains_required_keys(
        self,
        data: tuple[RaceData, np.ndarray],
    ) -> None:
        """factors辞書に"unpredictability", "strength_gap", "data_completeness"の
        3つのキーが含まれ、すべて[0, 1]の範囲内であることを検証する。

        Validates: Requirements 3.1, 3.4
        """
        race, predictions = data
        evaluator = RaceEvaluator(confidence_threshold=50)

        evaluation = evaluator.evaluate(race, predictions)

        required_keys = {"unpredictability", "strength_gap", "data_completeness"}

        # 必要なキーがすべて含まれること
        assert required_keys.issubset(evaluation.factors.keys()), (
            f"factors辞書に必要なキーが不足: "
            f"keys={set(evaluation.factors.keys())}, "
            f"required={required_keys}"
        )

        # 各値が[0, 1]の範囲内であること
        for key in required_keys:
            value = evaluation.factors[key]
            assert 0.0 <= value <= 1.0, (
                f"factors['{key}']が[0, 1]の範囲外: value={value}"
            )

    @settings(max_examples=100)
    @given(
        data=race_and_predictions_strategy(),
        threshold=st.integers(min_value=0, max_value=100),
    )
    def test_skip_reason_present_when_not_betting(
        self,
        data: tuple[RaceData, np.ndarray],
        threshold: int,
    ) -> None:
        """投資判定がFalseの場合、見送り理由が設定されていることを検証する。

        Validates: Requirements 3.2
        """
        race, predictions = data
        evaluator = RaceEvaluator(confidence_threshold=threshold)

        evaluation = evaluator.evaluate(race, predictions)

        if not evaluation.should_bet:
            assert evaluation.skip_reason is not None, (
                f"投資判定がFalseだが見送り理由が設定されていない: "
                f"score={evaluation.confidence_score}, "
                f"threshold={threshold}"
            )
            assert len(evaluation.skip_reason) > 0, (
                f"見送り理由が空文字列: "
                f"score={evaluation.confidence_score}, "
                f"threshold={threshold}"
            )
