"""買い目選択モジュールのプロパティベーステスト。

Feature: horse-race-predictor, Property 5: 期待値計算の正確性
Feature: horse-race-predictor, Property 7: 買い目選択の制約

Validates: Requirements 2.3, 4.1, 4.2, 4.3, 4.4, 4.6
"""

from datetime import date

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

from src.betting.bet_selector import BetSelector
from src.data.models import BetRecommendation, BetType, HorseEntry, RaceData, TrackCondition


# --- Strategies ---


def horse_entry_with_odds_strategy(horse_number: int) -> st.SearchStrategy[HorseEntry]:
    """有効なオッズを持つHorseEntryを生成するストラテジ。

    buy_目選択では有効なオッズが必須のため、win_oddsは必ず正の値を設定する。
    """
    return st.builds(
        HorseEntry,
        horse_name=st.text(min_size=1, max_size=10, alphabet=st.characters(categories=("L",))),
        jockey_name=st.text(min_size=1, max_size=10, alphabet=st.characters(categories=("L",))),
        gate_number=st.integers(min_value=1, max_value=8),
        horse_number=st.just(horse_number),
        weight=st.integers(min_value=400, max_value=600),
        weight_change=st.integers(min_value=-20, max_value=20),
        win_odds=st.floats(min_value=1.1, max_value=200.0, allow_nan=False, allow_infinity=False),
    )


@st.composite
def race_with_valid_entries_strategy(draw: st.DrawFn) -> RaceData:
    """有効なオッズを持つ出走馬を含むRaceDataを生成するストラテジ。

    出走馬数は3〜18頭の範囲で生成し、各馬に一意な馬番を割り当てる。
    """
    n_entries = draw(st.integers(min_value=3, max_value=18))
    entries = []
    for i in range(n_entries):
        entry = draw(horse_entry_with_odds_strategy(horse_number=i + 1))
        entries.append(entry)

    race = RaceData(
        race_id=draw(st.text(min_size=1, max_size=12, alphabet=st.characters(categories=("L", "N")))),
        race_name=draw(st.text(min_size=1, max_size=20, alphabet=st.characters(categories=("L",)))),
        race_date=draw(st.dates(min_value=date(2020, 1, 1), max_value=date(2024, 12, 31))),
        post_time=None,
        venue=draw(st.text(min_size=1, max_size=10, alphabet=st.characters(categories=("L",)))),
        course_type=draw(st.sampled_from(["芝", "ダート"])),
        distance=draw(st.integers(min_value=1000, max_value=3600)),
        track_condition=draw(st.sampled_from(list(TrackCondition))),
        weather=None,
        entries=entries,
        results=None,
        payouts=None,
    )
    return race


@st.composite
def probability_distribution_strategy(draw: st.DrawFn, n_horses: int) -> np.ndarray:
    """有効な確率分布（合計≈1.0、各値∈(0,1)）を生成するストラテジ。

    各馬に正の確率を割り当て、合計が1.0になるよう正規化する。
    """
    raw_values = draw(
        st.lists(
            st.floats(min_value=0.01, max_value=10.0, allow_nan=False, allow_infinity=False),
            min_size=n_horses,
            max_size=n_horses,
        )
    )
    arr = np.array(raw_values)
    return arr / arr.sum()


@st.composite
def race_and_probabilities_strategy(draw: st.DrawFn) -> tuple[RaceData, np.ndarray]:
    """RaceDataとそれに対応する確率分布をペアで生成するストラテジ。"""
    race = draw(race_with_valid_entries_strategy())
    n_horses = len(race.entries)
    probabilities = draw(probability_distribution_strategy(n_horses))
    return race, probabilities


# --- Property 7 Tests ---


class TestBetSelectionConstraints:
    """Property 7: 買い目選択の制約

    Feature: horse-race-predictor, Property 7: 買い目選択の制約

    For any レース予測結果に対して、Bet_Selectorが出力する買い目は以下をすべて満たすこと：
    (1) 件数が設定された最大買い目数以下、
    (2) すべての買い目の期待値が設定された最低基準を超える、
    (3) 各買い目に券種・馬番組み合わせ・推定的中確率・推定オッズ・期待値が含まれる、
    (4) 買い目は期待値の降順でソートされていること。

    Validates: Requirements 4.1, 4.2, 4.3, 4.4, 4.6
    """

    @settings(max_examples=100)
    @given(
        data=race_and_probabilities_strategy(),
        max_bets=st.integers(min_value=1, max_value=10),
        min_expected_value=st.floats(min_value=0.5, max_value=2.0, allow_nan=False, allow_infinity=False),
    )
    def test_bet_count_within_max_limit(
        self,
        data: tuple[RaceData, np.ndarray],
        max_bets: int,
        min_expected_value: float,
    ) -> None:
        """(1) 件数が設定された最大買い目数以下であることを検証する。

        Validates: Requirements 4.1, 4.4
        """
        race, probabilities = data
        selector = BetSelector(min_expected_value=min_expected_value)

        bets = selector.select_bets(race, probabilities, max_bets=max_bets)

        assert len(bets) <= max_bets, (
            f"買い目件数が最大買い目数を超過: "
            f"件数={len(bets)}, max_bets={max_bets}"
        )

    @settings(max_examples=100)
    @given(
        data=race_and_probabilities_strategy(),
        max_bets=st.integers(min_value=1, max_value=10),
        min_expected_value=st.floats(min_value=0.5, max_value=2.0, allow_nan=False, allow_infinity=False),
    )
    def test_all_bets_exceed_minimum_expected_value(
        self,
        data: tuple[RaceData, np.ndarray],
        max_bets: int,
        min_expected_value: float,
    ) -> None:
        """(2) すべての買い目の期待値が設定された最低基準を超えることを検証する。

        Validates: Requirements 4.2
        """
        race, probabilities = data
        selector = BetSelector(min_expected_value=min_expected_value)

        bets = selector.select_bets(race, probabilities, max_bets=max_bets)

        for i, bet in enumerate(bets):
            assert bet.expected_value > min_expected_value, (
                f"買い目[{i}]の期待値が最低基準以下: "
                f"expected_value={bet.expected_value}, "
                f"min_expected_value={min_expected_value}, "
                f"bet_type={bet.bet_type.value}, "
                f"combination={bet.combination}"
            )

    @settings(max_examples=100)
    @given(
        data=race_and_probabilities_strategy(),
        max_bets=st.integers(min_value=1, max_value=10),
        min_expected_value=st.floats(min_value=0.5, max_value=2.0, allow_nan=False, allow_infinity=False),
    )
    def test_all_bets_contain_required_fields(
        self,
        data: tuple[RaceData, np.ndarray],
        max_bets: int,
        min_expected_value: float,
    ) -> None:
        """(3) 各買い目に券種・馬番組み合わせ・推定的中確率・推定オッズ・期待値が
        含まれることを検証する。

        Validates: Requirements 4.3
        """
        race, probabilities = data
        selector = BetSelector(min_expected_value=min_expected_value)

        bets = selector.select_bets(race, probabilities, max_bets=max_bets)

        for i, bet in enumerate(bets):
            # 券種が有効なBetTypeであること
            assert isinstance(bet.bet_type, BetType), (
                f"買い目[{i}]のbet_typeがBetType型ではない: "
                f"type={type(bet.bet_type)}"
            )

            # 馬番組み合わせがタプルで、少なくとも1つの馬番を含むこと
            assert isinstance(bet.combination, tuple), (
                f"買い目[{i}]のcombinationがtuple型ではない: "
                f"type={type(bet.combination)}"
            )
            assert len(bet.combination) >= 1, (
                f"買い目[{i}]のcombinationが空: "
                f"combination={bet.combination}"
            )
            for horse_num in bet.combination:
                assert isinstance(horse_num, int) and horse_num > 0, (
                    f"買い目[{i}]のcombinationに無効な馬番: "
                    f"horse_num={horse_num}"
                )

            # 推定的中確率が(0, 1]の範囲内であること
            assert 0 < bet.estimated_probability <= 1.0, (
                f"買い目[{i}]の推定的中確率が範囲外: "
                f"estimated_probability={bet.estimated_probability}"
            )

            # 推定オッズが正の値であること
            assert bet.estimated_odds > 0, (
                f"買い目[{i}]の推定オッズが正でない: "
                f"estimated_odds={bet.estimated_odds}"
            )

            # 期待値が正の値であること
            assert bet.expected_value > 0, (
                f"買い目[{i}]の期待値が正でない: "
                f"expected_value={bet.expected_value}"
            )

    @settings(max_examples=100)
    @given(
        data=race_and_probabilities_strategy(),
        max_bets=st.integers(min_value=1, max_value=10),
        min_expected_value=st.floats(min_value=0.5, max_value=2.0, allow_nan=False, allow_infinity=False),
    )
    def test_bets_sorted_by_expected_value_descending(
        self,
        data: tuple[RaceData, np.ndarray],
        max_bets: int,
        min_expected_value: float,
    ) -> None:
        """(4) 買い目は期待値の降順でソートされていることを検証する。

        Validates: Requirements 4.6
        """
        race, probabilities = data
        selector = BetSelector(min_expected_value=min_expected_value)

        bets = selector.select_bets(race, probabilities, max_bets=max_bets)

        for i in range(len(bets) - 1):
            assert bets[i].expected_value >= bets[i + 1].expected_value, (
                f"買い目が期待値の降順でソートされていない: "
                f"bets[{i}].expected_value={bets[i].expected_value}, "
                f"bets[{i + 1}].expected_value={bets[i + 1].expected_value}"
            )
