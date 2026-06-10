"""資金配分モジュールのプロパティベーステスト。

Feature: horse-race-predictor, Property 8: 資金配分不変量

Validates: Requirements 5.1, 5.2, 5.3, 5.5
"""

from hypothesis import given, settings
from hypothesis import strategies as st

from src.betting.fund_allocator import FundAllocator
from src.data.models import BetRecommendation, BetType


# --- Strategies ---


@st.composite
def bet_recommendation_strategy(draw: st.DrawFn) -> BetRecommendation:
    """有効なBetRecommendationを生成するストラテジ。

    ケリー基準で正の配分が得られるよう、probability * odds > 1 となる
    確率とオッズの組み合わせを生成する。
    """
    bet_type = draw(st.sampled_from(list(BetType)))
    # 馬番の組み合わせ（1〜18の範囲で1〜3頭）
    n_horses = draw(st.integers(min_value=1, max_value=3))
    combination = tuple(
        draw(
            st.lists(
                st.integers(min_value=1, max_value=18),
                min_size=n_horses,
                max_size=n_horses,
                unique=True,
            )
        )
    )

    # 確率とオッズはケリー基準で正の配分が出るよう設定
    # kelly_fraction = (p * odds - 1) / (odds - 1) > 0 => p * odds > 1
    probability = draw(
        st.floats(min_value=0.05, max_value=0.8, allow_nan=False, allow_infinity=False)
    )
    # odds > 1/probability を保証してケリー基準で正の配分を得る
    min_odds = 1.0 / probability + 0.1
    odds = draw(
        st.floats(
            min_value=min_odds,
            max_value=max(min_odds + 50.0, 100.0),
            allow_nan=False,
            allow_infinity=False,
        )
    )
    expected_value = probability * odds

    return BetRecommendation(
        bet_type=bet_type,
        combination=combination,
        estimated_probability=probability,
        estimated_odds=odds,
        expected_value=expected_value,
    )


@st.composite
def bet_list_strategy(draw: st.DrawFn) -> list[BetRecommendation]:
    """1〜10件のBetRecommendationリストを生成するストラテジ。"""
    n_bets = draw(st.integers(min_value=1, max_value=10))
    bets = [draw(bet_recommendation_strategy()) for _ in range(n_bets)]
    return bets


# --- Property 8 Tests ---


class TestFundAllocationInvariants:
    """Property 8: 資金配分不変量

    Feature: horse-race-predictor, Property 8: 資金配分不変量

    For any 買い目リストと予算に対して、Fund_Allocatorの出力は以下をすべて満たすこと：
    (1) 全買い目の合計投資金額が予算を超えない、
    (2) 各買い目の投資金額は100円の倍数である、
    (3) 単一買い目の投資金額は予算の30%（設定値）を超えない。

    Validates: Requirements 5.1, 5.2, 5.3, 5.5
    """

    @settings(max_examples=100)
    @given(
        bets=bet_list_strategy(),
        budget=st.integers(min_value=1000, max_value=100000),
    )
    def test_total_allocation_does_not_exceed_budget(
        self,
        bets: list[BetRecommendation],
        budget: int,
    ) -> None:
        """(1) 全買い目の合計投資金額が予算を超えないことを検証する。

        Validates: Requirements 5.1, 5.3
        """
        allocator = FundAllocator(max_single_bet_ratio=0.3)

        allocated = allocator.allocate(bets, budget)

        total = sum(a.amount for a in allocated)
        assert total <= budget, (
            f"合計投資金額が予算を超過: "
            f"total={total}, budget={budget}, "
            f"bets={len(bets)}, allocated={len(allocated)}"
        )

    @settings(max_examples=100)
    @given(
        bets=bet_list_strategy(),
        budget=st.integers(min_value=1000, max_value=100000),
    )
    def test_each_amount_is_multiple_of_100(
        self,
        bets: list[BetRecommendation],
        budget: int,
    ) -> None:
        """(2) 各買い目の投資金額は100円の倍数であることを検証する。

        Validates: Requirements 5.2
        """
        allocator = FundAllocator(max_single_bet_ratio=0.3)

        allocated = allocator.allocate(bets, budget)

        for i, alloc in enumerate(allocated):
            assert alloc.amount % 100 == 0, (
                f"買い目[{i}]の投資金額が100円の倍数でない: "
                f"amount={alloc.amount}, "
                f"bet_type={alloc.recommendation.bet_type.value}, "
                f"combination={alloc.recommendation.combination}"
            )

    @settings(max_examples=100)
    @given(
        bets=bet_list_strategy(),
        budget=st.integers(min_value=1000, max_value=100000),
    )
    def test_single_bet_does_not_exceed_max_ratio(
        self,
        bets: list[BetRecommendation],
        budget: int,
    ) -> None:
        """(3) 単一買い目の投資金額は予算の30%を超えないことを検証する。

        Validates: Requirements 5.5
        """
        max_ratio = 0.3
        allocator = FundAllocator(max_single_bet_ratio=max_ratio)

        allocated = allocator.allocate(bets, budget)

        # 30%上限は100円単位に切り捨てられた値
        import math

        max_allowed = math.floor(budget * max_ratio / 100) * 100

        for i, alloc in enumerate(allocated):
            assert alloc.amount <= max_allowed, (
                f"買い目[{i}]の投資金額が予算の{max_ratio*100:.0f}%を超過: "
                f"amount={alloc.amount}, "
                f"max_allowed={max_allowed} "
                f"(budget={budget}, max_ratio={max_ratio}), "
                f"bet_type={alloc.recommendation.bet_type.value}, "
                f"combination={alloc.recommendation.combination}"
            )
