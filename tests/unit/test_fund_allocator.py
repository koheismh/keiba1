"""FundAllocator のユニットテスト。"""

import pytest

from src.betting.fund_allocator import FundAllocator
from src.data.models import AllocatedBet, BetRecommendation, BetType


@pytest.fixture
def allocator() -> FundAllocator:
    """デフォルト設定のFundAllocatorを返す。"""
    return FundAllocator(max_single_bet_ratio=0.3)


@pytest.fixture
def sample_bet() -> BetRecommendation:
    """テスト用のBetRecommendationを返す。"""
    return BetRecommendation(
        bet_type=BetType.WIN,
        combination=(1,),
        estimated_probability=0.3,
        estimated_odds=5.0,
        expected_value=1.5,
    )


class TestApplyKellyCriterion:
    """apply_kelly_criterion メソッドのテスト。"""

    def test_positive_kelly_fraction(self, allocator: FundAllocator) -> None:
        """ケリー基準が正の場合、100円単位の金額を返す。"""
        # Kelly = (0.3 * 5.0 - 1) / (5.0 - 1) = 0.5 / 4.0 = 0.125
        # Amount = floor(0.125 * 10000 / 100) * 100 = floor(12.5) * 100 = 1200
        amount = allocator.apply_kelly_criterion(0.3, 5.0, 10000)
        assert amount == 1200

    def test_negative_kelly_fraction_returns_zero(self, allocator: FundAllocator) -> None:
        """ケリー基準が負の場合、0を返す（ベットしない）。"""
        # Kelly = (0.1 * 2.0 - 1) / (2.0 - 1) = -0.8 / 1.0 = -0.8
        amount = allocator.apply_kelly_criterion(0.1, 2.0, 10000)
        assert amount == 0

    def test_exact_breakeven_returns_zero(self, allocator: FundAllocator) -> None:
        """期待値がちょうど1.0（ケリー=0）の場合、0を返す。"""
        # Kelly = (0.5 * 2.0 - 1) / (2.0 - 1) = 0 / 1 = 0
        amount = allocator.apply_kelly_criterion(0.5, 2.0, 10000)
        assert amount == 0

    def test_odds_one_returns_zero(self, allocator: FundAllocator) -> None:
        """オッズが1.0以下の場合、0を返す。"""
        amount = allocator.apply_kelly_criterion(0.5, 1.0, 10000)
        assert amount == 0

    def test_zero_probability_returns_zero(self, allocator: FundAllocator) -> None:
        """確率が0の場合、0を返す。"""
        amount = allocator.apply_kelly_criterion(0.0, 5.0, 10000)
        assert amount == 0

    def test_zero_budget_returns_zero(self, allocator: FundAllocator) -> None:
        """予算が0の場合、0を返す。"""
        amount = allocator.apply_kelly_criterion(0.3, 5.0, 0)
        assert amount == 0

    def test_result_is_multiple_of_100(self, allocator: FundAllocator) -> None:
        """結果は必ず100の倍数であること。"""
        amount = allocator.apply_kelly_criterion(0.4, 3.0, 7777)
        assert amount % 100 == 0

    def test_high_probability_high_odds(self, allocator: FundAllocator) -> None:
        """高確率・高オッズの場合の計算。"""
        # Kelly = (0.5 * 10.0 - 1) / (10.0 - 1) = 4.0 / 9.0 ≈ 0.444
        # Amount = floor(0.444 * 10000 / 100) * 100 = floor(44.4) * 100 = 4400
        amount = allocator.apply_kelly_criterion(0.5, 10.0, 10000)
        assert amount == 4400


class TestCapAllocation:
    """cap_allocation メソッドのテスト。"""

    def test_no_cap_needed(self, allocator: FundAllocator) -> None:
        """全ての配分が上限以下の場合、変更なし。"""
        bet = BetRecommendation(
            bet_type=BetType.WIN,
            combination=(1,),
            estimated_probability=0.3,
            estimated_odds=5.0,
            expected_value=1.5,
        )
        allocations = [
            AllocatedBet(recommendation=bet, amount=2000),
            AllocatedBet(recommendation=bet, amount=1000),
        ]
        result = allocator.cap_allocation(allocations, 10000, 0.3)
        assert result[0].amount == 2000
        assert result[1].amount == 1000

    def test_cap_applied(self, allocator: FundAllocator) -> None:
        """配分が上限を超える場合、キャップが適用される。"""
        bet = BetRecommendation(
            bet_type=BetType.WIN,
            combination=(1,),
            estimated_probability=0.3,
            estimated_odds=5.0,
            expected_value=1.5,
        )
        allocations = [
            AllocatedBet(recommendation=bet, amount=5000),  # > 3000 (30% of 10000)
            AllocatedBet(recommendation=bet, amount=1000),
        ]
        result = allocator.cap_allocation(allocations, 10000, 0.3)
        # First bet capped at floor(10000 * 0.3 / 100) * 100 = 3000
        assert result[0].amount == 3000
        # Second bet gets redistribution
        assert result[1].amount >= 1000

    def test_all_amounts_within_cap(self, allocator: FundAllocator) -> None:
        """キャップ後、全ての配分が上限以下であること。"""
        bet = BetRecommendation(
            bet_type=BetType.WIN,
            combination=(1,),
            estimated_probability=0.3,
            estimated_odds=5.0,
            expected_value=1.5,
        )
        allocations = [
            AllocatedBet(recommendation=bet, amount=8000),
            AllocatedBet(recommendation=bet, amount=6000),
            AllocatedBet(recommendation=bet, amount=2000),
        ]
        result = allocator.cap_allocation(allocations, 10000, 0.3)
        max_allowed = 3000  # floor(10000 * 0.3 / 100) * 100
        for alloc in result:
            assert alloc.amount <= max_allowed

    def test_amounts_are_multiples_of_100(self, allocator: FundAllocator) -> None:
        """キャップ後も全ての金額が100の倍数であること。"""
        bet = BetRecommendation(
            bet_type=BetType.WIN,
            combination=(1,),
            estimated_probability=0.3,
            estimated_odds=5.0,
            expected_value=1.5,
        )
        allocations = [
            AllocatedBet(recommendation=bet, amount=5000),
            AllocatedBet(recommendation=bet, amount=3500),
        ]
        result = allocator.cap_allocation(allocations, 10000, 0.3)
        for alloc in result:
            assert alloc.amount % 100 == 0

    def test_empty_allocations(self, allocator: FundAllocator) -> None:
        """空リストの場合、空リストを返す。"""
        result = allocator.cap_allocation([], 10000, 0.3)
        assert result == []


class TestAllocate:
    """allocate メソッドのテスト。"""

    def test_basic_allocation(self, allocator: FundAllocator) -> None:
        """基本的な配分テスト。"""
        bets = [
            BetRecommendation(
                bet_type=BetType.WIN,
                combination=(1,),
                estimated_probability=0.3,
                estimated_odds=5.0,
                expected_value=1.5,
            ),
            BetRecommendation(
                bet_type=BetType.WIN,
                combination=(2,),
                estimated_probability=0.25,
                estimated_odds=6.0,
                expected_value=1.5,
            ),
        ]
        result = allocator.allocate(bets, 10000)
        assert len(result) > 0
        # All amounts are multiples of 100
        for alloc in result:
            assert alloc.amount % 100 == 0
        # Total doesn't exceed budget
        assert sum(a.amount for a in result) <= 10000

    def test_total_within_budget(self, allocator: FundAllocator) -> None:
        """合計が予算を超えないこと。"""
        bets = [
            BetRecommendation(
                bet_type=BetType.WIN,
                combination=(i,),
                estimated_probability=0.4,
                estimated_odds=5.0,
                expected_value=2.0,
            )
            for i in range(1, 6)
        ]
        result = allocator.allocate(bets, 10000)
        total = sum(a.amount for a in result)
        assert total <= 10000

    def test_no_single_bet_exceeds_max_ratio(self, allocator: FundAllocator) -> None:
        """単一買い目が予算の30%を超えないこと。"""
        bets = [
            BetRecommendation(
                bet_type=BetType.WIN,
                combination=(1,),
                estimated_probability=0.8,
                estimated_odds=10.0,
                expected_value=8.0,
            ),
            BetRecommendation(
                bet_type=BetType.WIN,
                combination=(2,),
                estimated_probability=0.1,
                estimated_odds=15.0,
                expected_value=1.5,
            ),
        ]
        result = allocator.allocate(bets, 10000)
        max_allowed = 3000  # 10000 * 0.3
        for alloc in result:
            assert alloc.amount <= max_allowed

    def test_empty_bets_returns_empty(self, allocator: FundAllocator) -> None:
        """空の買い目リストでは空リストを返す。"""
        result = allocator.allocate([], 10000)
        assert result == []

    def test_zero_budget_returns_empty(self, allocator: FundAllocator) -> None:
        """予算0の場合、空リストを返す。"""
        bets = [
            BetRecommendation(
                bet_type=BetType.WIN,
                combination=(1,),
                estimated_probability=0.3,
                estimated_odds=5.0,
                expected_value=1.5,
            )
        ]
        result = allocator.allocate(bets, 0)
        assert result == []

    def test_negative_expected_value_excluded(self, allocator: FundAllocator) -> None:
        """ケリー基準が負の買い目は除外される。"""
        bets = [
            BetRecommendation(
                bet_type=BetType.WIN,
                combination=(1,),
                estimated_probability=0.1,
                estimated_odds=2.0,
                expected_value=0.2,
            )
        ]
        result = allocator.allocate(bets, 10000)
        assert result == []

    def test_custom_max_ratio(self) -> None:
        """カスタムmax_ratioが適用されること。"""
        allocator = FundAllocator(max_single_bet_ratio=0.5)
        bets = [
            BetRecommendation(
                bet_type=BetType.WIN,
                combination=(1,),
                estimated_probability=0.8,
                estimated_odds=10.0,
                expected_value=8.0,
            )
        ]
        result = allocator.allocate(bets, 10000)
        # Max allowed is 5000 (50% of 10000)
        for alloc in result:
            assert alloc.amount <= 5000

    def test_all_amounts_multiples_of_100(self, allocator: FundAllocator) -> None:
        """全ての配分金額が100の倍数であること。"""
        bets = [
            BetRecommendation(
                bet_type=BetType.WIN,
                combination=(i,),
                estimated_probability=0.25 + i * 0.05,
                estimated_odds=4.0 + i,
                expected_value=1.5,
            )
            for i in range(1, 4)
        ]
        result = allocator.allocate(bets, 7777)
        for alloc in result:
            assert alloc.amount % 100 == 0
