"""資金配分モジュール。

ケリー基準に基づく最適な資金配分を行う。
各買い目に対して予算内で回収率を最大化する投資金額を算出する。
"""

import math

from src.data.models import AllocatedBet, BetRecommendation


class FundAllocator:
    """ケリー基準ベースの資金配分。

    買い目リストと予算を受け取り、ケリー基準に基づいて
    各買い目への最適な投資金額を100円単位で算出する。
    単一買い目への過度な集中を防ぐため、配分上限を設ける。
    """

    def __init__(self, max_single_bet_ratio: float = 0.3) -> None:
        """Initialize with max single bet ratio.

        Args:
            max_single_bet_ratio: 単一買い目の最大配分比率（デフォルト0.3 = 30%）
        """
        self.max_single_bet_ratio = max_single_bet_ratio

    def allocate(self, bets: list[BetRecommendation], budget: int) -> list[AllocatedBet]:
        """予算内で最適な資金配分を算出する。

        1. Each bet gets Kelly criterion allocation
        2. Cap each at max_ratio * budget
        3. Round to 100 yen units
        4. If total exceeds budget, scale down proportionally
        5. Ensure total <= budget

        Args:
            bets: 買い目推奨リスト
            budget: 投資予算（円）

        Returns:
            AllocatedBet list with amounts in 100-yen units.
        """
        if not bets or budget <= 0:
            return []

        # Step 1: Calculate Kelly criterion allocation for each bet
        allocations: list[AllocatedBet] = []
        for bet in bets:
            amount = self.apply_kelly_criterion(
                bet.estimated_probability, bet.estimated_odds, budget
            )
            if amount > 0:
                allocations.append(AllocatedBet(recommendation=bet, amount=amount))

        if not allocations:
            return []

        # Step 2 & 3: Cap allocations at max_ratio * budget
        allocations = self.cap_allocation(allocations, budget, self.max_single_bet_ratio)

        # Step 4 & 5: If total exceeds budget, scale down proportionally
        total = sum(a.amount for a in allocations)
        if total > budget:
            allocations = self._scale_down(allocations, budget)

        return allocations

    def apply_kelly_criterion(self, probability: float, odds: float, budget: int) -> int:
        """ケリー基準に基づく推奨投資金額を算出する（100円単位）。

        Kelly fraction = (p * odds - 1) / (odds - 1)
        If negative or odds <= 1, return 0 (don't bet).
        Amount = floor(kelly_fraction * budget / 100) * 100

        Args:
            probability: 推定的中確率 (0 < p < 1)
            odds: 推定オッズ (odds > 1)
            budget: 投資予算（円）

        Returns:
            推奨投資金額（100円単位）
        """
        if odds <= 1.0 or probability <= 0.0 or probability >= 1.0 or budget <= 0:
            return 0

        kelly_fraction = (probability * odds - 1.0) / (odds - 1.0)

        if kelly_fraction <= 0:
            return 0

        amount = math.floor(kelly_fraction * budget / 100) * 100
        return amount

    def cap_allocation(
        self, allocations: list[AllocatedBet], budget: int, max_ratio: float = 0.3
    ) -> list[AllocatedBet]:
        """単一買い目の配分上限を適用し再配分する。

        If any bet amount > budget * max_ratio:
          - Cap it at floor(budget * max_ratio / 100) * 100
          - Redistribute excess proportionally to remaining bets
          - Repeat until no bet exceeds the cap

        Args:
            allocations: 配分済み買い目リスト
            budget: 投資予算（円）
            max_ratio: 単一買い目の最大配分比率

        Returns:
            上限適用後の配分済み買い目リスト
        """
        if not allocations or budget <= 0:
            return allocations

        max_amount = math.floor(budget * max_ratio / 100) * 100

        if max_amount <= 0:
            return [
                AllocatedBet(recommendation=a.recommendation, amount=0)
                for a in allocations
            ]

        # Iteratively cap and redistribute
        result = list(allocations)
        changed = True
        while changed:
            changed = False
            excess = 0
            capped_indices: set[int] = set()
            uncapped_indices: list[int] = []

            for i, alloc in enumerate(result):
                if alloc.amount > max_amount:
                    excess += alloc.amount - max_amount
                    capped_indices.add(i)
                    changed = True
                else:
                    uncapped_indices.append(i)

            if not changed:
                break

            # Cap the over-limit bets
            new_result: list[AllocatedBet] = []
            for i, alloc in enumerate(result):
                if i in capped_indices:
                    new_result.append(
                        AllocatedBet(recommendation=alloc.recommendation, amount=max_amount)
                    )
                else:
                    new_result.append(alloc)

            # Redistribute excess proportionally to uncapped bets
            if uncapped_indices and excess > 0:
                uncapped_total = sum(new_result[i].amount for i in uncapped_indices)
                for i in uncapped_indices:
                    if uncapped_total > 0:
                        share = excess * (new_result[i].amount / uncapped_total)
                        new_amount = math.floor((new_result[i].amount + share) / 100) * 100
                        # Ensure redistributed amount doesn't exceed cap
                        new_amount = min(new_amount, max_amount)
                        new_result[i] = AllocatedBet(
                            recommendation=new_result[i].recommendation,
                            amount=new_amount,
                        )

            result = new_result

        return result

    def _scale_down(self, allocations: list[AllocatedBet], budget: int) -> list[AllocatedBet]:
        """合計が予算を超える場合、比例的にスケールダウンする。

        Args:
            allocations: 配分済み買い目リスト
            budget: 投資予算（円）

        Returns:
            スケールダウン後の配分済み買い目リスト
        """
        total = sum(a.amount for a in allocations)
        if total <= budget:
            return allocations

        scale_factor = budget / total
        result: list[AllocatedBet] = []
        for alloc in allocations:
            new_amount = math.floor(alloc.amount * scale_factor / 100) * 100
            result.append(
                AllocatedBet(recommendation=alloc.recommendation, amount=new_amount)
            )

        # Verify total doesn't exceed budget (should be guaranteed by floor)
        final_total = sum(a.amount for a in result)
        assert final_total <= budget

        # Filter out zero-amount allocations
        result = [a for a in result if a.amount > 0]

        return result
