"""券種ごとの的中判定ロジック"""

from src.data.models import BetType, RaceResult


class HitJudge:
    """券種ごとの的中判定ロジック"""

    @staticmethod
    def is_hit(
        bet_type: BetType, combination: tuple[int, ...], results: list[RaceResult]
    ) -> bool:
        """Determine if a bet is a hit based on the race results.

        Args:
            bet_type: 券種
            combination: 馬番の組み合わせ
            results: レース結果（着順付き）

        Returns:
            True if the bet is a hit, False otherwise
        """
        if not results or not combination:
            return False

        if bet_type == BetType.WIN:
            # 単勝: 1着一致
            top1 = HitJudge._get_top_n(results, 1)
            return combination[0] == top1[0]

        elif bet_type == BetType.PLACE:
            # 複勝: 3着以内一致
            top3 = HitJudge._get_top_n(results, 3)
            return combination[0] in top3

        elif bet_type == BetType.QUINELLA:
            # 馬連: 1-2着の組み合わせ一致（順不同）
            top2 = HitJudge._get_top_n(results, 2)
            return set(combination) == set(top2)

        elif bet_type == BetType.EXACTA:
            # 馬単: 1-2着の順序一致
            top2 = HitJudge._get_top_n(results, 2)
            return combination[0] == top2[0] and combination[1] == top2[1]

        elif bet_type == BetType.WIDE:
            # ワイド: 3着以内2頭の組み合わせ
            top3 = HitJudge._get_top_n(results, 3)
            return all(horse in top3 for horse in combination)

        elif bet_type == BetType.TRIO:
            # 三連複: 1-3着の組み合わせ一致（順不同）
            top3 = HitJudge._get_top_n(results, 3)
            return set(combination) == set(top3)

        elif bet_type == BetType.TRIFECTA:
            # 三連単: 1-3着の順序一致
            top3 = HitJudge._get_top_n(results, 3)
            return (
                combination[0] == top3[0]
                and combination[1] == top3[1]
                and combination[2] == top3[2]
            )

        return False

    @staticmethod
    def _get_top_n(results: list[RaceResult], n: int) -> list[int]:
        """Get horse numbers for the top N finishers.

        Args:
            results: レース結果リスト
            n: 上位何着まで取得するか

        Returns:
            上位N着の馬番リスト（着順順）
        """
        sorted_results = sorted(results, key=lambda r: r.finish_position)
        return [r.horse_number for r in sorted_results[:n]]
