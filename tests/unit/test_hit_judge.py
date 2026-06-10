"""HitJudge のユニットテスト。"""

import pytest

from src.backtest.hit_judge import HitJudge
from src.data.models import BetType, RaceResult


def _make_results(positions: list[tuple[int, int]]) -> list[RaceResult]:
    """テスト用のレース結果を作成する。

    Args:
        positions: (馬番, 着順) のリスト
    """
    return [RaceResult(horse_number=h, finish_position=p) for h, p in positions]


# 基本レース結果: 馬番3が1着, 馬番1が2着, 馬番5が3着, 馬番2が4着, 馬番4が5着
BASIC_RESULTS = _make_results([(3, 1), (1, 2), (5, 3), (2, 4), (4, 5)])


class TestWin:
    """単勝の的中判定テスト。"""

    def test_hit_when_first_place_matches(self):
        """1着馬と一致すれば的中。"""
        assert HitJudge.is_hit(BetType.WIN, (3,), BASIC_RESULTS) is True

    def test_miss_when_first_place_does_not_match(self):
        """1着馬と一致しなければ不的中。"""
        assert HitJudge.is_hit(BetType.WIN, (1,), BASIC_RESULTS) is False

    def test_miss_with_second_place_horse(self):
        """2着馬は不的中。"""
        assert HitJudge.is_hit(BetType.WIN, (1,), BASIC_RESULTS) is False

    def test_miss_with_last_place_horse(self):
        """最下位馬は不的中。"""
        assert HitJudge.is_hit(BetType.WIN, (4,), BASIC_RESULTS) is False


class TestPlace:
    """複勝の的中判定テスト。"""

    def test_hit_with_first_place(self):
        """1着馬で的中。"""
        assert HitJudge.is_hit(BetType.PLACE, (3,), BASIC_RESULTS) is True

    def test_hit_with_second_place(self):
        """2着馬で的中。"""
        assert HitJudge.is_hit(BetType.PLACE, (1,), BASIC_RESULTS) is True

    def test_hit_with_third_place(self):
        """3着馬で的中。"""
        assert HitJudge.is_hit(BetType.PLACE, (5,), BASIC_RESULTS) is True

    def test_miss_with_fourth_place(self):
        """4着馬は不的中。"""
        assert HitJudge.is_hit(BetType.PLACE, (2,), BASIC_RESULTS) is False

    def test_miss_with_fifth_place(self):
        """5着馬は不的中。"""
        assert HitJudge.is_hit(BetType.PLACE, (4,), BASIC_RESULTS) is False


class TestQuinella:
    """馬連の的中判定テスト。"""

    def test_hit_with_correct_combination(self):
        """1-2着の組み合わせで的中。"""
        assert HitJudge.is_hit(BetType.QUINELLA, (3, 1), BASIC_RESULTS) is True

    def test_hit_with_reversed_order(self):
        """1-2着の組み合わせ（逆順）でも的中。"""
        assert HitJudge.is_hit(BetType.QUINELLA, (1, 3), BASIC_RESULTS) is True

    def test_miss_with_first_and_third(self):
        """1着と3着の組み合わせは不的中。"""
        assert HitJudge.is_hit(BetType.QUINELLA, (3, 5), BASIC_RESULTS) is False

    def test_miss_with_second_and_third(self):
        """2着と3着の組み合わせは不的中。"""
        assert HitJudge.is_hit(BetType.QUINELLA, (1, 5), BASIC_RESULTS) is False

    def test_miss_with_unplaced_horses(self):
        """着外馬の組み合わせは不的中。"""
        assert HitJudge.is_hit(BetType.QUINELLA, (2, 4), BASIC_RESULTS) is False


class TestExacta:
    """馬単の的中判定テスト。"""

    def test_hit_with_correct_order(self):
        """1着→2着の順序で的中。"""
        assert HitJudge.is_hit(BetType.EXACTA, (3, 1), BASIC_RESULTS) is True

    def test_miss_with_reversed_order(self):
        """2着→1着の順序は不的中。"""
        assert HitJudge.is_hit(BetType.EXACTA, (1, 3), BASIC_RESULTS) is False

    def test_miss_with_first_and_third(self):
        """1着と3着の組み合わせは不的中。"""
        assert HitJudge.is_hit(BetType.EXACTA, (3, 5), BASIC_RESULTS) is False

    def test_miss_with_unplaced_horses(self):
        """着外馬は不的中。"""
        assert HitJudge.is_hit(BetType.EXACTA, (2, 4), BASIC_RESULTS) is False


class TestWide:
    """ワイドの的中判定テスト。"""

    def test_hit_with_first_and_second(self):
        """1着と2着で的中。"""
        assert HitJudge.is_hit(BetType.WIDE, (3, 1), BASIC_RESULTS) is True

    def test_hit_with_first_and_third(self):
        """1着と3着で的中。"""
        assert HitJudge.is_hit(BetType.WIDE, (3, 5), BASIC_RESULTS) is True

    def test_hit_with_second_and_third(self):
        """2着と3着で的中。"""
        assert HitJudge.is_hit(BetType.WIDE, (1, 5), BASIC_RESULTS) is True

    def test_hit_with_reversed_order(self):
        """順序を入れ替えても的中。"""
        assert HitJudge.is_hit(BetType.WIDE, (5, 3), BASIC_RESULTS) is True

    def test_miss_with_one_in_top3_one_out(self):
        """1頭が3着以内、1頭が着外は不的中。"""
        assert HitJudge.is_hit(BetType.WIDE, (3, 2), BASIC_RESULTS) is False

    def test_miss_with_both_out_of_top3(self):
        """両方が4着以下は不的中。"""
        assert HitJudge.is_hit(BetType.WIDE, (2, 4), BASIC_RESULTS) is False


class TestTrio:
    """三連複の的中判定テスト。"""

    def test_hit_with_correct_combination(self):
        """1-2-3着の組み合わせで的中。"""
        assert HitJudge.is_hit(BetType.TRIO, (3, 1, 5), BASIC_RESULTS) is True

    def test_hit_with_any_order(self):
        """順番を変えても的中。"""
        assert HitJudge.is_hit(BetType.TRIO, (5, 3, 1), BASIC_RESULTS) is True
        assert HitJudge.is_hit(BetType.TRIO, (1, 5, 3), BASIC_RESULTS) is True

    def test_miss_with_one_wrong_horse(self):
        """1頭でも違えば不的中。"""
        assert HitJudge.is_hit(BetType.TRIO, (3, 1, 2), BASIC_RESULTS) is False

    def test_miss_with_all_wrong_horses(self):
        """全馬違えば不的中。"""
        assert HitJudge.is_hit(BetType.TRIO, (2, 4, 6), BASIC_RESULTS) is False


class TestTrifecta:
    """三連単の的中判定テスト。"""

    def test_hit_with_exact_order(self):
        """1着→2着→3着の完全一致で的中。"""
        assert HitJudge.is_hit(BetType.TRIFECTA, (3, 1, 5), BASIC_RESULTS) is True

    def test_miss_with_wrong_first(self):
        """1着が違えば不的中。"""
        assert HitJudge.is_hit(BetType.TRIFECTA, (1, 3, 5), BASIC_RESULTS) is False

    def test_miss_with_wrong_second(self):
        """2着が違えば不的中。"""
        assert HitJudge.is_hit(BetType.TRIFECTA, (3, 5, 1), BASIC_RESULTS) is False

    def test_miss_with_wrong_third(self):
        """3着が違えば不的中。"""
        assert HitJudge.is_hit(BetType.TRIFECTA, (3, 1, 2), BASIC_RESULTS) is False

    def test_miss_with_reversed_order(self):
        """逆順は不的中。"""
        assert HitJudge.is_hit(BetType.TRIFECTA, (5, 1, 3), BASIC_RESULTS) is False


class TestEdgeCases:
    """エッジケースのテスト。"""

    def test_empty_results_returns_false(self):
        """結果が空の場合はFalse。"""
        assert HitJudge.is_hit(BetType.WIN, (1,), []) is False

    def test_empty_combination_returns_false(self):
        """組み合わせが空の場合はFalse。"""
        assert HitJudge.is_hit(BetType.WIN, (), BASIC_RESULTS) is False

    def test_results_not_in_order(self):
        """レース結果がソートされていなくても正しく判定。"""
        unordered = _make_results([(4, 5), (1, 2), (3, 1), (2, 4), (5, 3)])
        assert HitJudge.is_hit(BetType.WIN, (3,), unordered) is True
        assert HitJudge.is_hit(BetType.PLACE, (5,), unordered) is True
        assert HitJudge.is_hit(BetType.QUINELLA, (3, 1), unordered) is True

    def test_place_with_fewer_than_3_runners(self):
        """出走馬が3頭未満でも正しく判定。"""
        results = _make_results([(1, 1), (2, 2)])
        # 1着馬は的中
        assert HitJudge.is_hit(BetType.PLACE, (1,), results) is True
        # 2着馬は的中
        assert HitJudge.is_hit(BetType.PLACE, (2,), results) is True
        # 出走していない馬は不的中
        assert HitJudge.is_hit(BetType.PLACE, (3,), results) is False


class TestGetTopN:
    """_get_top_n のテスト。"""

    def test_returns_correct_top_1(self):
        """1着馬を正しく返す。"""
        top1 = HitJudge._get_top_n(BASIC_RESULTS, 1)
        assert top1 == [3]

    def test_returns_correct_top_2(self):
        """1-2着馬を正しく返す。"""
        top2 = HitJudge._get_top_n(BASIC_RESULTS, 2)
        assert top2 == [3, 1]

    def test_returns_correct_top_3(self):
        """1-2-3着馬を正しく返す。"""
        top3 = HitJudge._get_top_n(BASIC_RESULTS, 3)
        assert top3 == [3, 1, 5]

    def test_handles_unordered_results(self):
        """結果がソートされていなくても正しく返す。"""
        unordered = _make_results([(4, 5), (1, 2), (3, 1), (2, 4), (5, 3)])
        top3 = HitJudge._get_top_n(unordered, 3)
        assert top3 == [3, 1, 5]
