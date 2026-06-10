"""的中判定ロジックのプロパティベーステスト。

Feature: horse-race-predictor, Property 14: 的中判定の正確性

Validates: Requirements 9.1
"""

from hypothesis import given, settings
from hypothesis import strategies as st

from src.backtest.hit_judge import HitJudge
from src.data.models import BetType, RaceResult


# --- Strategies ---


@st.composite
def race_results_strategy(draw: st.DrawFn, min_horses: int = 5, max_horses: int = 18) -> list[RaceResult]:
    """ユニークな着順を持つレース結果を生成するストラテジ。

    各馬に一意な馬番と着順を割り当てる。
    """
    n_horses = draw(st.integers(min_value=min_horses, max_value=max_horses))
    horse_numbers = list(range(1, n_horses + 1))
    # シャッフルして着順を割り当てる
    finish_order = draw(st.permutations(horse_numbers))
    results = [
        RaceResult(horse_number=horse_numbers[i], finish_position=finish_order.index(horse_numbers[i]) + 1)
        for i in range(n_horses)
    ]
    return results


def get_horse_at_position(results: list[RaceResult], position: int) -> int:
    """指定着順の馬番を返すヘルパー関数。"""
    for r in results:
        if r.finish_position == position:
            return r.horse_number
    raise ValueError(f"Position {position} not found in results")


def get_horses_not_in_top_n(results: list[RaceResult], n: int) -> list[int]:
    """上位n頭に含まれない馬番リストを返すヘルパー関数。"""
    top_n = {r.horse_number for r in results if r.finish_position <= n}
    return [r.horse_number for r in results if r.horse_number not in top_n]


# --- Property 14 Tests ---


class TestHitJudgmentProperties:
    """Property 14: 的中判定の正確性

    Feature: horse-race-predictor, Property 14: 的中判定の正確性

    For any 買い目（券種と馬番組み合わせ）と実際のレース結果（着順）に対して、
    的中判定は券種ごとのルール（単勝なら1着一致、馬連なら1-2着の組み合わせ一致など）
    に基づいて正しく判定されること。

    Validates: Requirements 9.1
    """

    # --- 単勝 (WIN) ---

    @settings(max_examples=100)
    @given(results=race_results_strategy())
    def test_win_hit_when_horse_is_first(self, results: list[RaceResult]) -> None:
        """単勝: 1着の馬番を指定した場合、的中と判定されること。

        Validates: Requirements 9.1
        """
        first_place = get_horse_at_position(results, 1)
        combination = (first_place,)

        assert HitJudge.is_hit(BetType.WIN, combination, results) is True, (
            f"単勝で1着馬を指定したのに不的中: "
            f"combination={combination}, 1着={first_place}"
        )

    @settings(max_examples=100)
    @given(results=race_results_strategy(), data=st.data())
    def test_win_miss_when_horse_is_not_first(self, results: list[RaceResult], data: st.DataObject) -> None:
        """単勝: 1着でない馬番を指定した場合、不的中と判定されること。

        Validates: Requirements 9.1
        """
        non_first = get_horses_not_in_top_n(results, 1)
        horse = data.draw(st.sampled_from(non_first))
        combination = (horse,)

        assert HitJudge.is_hit(BetType.WIN, combination, results) is False, (
            f"単勝で1着でない馬を指定したのに的中: "
            f"combination={combination}, horse_position != 1"
        )

    # --- 複勝 (PLACE) ---

    @settings(max_examples=100)
    @given(results=race_results_strategy(), data=st.data())
    def test_place_hit_when_horse_in_top3(self, results: list[RaceResult], data: st.DataObject) -> None:
        """複勝: 3着以内の馬番を指定した場合、的中と判定されること。

        Validates: Requirements 9.1
        """
        top3_horses = [r.horse_number for r in results if r.finish_position <= 3]
        horse = data.draw(st.sampled_from(top3_horses))
        combination = (horse,)

        assert HitJudge.is_hit(BetType.PLACE, combination, results) is True, (
            f"複勝で3着以内の馬を指定したのに不的中: "
            f"combination={combination}, top3={top3_horses}"
        )

    @settings(max_examples=100)
    @given(results=race_results_strategy(), data=st.data())
    def test_place_miss_when_horse_not_in_top3(self, results: list[RaceResult], data: st.DataObject) -> None:
        """複勝: 3着以内でない馬番を指定した場合、不的中と判定されること。

        Validates: Requirements 9.1
        """
        non_top3 = get_horses_not_in_top_n(results, 3)
        horse = data.draw(st.sampled_from(non_top3))
        combination = (horse,)

        assert HitJudge.is_hit(BetType.PLACE, combination, results) is False, (
            f"複勝で3着以内でない馬を指定したのに的中: "
            f"combination={combination}"
        )

    # --- 馬連 (QUINELLA) ---

    @settings(max_examples=100)
    @given(results=race_results_strategy())
    def test_quinella_hit_when_top2_set_matches(self, results: list[RaceResult]) -> None:
        """馬連: 1-2着の馬番の組み合わせ（順不同）を指定した場合、的中と判定されること。

        Validates: Requirements 9.1
        """
        first = get_horse_at_position(results, 1)
        second = get_horse_at_position(results, 2)
        # 順序を入れ替えてもOK
        combination = (second, first)

        assert HitJudge.is_hit(BetType.QUINELLA, combination, results) is True, (
            f"馬連で1-2着の組み合わせを指定したのに不的中: "
            f"combination={combination}, top2=({first}, {second})"
        )

    @settings(max_examples=100)
    @given(results=race_results_strategy(), data=st.data())
    def test_quinella_miss_when_not_top2_pair(self, results: list[RaceResult], data: st.DataObject) -> None:
        """馬連: 1-2着の組み合わせでない馬番ペアを指定した場合、不的中と判定されること。

        Validates: Requirements 9.1
        """
        first = get_horse_at_position(results, 1)
        second = get_horse_at_position(results, 2)
        non_top2 = get_horses_not_in_top_n(results, 2)
        # 1頭はtop2外の馬を使って不的中を確実にする
        wrong_horse = data.draw(st.sampled_from(non_top2))
        combination = (first, wrong_horse)

        assert HitJudge.is_hit(BetType.QUINELLA, combination, results) is False, (
            f"馬連で1-2着でない組み合わせを指定したのに的中: "
            f"combination={combination}, top2=({first}, {second})"
        )

    # --- 馬単 (EXACTA) ---

    @settings(max_examples=100)
    @given(results=race_results_strategy())
    def test_exacta_hit_when_top2_order_matches(self, results: list[RaceResult]) -> None:
        """馬単: 1着→2着の順序で馬番を指定した場合、的中と判定されること。

        Validates: Requirements 9.1
        """
        first = get_horse_at_position(results, 1)
        second = get_horse_at_position(results, 2)
        combination = (first, second)

        assert HitJudge.is_hit(BetType.EXACTA, combination, results) is True, (
            f"馬単で1着→2着の順序を指定したのに不的中: "
            f"combination={combination}"
        )

    @settings(max_examples=100)
    @given(results=race_results_strategy())
    def test_exacta_miss_when_top2_order_reversed(self, results: list[RaceResult]) -> None:
        """馬単: 2着→1着の逆順で馬番を指定した場合、不的中と判定されること。

        Validates: Requirements 9.1
        """
        first = get_horse_at_position(results, 1)
        second = get_horse_at_position(results, 2)
        combination = (second, first)

        assert HitJudge.is_hit(BetType.EXACTA, combination, results) is False, (
            f"馬単で逆順を指定したのに的中: "
            f"combination={combination}, 正解=(({first}, {second}))"
        )

    # --- ワイド (WIDE) ---

    @settings(max_examples=100)
    @given(results=race_results_strategy(), data=st.data())
    def test_wide_hit_when_both_horses_in_top3(self, results: list[RaceResult], data: st.DataObject) -> None:
        """ワイド: 3着以内の2頭の組み合わせを指定した場合、的中と判定されること。

        Validates: Requirements 9.1
        """
        top3_horses = [r.horse_number for r in results if r.finish_position <= 3]
        # 3着以内の2頭を選ぶ（重複なし）
        pair = data.draw(
            st.lists(st.sampled_from(top3_horses), min_size=2, max_size=2, unique=True)
        )
        combination = tuple(pair)

        assert HitJudge.is_hit(BetType.WIDE, combination, results) is True, (
            f"ワイドで3着以内の2頭を指定したのに不的中: "
            f"combination={combination}, top3={top3_horses}"
        )

    @settings(max_examples=100)
    @given(results=race_results_strategy(), data=st.data())
    def test_wide_miss_when_horse_not_in_top3(self, results: list[RaceResult], data: st.DataObject) -> None:
        """ワイド: 3着以内でない馬を含む組み合わせを指定した場合、不的中と判定されること。

        Validates: Requirements 9.1
        """
        first = get_horse_at_position(results, 1)
        non_top3 = get_horses_not_in_top_n(results, 3)
        wrong_horse = data.draw(st.sampled_from(non_top3))
        combination = (first, wrong_horse)

        assert HitJudge.is_hit(BetType.WIDE, combination, results) is False, (
            f"ワイドで3着以内でない馬を含む組み合わせを指定したのに的中: "
            f"combination={combination}"
        )

    # --- 三連複 (TRIO) ---

    @settings(max_examples=100)
    @given(results=race_results_strategy())
    def test_trio_hit_when_top3_set_matches(self, results: list[RaceResult]) -> None:
        """三連複: 1-3着の馬番の組み合わせ（順不同）を指定した場合、的中と判定されること。

        Validates: Requirements 9.1
        """
        first = get_horse_at_position(results, 1)
        second = get_horse_at_position(results, 2)
        third = get_horse_at_position(results, 3)
        # 順番をわざと変えて順不同を確認
        combination = (third, first, second)

        assert HitJudge.is_hit(BetType.TRIO, combination, results) is True, (
            f"三連複で1-3着の組み合わせを指定したのに不的中: "
            f"combination={combination}, top3=({first}, {second}, {third})"
        )

    @settings(max_examples=100)
    @given(results=race_results_strategy(), data=st.data())
    def test_trio_miss_when_not_top3_set(self, results: list[RaceResult], data: st.DataObject) -> None:
        """三連複: 1-3着の組み合わせでない馬番トリオを指定した場合、不的中と判定されること。

        Validates: Requirements 9.1
        """
        first = get_horse_at_position(results, 1)
        second = get_horse_at_position(results, 2)
        non_top3 = get_horses_not_in_top_n(results, 3)
        wrong_horse = data.draw(st.sampled_from(non_top3))
        combination = (first, second, wrong_horse)

        assert HitJudge.is_hit(BetType.TRIO, combination, results) is False, (
            f"三連複で1-3着でない組み合わせを指定したのに的中: "
            f"combination={combination}"
        )

    # --- 三連単 (TRIFECTA) ---

    @settings(max_examples=100)
    @given(results=race_results_strategy())
    def test_trifecta_hit_when_top3_order_matches(self, results: list[RaceResult]) -> None:
        """三連単: 1着→2着→3着の順序で馬番を指定した場合、的中と判定されること。

        Validates: Requirements 9.1
        """
        first = get_horse_at_position(results, 1)
        second = get_horse_at_position(results, 2)
        third = get_horse_at_position(results, 3)
        combination = (first, second, third)

        assert HitJudge.is_hit(BetType.TRIFECTA, combination, results) is True, (
            f"三連単で1-2-3着の順序を指定したのに不的中: "
            f"combination={combination}"
        )

    @settings(max_examples=100)
    @given(results=race_results_strategy())
    def test_trifecta_miss_when_top3_order_wrong(self, results: list[RaceResult]) -> None:
        """三連単: 1-3着の馬番でも順序が異なる場合、不的中と判定されること。

        Validates: Requirements 9.1
        """
        first = get_horse_at_position(results, 1)
        second = get_horse_at_position(results, 2)
        third = get_horse_at_position(results, 3)
        # 順序を入れ替え: 2着→1着→3着
        combination = (second, first, third)

        assert HitJudge.is_hit(BetType.TRIFECTA, combination, results) is False, (
            f"三連単で順序が異なる組み合わせを指定したのに的中: "
            f"combination={combination}, 正解=({first}, {second}, {third})"
        )
