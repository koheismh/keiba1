"""BetSelector のユニットテスト。"""

from datetime import date

import numpy as np
import pytest

from src.betting.bet_selector import BetSelector
from src.data.models import BetRecommendation, BetType, HorseEntry, RaceData, TrackCondition


def _make_entry(horse_number: int, win_odds: float) -> HorseEntry:
    """テスト用の出走馬エントリを作成する。"""
    return HorseEntry(
        horse_name=f"Horse{horse_number}",
        jockey_name=f"Jockey{horse_number}",
        gate_number=min(horse_number, 8),
        horse_number=horse_number,
        weight=480,
        weight_change=0,
        win_odds=win_odds,
    )


def _make_race(entries: list[HorseEntry]) -> RaceData:
    """テスト用のレースデータを作成する。"""
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


class TestCalculateExpectedValue:
    """calculate_expected_value のテスト。"""

    def test_basic_calculation(self):
        selector = BetSelector()
        assert selector.calculate_expected_value(0.5, 3.0) == 1.5

    def test_probability_one_returns_odds(self):
        selector = BetSelector()
        assert selector.calculate_expected_value(1.0, 5.0) == 5.0

    def test_zero_probability_returns_zero(self):
        selector = BetSelector()
        assert selector.calculate_expected_value(0.0, 10.0) == 0.0

    def test_low_expected_value(self):
        selector = BetSelector()
        assert selector.calculate_expected_value(0.1, 5.0) == pytest.approx(0.5)

    def test_high_odds_low_probability(self):
        selector = BetSelector()
        assert selector.calculate_expected_value(0.01, 200.0) == pytest.approx(2.0)


class TestSelectBets:
    """select_bets のテスト。"""

    def setup_method(self):
        """テスト共通のセットアップ。"""
        self.entries = [
            _make_entry(1, 2.0),   # 人気馬
            _make_entry(2, 5.0),
            _make_entry(3, 10.0),
            _make_entry(4, 20.0),
            _make_entry(5, 50.0),
        ]
        self.race = _make_race(self.entries)

    def test_returns_empty_when_no_bets_exceed_threshold(self):
        """期待値が閾値を超える買い目がない場合、空リストを返す。"""
        selector = BetSelector(min_expected_value=1.0, target_bet_types=[BetType.WIN])
        # 非常に低い確率→期待値は1.0以下
        probabilities = np.array([0.01, 0.01, 0.01, 0.01, 0.01])
        result = selector.select_bets(self.race, probabilities)
        assert result == []

    def test_returns_bets_sorted_by_expected_value_descending(self):
        """買い目が期待値の降順でソートされる。"""
        selector = BetSelector(min_expected_value=1.0, target_bet_types=[BetType.WIN])
        # 馬3: 0.3 * 10 = 3.0, 馬2: 0.4 * 5 = 2.0, 馬1: 0.6 * 2 = 1.2
        probabilities = np.array([0.6, 0.4, 0.3, 0.01, 0.01])
        result = selector.select_bets(self.race, probabilities)

        assert len(result) >= 2
        # 期待値降順であることを確認
        for i in range(len(result) - 1):
            assert result[i].expected_value >= result[i + 1].expected_value

    def test_respects_max_bets_limit(self):
        """max_bets で件数制限される。"""
        selector = BetSelector(min_expected_value=0.5, target_bet_types=[BetType.WIN])
        probabilities = np.array([0.5, 0.3, 0.2, 0.15, 0.1])
        result = selector.select_bets(self.race, probabilities, max_bets=2)
        assert len(result) <= 2

    def test_all_bets_exceed_min_expected_value(self):
        """すべての買い目の期待値が閾値を超える。"""
        selector = BetSelector(min_expected_value=1.5, target_bet_types=[BetType.WIN])
        probabilities = np.array([0.5, 0.4, 0.3, 0.2, 0.1])
        result = selector.select_bets(self.race, probabilities)

        for bet in result:
            assert bet.expected_value > 1.5

    def test_bet_recommendation_contains_required_fields(self):
        """各買い目に必須フィールドが含まれる。"""
        selector = BetSelector(min_expected_value=0.5, target_bet_types=[BetType.WIN])
        probabilities = np.array([0.5, 0.3, 0.2, 0.15, 0.1])
        result = selector.select_bets(self.race, probabilities)

        for bet in result:
            assert isinstance(bet.bet_type, BetType)
            assert isinstance(bet.combination, tuple)
            assert len(bet.combination) > 0
            assert bet.estimated_probability > 0
            assert bet.estimated_odds > 0
            assert bet.expected_value > 0

    def test_win_bet_combination_is_single_horse_number(self):
        """単勝の買い目の組み合わせは馬番1つ。"""
        selector = BetSelector(min_expected_value=0.5, target_bet_types=[BetType.WIN])
        probabilities = np.array([0.5, 0.3, 0.2, 0.15, 0.1])
        result = selector.select_bets(self.race, probabilities)

        for bet in result:
            assert bet.bet_type == BetType.WIN
            assert len(bet.combination) == 1

    def test_place_bets_generated_correctly(self):
        """複勝の買い目が正しく生成される。"""
        selector = BetSelector(min_expected_value=0.5, target_bet_types=[BetType.PLACE])
        probabilities = np.array([0.4, 0.3, 0.2, 0.05, 0.05])
        result = selector.select_bets(self.race, probabilities)

        for bet in result:
            assert bet.bet_type == BetType.PLACE
            assert len(bet.combination) == 1

    def test_quinella_bets_contain_two_horse_numbers(self):
        """馬連の買い目には2頭の馬番が含まれる。"""
        selector = BetSelector(min_expected_value=0.0, target_bet_types=[BetType.QUINELLA])
        probabilities = np.array([0.4, 0.3, 0.2, 0.05, 0.05])
        result = selector.select_bets(self.race, probabilities)

        for bet in result:
            assert bet.bet_type == BetType.QUINELLA
            assert len(bet.combination) == 2
            # 馬連は順不同→ソート済み
            assert bet.combination[0] < bet.combination[1]

    def test_multiple_bet_types(self):
        """複数券種から買い目を選択する。"""
        selector = BetSelector(
            min_expected_value=0.5,
            target_bet_types=[BetType.WIN, BetType.PLACE, BetType.QUINELLA],
        )
        probabilities = np.array([0.4, 0.3, 0.2, 0.05, 0.05])
        result = selector.select_bets(self.race, probabilities)

        bet_types_in_result = {bet.bet_type for bet in result}
        # 少なくとも1つの券種が含まれる
        assert len(bet_types_in_result) >= 1

    def test_skips_horses_with_no_odds(self):
        """オッズがNoneの馬はスキップされる。"""
        entries = [
            _make_entry(1, 2.0),
            HorseEntry(
                horse_name="NoOdds",
                jockey_name="Jockey",
                gate_number=2,
                horse_number=2,
                weight=480,
                weight_change=0,
                win_odds=None,
            ),
            _make_entry(3, 10.0),
        ]
        race = _make_race(entries)
        selector = BetSelector(min_expected_value=0.5, target_bet_types=[BetType.WIN])
        probabilities = np.array([0.5, 0.3, 0.2])
        result = selector.select_bets(race, probabilities)

        # 馬番2は含まれない
        for bet in result:
            assert 2 not in bet.combination

    def test_default_target_bet_types_includes_all(self):
        """target_bet_types未指定時は全券種が対象。"""
        selector = BetSelector(min_expected_value=1.0)
        assert set(selector.target_bet_types) == set(BetType)

    def test_empty_entries_returns_empty(self):
        """出走馬がいない場合、空リストを返す。"""
        race = _make_race([])
        selector = BetSelector(min_expected_value=0.5, target_bet_types=[BetType.WIN])
        probabilities = np.array([])
        result = selector.select_bets(race, probabilities)
        assert result == []


class TestExactaAndTrifectaCandidates:
    """馬単・三連単の買い目候補テスト。"""

    def setup_method(self):
        self.entries = [
            _make_entry(1, 3.0),
            _make_entry(2, 5.0),
            _make_entry(3, 8.0),
            _make_entry(4, 15.0),
            _make_entry(5, 30.0),
        ]
        self.race = _make_race(self.entries)

    def test_exacta_combination_is_ordered_pair(self):
        """馬単の買い目は順序付きペア。"""
        selector = BetSelector(min_expected_value=0.0, target_bet_types=[BetType.EXACTA])
        probabilities = np.array([0.35, 0.25, 0.2, 0.12, 0.08])
        result = selector.select_bets(self.race, probabilities)

        for bet in result:
            assert bet.bet_type == BetType.EXACTA
            assert len(bet.combination) == 2

    def test_trifecta_combination_has_three_horses(self):
        """三連単の買い目は3頭の順序付き組み合わせ。"""
        selector = BetSelector(min_expected_value=0.0, target_bet_types=[BetType.TRIFECTA])
        probabilities = np.array([0.35, 0.25, 0.2, 0.12, 0.08])
        result = selector.select_bets(self.race, probabilities)

        for bet in result:
            assert bet.bet_type == BetType.TRIFECTA
            assert len(bet.combination) == 3

    def test_trio_combination_is_sorted(self):
        """三連複の買い目はソート済み。"""
        selector = BetSelector(min_expected_value=0.0, target_bet_types=[BetType.TRIO])
        probabilities = np.array([0.35, 0.25, 0.2, 0.12, 0.08])
        result = selector.select_bets(self.race, probabilities)

        for bet in result:
            assert bet.bet_type == BetType.TRIO
            assert len(bet.combination) == 3
            assert bet.combination == tuple(sorted(bet.combination))
