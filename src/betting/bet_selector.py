"""買い目選択モジュール。

期待値に基づいて最適な買い目を選択・絞り込むモジュール。
複数券種（単勝、複勝、馬連、馬単、ワイド、三連複、三連単）から
最も期待値の高い組み合わせを選択する。
"""

from itertools import combinations, permutations

import numpy as np

from src.data.models import BetRecommendation, BetType, RaceData


class BetSelector:
    """期待値に基づく買い目選択。

    各買い目の期待値（推定的中確率 × 推定オッズ）を計算し、
    閾値を超える買い目を期待値の降順で最大max_bets件選択する。
    """

    def __init__(
        self,
        min_expected_value: float = 1.0,
        target_bet_types: list[BetType] | None = None,
    ) -> None:
        """Initialize with minimum expected value threshold and target bet types.

        Args:
            min_expected_value: 期待値の最低基準。これを超える買い目のみ推奨する。
            target_bet_types: 対象券種リスト。Noneの場合は全券種を対象とする。
        """
        self.min_expected_value = min_expected_value
        self.target_bet_types = target_bet_types if target_bet_types is not None else list(BetType)

    def select_bets(
        self, race: RaceData, probabilities: np.ndarray, max_bets: int = 10
    ) -> list[BetRecommendation]:
        """期待値が閾値を超える買い目を最大max_bets件選択する。

        Args:
            race: レースデータ（出走馬情報とオッズを含む）。
            probabilities: 各馬の1着確率の配列（インデックスは出走馬リスト順）。
            max_bets: 選択する最大買い目数。

        Returns:
            期待値の降順でソートされた買い目リスト。
            期待値が閾値を超える買い目がない場合は空リストを返す（見送り対象）。
        """
        candidates: list[BetRecommendation] = []

        for bet_type in self.target_bet_types:
            candidates.extend(
                self._generate_candidates(race, probabilities, bet_type)
            )

        # 期待値が閾値を超える買い目のみフィルタ
        filtered = [c for c in candidates if c.expected_value > self.min_expected_value]

        # 期待値の降順でソート
        filtered.sort(key=lambda x: x.expected_value, reverse=True)

        # 最大max_bets件に制限
        return filtered[:max_bets]

    def calculate_expected_value(self, probability: float, odds: float) -> float:
        """期待値を計算する。

        期待値 = probability × odds

        Args:
            probability: 推定的中確率 (0 < p < 1)。
            odds: 推定オッズ (odds > 0)。

        Returns:
            期待値。
        """
        return probability * odds

    def _generate_candidates(
        self, race: RaceData, probabilities: np.ndarray, bet_type: BetType
    ) -> list[BetRecommendation]:
        """指定券種の買い目候補を生成する。

        Args:
            race: レースデータ。
            probabilities: 各馬の1着確率の配列。
            bet_type: 対象券種。

        Returns:
            買い目候補リスト。
        """
        if bet_type == BetType.WIN:
            return self._generate_win_candidates(race, probabilities)
        elif bet_type == BetType.PLACE:
            return self._generate_place_candidates(race, probabilities)
        elif bet_type == BetType.QUINELLA:
            return self._generate_quinella_candidates(race, probabilities)
        elif bet_type == BetType.EXACTA:
            return self._generate_exacta_candidates(race, probabilities)
        elif bet_type == BetType.WIDE:
            return self._generate_wide_candidates(race, probabilities)
        elif bet_type == BetType.TRIO:
            return self._generate_trio_candidates(race, probabilities)
        elif bet_type == BetType.TRIFECTA:
            return self._generate_trifecta_candidates(race, probabilities)
        return []

    def _generate_win_candidates(
        self, race: RaceData, probabilities: np.ndarray
    ) -> list[BetRecommendation]:
        """単勝の買い目候補を生成する。

        各馬の1着確率をそのまま使用し、単勝オッズと組み合わせて期待値を算出する。
        """
        candidates: list[BetRecommendation] = []

        for i, entry in enumerate(race.entries):
            if i >= len(probabilities):
                break
            if entry.win_odds is None or entry.win_odds <= 0:
                continue

            prob = float(probabilities[i])
            odds = entry.win_odds
            ev = self.calculate_expected_value(prob, odds)

            candidates.append(
                BetRecommendation(
                    bet_type=BetType.WIN,
                    combination=(entry.horse_number,),
                    estimated_probability=prob,
                    estimated_odds=odds,
                    expected_value=ev,
                )
            )

        return candidates

    def _generate_place_candidates(
        self, race: RaceData, probabilities: np.ndarray
    ) -> list[BetRecommendation]:
        """複勝の買い目候補を生成する。

        複勝確率は1着確率を基に3着以内に入る確率を概算する。
        簡易推定: prob_place = min(prob * 3, 1.0)
        オッズ推定: win_odds / 3
        """
        candidates: list[BetRecommendation] = []

        for i, entry in enumerate(race.entries):
            if i >= len(probabilities):
                break
            if entry.win_odds is None or entry.win_odds <= 0:
                continue

            prob_place = min(float(probabilities[i]) * 3, 1.0)
            odds_place = entry.win_odds / 3.0
            ev = self.calculate_expected_value(prob_place, odds_place)

            candidates.append(
                BetRecommendation(
                    bet_type=BetType.PLACE,
                    combination=(entry.horse_number,),
                    estimated_probability=prob_place,
                    estimated_odds=odds_place,
                    expected_value=ev,
                )
            )

        return candidates

    def _generate_quinella_candidates(
        self, race: RaceData, probabilities: np.ndarray
    ) -> list[BetRecommendation]:
        """馬連の買い目候補を生成する。

        馬連確率: prob_i * prob_j * 2（順不同のため2倍）
        オッズ推定: win_odds_i * win_odds_j の平方根を簡易推定とする
        上位馬のペアに限定して候補を生成する。
        """
        candidates: list[BetRecommendation] = []
        n_entries = min(len(race.entries), len(probabilities))

        # 確率上位の馬に絞って組み合わせを生成（計算量抑制）
        top_indices = np.argsort(probabilities[:n_entries])[::-1][:8]

        for i, j in combinations(top_indices, 2):
            entry_i = race.entries[i]
            entry_j = race.entries[j]

            if entry_i.win_odds is None or entry_i.win_odds <= 0:
                continue
            if entry_j.win_odds is None or entry_j.win_odds <= 0:
                continue

            prob = float(probabilities[i]) * float(probabilities[j]) * 2
            odds = (entry_i.win_odds * entry_j.win_odds) ** 0.5
            ev = self.calculate_expected_value(prob, odds)

            horse_numbers = tuple(sorted([entry_i.horse_number, entry_j.horse_number]))
            candidates.append(
                BetRecommendation(
                    bet_type=BetType.QUINELLA,
                    combination=horse_numbers,
                    estimated_probability=prob,
                    estimated_odds=odds,
                    expected_value=ev,
                )
            )

        return candidates

    def _generate_exacta_candidates(
        self, race: RaceData, probabilities: np.ndarray
    ) -> list[BetRecommendation]:
        """馬単の買い目候補を生成する。

        馬単確率: prob_i * prob_j（順序あり、iが1着, jが2着）
        オッズ推定: win_odds_i * win_odds_j の平方根 * 1.5（順序による割増）
        上位馬のペアに限定して候補を生成する。
        """
        candidates: list[BetRecommendation] = []
        n_entries = min(len(race.entries), len(probabilities))

        top_indices = np.argsort(probabilities[:n_entries])[::-1][:6]

        for i, j in permutations(top_indices, 2):
            entry_i = race.entries[i]
            entry_j = race.entries[j]

            if entry_i.win_odds is None or entry_i.win_odds <= 0:
                continue
            if entry_j.win_odds is None or entry_j.win_odds <= 0:
                continue

            prob = float(probabilities[i]) * float(probabilities[j])
            odds = (entry_i.win_odds * entry_j.win_odds) ** 0.5 * 1.5
            ev = self.calculate_expected_value(prob, odds)

            candidates.append(
                BetRecommendation(
                    bet_type=BetType.EXACTA,
                    combination=(entry_i.horse_number, entry_j.horse_number),
                    estimated_probability=prob,
                    estimated_odds=odds,
                    expected_value=ev,
                )
            )

        return candidates

    def _generate_wide_candidates(
        self, race: RaceData, probabilities: np.ndarray
    ) -> list[BetRecommendation]:
        """ワイドの買い目候補を生成する。

        ワイド確率: 2馬がともに3着以内に入る確率の概算
        簡易推定: prob_i_place * prob_j_place * 0.5
        オッズ推定: (win_odds_i * win_odds_j) ** 0.5 / 3
        """
        candidates: list[BetRecommendation] = []
        n_entries = min(len(race.entries), len(probabilities))

        top_indices = np.argsort(probabilities[:n_entries])[::-1][:8]

        for i, j in combinations(top_indices, 2):
            entry_i = race.entries[i]
            entry_j = race.entries[j]

            if entry_i.win_odds is None or entry_i.win_odds <= 0:
                continue
            if entry_j.win_odds is None or entry_j.win_odds <= 0:
                continue

            prob_i_place = min(float(probabilities[i]) * 3, 1.0)
            prob_j_place = min(float(probabilities[j]) * 3, 1.0)
            prob = prob_i_place * prob_j_place * 0.5
            odds = (entry_i.win_odds * entry_j.win_odds) ** 0.5 / 3.0
            ev = self.calculate_expected_value(prob, odds)

            horse_numbers = tuple(sorted([entry_i.horse_number, entry_j.horse_number]))
            candidates.append(
                BetRecommendation(
                    bet_type=BetType.WIDE,
                    combination=horse_numbers,
                    estimated_probability=prob,
                    estimated_odds=odds,
                    expected_value=ev,
                )
            )

        return candidates

    def _generate_trio_candidates(
        self, race: RaceData, probabilities: np.ndarray
    ) -> list[BetRecommendation]:
        """三連複の買い目候補を生成する。

        三連複確率: 3馬がともに3着以内に入る確率の概算
        簡易推定: prob_i * prob_j * prob_k * 6（順不同のため6通り）
        オッズ推定: (win_odds_i * win_odds_j * win_odds_k) ** (1/3) * 2
        """
        candidates: list[BetRecommendation] = []
        n_entries = min(len(race.entries), len(probabilities))

        top_indices = np.argsort(probabilities[:n_entries])[::-1][:6]

        for i, j, k in combinations(top_indices, 3):
            entry_i = race.entries[i]
            entry_j = race.entries[j]
            entry_k = race.entries[k]

            if entry_i.win_odds is None or entry_i.win_odds <= 0:
                continue
            if entry_j.win_odds is None or entry_j.win_odds <= 0:
                continue
            if entry_k.win_odds is None or entry_k.win_odds <= 0:
                continue

            prob = float(probabilities[i]) * float(probabilities[j]) * float(probabilities[k]) * 6
            odds = (entry_i.win_odds * entry_j.win_odds * entry_k.win_odds) ** (1.0 / 3.0) * 2
            ev = self.calculate_expected_value(prob, odds)

            horse_numbers = tuple(sorted([
                entry_i.horse_number, entry_j.horse_number, entry_k.horse_number
            ]))
            candidates.append(
                BetRecommendation(
                    bet_type=BetType.TRIO,
                    combination=horse_numbers,
                    estimated_probability=prob,
                    estimated_odds=odds,
                    expected_value=ev,
                )
            )

        return candidates

    def _generate_trifecta_candidates(
        self, race: RaceData, probabilities: np.ndarray
    ) -> list[BetRecommendation]:
        """三連単の買い目候補を生成する。

        三連単確率: prob_i * prob_j * prob_k（順序あり）
        オッズ推定: (win_odds_i * win_odds_j * win_odds_k) ** (1/3) * 6
        上位馬の順列に限定して候補を生成する。
        """
        candidates: list[BetRecommendation] = []
        n_entries = min(len(race.entries), len(probabilities))

        top_indices = np.argsort(probabilities[:n_entries])[::-1][:5]

        for i, j, k in permutations(top_indices, 3):
            entry_i = race.entries[i]
            entry_j = race.entries[j]
            entry_k = race.entries[k]

            if entry_i.win_odds is None or entry_i.win_odds <= 0:
                continue
            if entry_j.win_odds is None or entry_j.win_odds <= 0:
                continue
            if entry_k.win_odds is None or entry_k.win_odds <= 0:
                continue

            prob = float(probabilities[i]) * float(probabilities[j]) * float(probabilities[k])
            odds = (entry_i.win_odds * entry_j.win_odds * entry_k.win_odds) ** (1.0 / 3.0) * 6
            ev = self.calculate_expected_value(prob, odds)

            candidates.append(
                BetRecommendation(
                    bet_type=BetType.TRIFECTA,
                    combination=(entry_i.horse_number, entry_j.horse_number, entry_k.horse_number),
                    estimated_probability=prob,
                    estimated_odds=odds,
                    expected_value=ev,
                )
            )

        return candidates
