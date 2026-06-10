"""特徴量エンジニアリング"""

from __future__ import annotations

import numpy as np

from src.data.models import FeatureVector, HorseEntry, RaceData, TrackCondition


FEATURE_NAMES = [
    "past_performance",
    "jockey_performance",
    "course_aptitude",
    "distance_aptitude",
    "track_condition_aptitude",
    "gate_number",
    "weight_change",
    "class_performance",
]


class FeatureEngineer:
    """特徴量エンジニアリングを行う"""

    def __init__(self, historical_races: list[RaceData] | None = None) -> None:
        """過去レースデータを保持する。

        Args:
            historical_races: 過去レースデータのリスト。Noneの場合は空リストとして扱う。
        """
        self._historical_races: list[RaceData] = historical_races or []

    def extract_features(self, race: RaceData, horse: HorseEntry) -> FeatureVector:
        """個別の馬に対する特徴量を抽出する。

        Args:
            race: 対象レースデータ
            horse: 対象出走馬情報

        Returns:
            特徴量ベクトル
        """
        values = np.array([
            self._calc_past_performance(horse.horse_name),
            self._calc_jockey_performance(horse.jockey_name),
            self._calc_course_aptitude(horse.horse_name, race.venue),
            self._calc_distance_aptitude(horse.horse_name, race.distance),
            self._calc_track_condition_aptitude(horse.horse_name, race.track_condition),
            self._calc_gate_number(horse.gate_number),
            self._calc_weight_change(horse.weight_change),
            self._calc_class_performance(horse.horse_name),
        ], dtype=np.float64)

        return FeatureVector(values=values, feature_names=FEATURE_NAMES)

    def get_feature_names(self) -> list[str]:
        """使用する特徴量名の一覧を返す。

        Returns:
            特徴量名のリスト
        """
        return list(FEATURE_NAMES)

    # --- Private helper methods ---

    def _find_horse_results(self, horse_name: str) -> list[tuple[RaceData, int]]:
        """過去レースから指定馬の出走結果を取得する。

        Returns:
            (レースデータ, 着順) のリスト
        """
        results: list[tuple[RaceData, int]] = []
        for race in self._historical_races:
            if race.results is None:
                continue
            for entry in race.entries:
                if entry.horse_name == horse_name:
                    # 対応する結果を探す
                    for result in race.results:
                        if result.horse_number == entry.horse_number:
                            results.append((race, result.finish_position))
                            break
                    break
        return results

    def _calc_past_performance(self, horse_name: str) -> float:
        """過去成績（勝率）を計算する。"""
        results = self._find_horse_results(horse_name)
        if not results:
            return 0.0
        wins = sum(1 for _, pos in results if pos == 1)
        return wins / len(results)

    def _calc_jockey_performance(self, jockey_name: str) -> float:
        """騎手成績（勝率）を計算する。"""
        total = 0
        wins = 0
        for race in self._historical_races:
            if race.results is None:
                continue
            for entry in race.entries:
                if entry.jockey_name == jockey_name:
                    total += 1
                    for result in race.results:
                        if result.horse_number == entry.horse_number and result.finish_position == 1:
                            wins += 1
                            break
                    break
        if total == 0:
            return 0.0
        return wins / total

    def _calc_course_aptitude(self, horse_name: str, venue: str) -> float:
        """コース適性（同じ開催場での勝率）を計算する。"""
        results = self._find_horse_results(horse_name)
        venue_results = [(r, pos) for r, pos in results if r.venue == venue]
        if not venue_results:
            return 0.0
        wins = sum(1 for _, pos in venue_results if pos == 1)
        return wins / len(venue_results)

    def _calc_distance_aptitude(self, horse_name: str, distance: int) -> float:
        """距離適性（±200m以内の距離での勝率）を計算する。"""
        results = self._find_horse_results(horse_name)
        similar_results = [
            (r, pos) for r, pos in results
            if abs(r.distance - distance) <= 200
        ]
        if not similar_results:
            return 0.0
        wins = sum(1 for _, pos in similar_results if pos == 1)
        return wins / len(similar_results)

    def _calc_track_condition_aptitude(
        self, horse_name: str, track_condition: TrackCondition
    ) -> float:
        """馬場状態適性（同じ馬場状態での勝率）を計算する。"""
        results = self._find_horse_results(horse_name)
        condition_results = [
            (r, pos) for r, pos in results
            if r.track_condition == track_condition
        ]
        if not condition_results:
            return 0.0
        wins = sum(1 for _, pos in condition_results if pos == 1)
        return wins / len(condition_results)

    def _calc_gate_number(self, gate_number: int) -> float:
        """正規化された枠番を計算する。"""
        return gate_number / 8.0

    def _calc_weight_change(self, weight_change: int | None) -> float:
        """正規化された馬体重変動を計算する（[-1, 1]にクリッピング）。"""
        if weight_change is None:
            return 0.0
        normalized = weight_change / 20.0
        return max(-1.0, min(1.0, normalized))

    def _calc_class_performance(self, horse_name: str) -> float:
        """クラス実績（直近レースの平均着順を正規化）を計算する。"""
        results = self._find_horse_results(horse_name)
        if not results:
            return 0.5
        # 直近のレースを使用（最大5レース）
        recent_results = results[-5:]
        positions = [pos for _, pos in recent_results]
        # 着順を正規化: 1着=0.0, 18着=1.0 として線形変換
        # (position - 1) / (max_position - 1) として正規化
        # 一般的な出走頭数の最大値を18とする
        max_position = 18
        avg_position = sum(positions) / len(positions)
        normalized = (avg_position - 1) / (max_position - 1)
        return max(0.0, min(1.0, normalized))
