"""レース評価・見送り判定モジュール。

Race_Evaluator は各レースの予測信頼度を評価し、参加・見送りを判定する。
信頼度スコアは 0〜100 の整数で算出され、閾値以下の場合は見送り判定となる。
"""

import numpy as np

from src.data.models import RaceData, RaceEvaluation


class RaceEvaluator:
    """レースの信頼度スコアを算出し見送り判定を行う。

    信頼度スコアは以下の3要素から算出される:
    - unpredictability (荒れやすさ): 確率分布のエントロピーに基づく
    - strength_gap (実力差の明確さ): トップと平均の予測差
    - data_completeness (データの充実度): オッズ・馬体重データの充実度
    """

    def __init__(self, confidence_threshold: int = 50) -> None:
        """Initialize with confidence threshold.

        Args:
            confidence_threshold: 信頼度スコアの閾値 (0-100)。
                スコアがこの値未満の場合、見送り判定となる。
        """
        self.confidence_threshold = confidence_threshold

    def evaluate(self, race: RaceData, predictions: np.ndarray) -> RaceEvaluation:
        """レースの信頼度スコアと参加判定を行う。

        Args:
            race: レースデータ
            predictions: 各馬の着順確率分布 (numpy配列、合計≒1.0)

        Returns:
            RaceEvaluation: 信頼度スコア、投資判定、見送り理由を含む評価結果
        """
        unpredictability = self._calculate_unpredictability(predictions)
        strength_gap = self._calculate_strength_gap(predictions)
        data_completeness = self._calculate_data_completeness(race)

        # スコア算出: 40% 実力差 + 30% (1 - 荒れやすさ) + 30% データ充実度
        score = int(
            40 * strength_gap + 30 * (1 - unpredictability) + 30 * data_completeness
        )

        # スコアを0-100に制限
        score = max(0, min(100, score))

        should_bet = score >= self.confidence_threshold

        skip_reason: str | None = None
        if not should_bet:
            skip_reason = self._generate_skip_reason(
                unpredictability, strength_gap, data_completeness
            )

        factors = {
            "unpredictability": unpredictability,
            "strength_gap": strength_gap,
            "data_completeness": data_completeness,
        }

        return RaceEvaluation(
            race_id=race.race_id,
            confidence_score=score,
            should_bet=should_bet,
            skip_reason=skip_reason,
            factors=factors,
        )

    def evaluate_race_day(
        self, evaluations: list[RaceEvaluation]
    ) -> list[RaceEvaluation]:
        """1日のレース評価を行う。

        全レースが閾値以下の場合は全レース見送りと判定する。

        Args:
            evaluations: 各レースの評価結果リスト

        Returns:
            更新された評価結果リスト。全レースが閾値以下の場合、
            すべてのレースが should_bet=False に設定される。
        """
        if not evaluations:
            return evaluations

        # 既に should_bet=True のレースが1つでもあればそのまま返す
        has_any_bet = any(e.should_bet for e in evaluations)
        if has_any_bet:
            return evaluations

        # 全レースが閾値以下の場合、全レース見送りを明示する
        updated: list[RaceEvaluation] = []
        for evaluation in evaluations:
            reason = evaluation.skip_reason or ""
            if reason:
                reason += "。"
            reason += "本日は全レースの信頼度スコアが閾値を下回ったため全レース見送り"
            updated.append(
                RaceEvaluation(
                    race_id=evaluation.race_id,
                    confidence_score=evaluation.confidence_score,
                    should_bet=False,
                    skip_reason=reason,
                    factors=evaluation.factors,
                )
            )
        return updated

    def _calculate_unpredictability(self, predictions: np.ndarray) -> float:
        """確率分布のエントロピーに基づく荒れやすさを算出する。

        エントロピーが高い（確率が均等に分布）ほど荒れやすい。
        正規化エントロピー (0-1) を返す。

        Args:
            predictions: 各馬の着順確率分布

        Returns:
            0-1 の荒れやすさ指標。1に近いほど荒れやすい。
        """
        n = len(predictions)
        if n <= 1:
            return 0.0

        # 確率が0や負にならないようクリッピング
        probs = np.clip(predictions, 1e-10, None)
        # 正規化（合計を1にする）
        probs = probs / probs.sum()

        # シャノンエントロピー
        entropy = -np.sum(probs * np.log2(probs))

        # 最大エントロピー（一様分布）で正規化
        max_entropy = np.log2(n)
        if max_entropy == 0:
            return 0.0

        normalized_entropy = entropy / max_entropy
        return float(np.clip(normalized_entropy, 0.0, 1.0))

    def _calculate_strength_gap(self, predictions: np.ndarray) -> float:
        """トップと平均の予測差に基づく実力差の明確さを算出する。

        トップの確率が平均よりも大きく離れていれば、
        明確な実力差があると判断する。

        Args:
            predictions: 各馬の着順確率分布

        Returns:
            0-1 の実力差指標。1に近いほど実力差が明確。
        """
        n = len(predictions)
        if n <= 1:
            return 1.0

        top_prob = float(np.max(predictions))
        avg_prob = float(np.mean(predictions))

        if avg_prob == 0:
            return 0.0

        # トップと平均の差を平均で正規化し、0-1にスケーリング
        # 最大理論値: (n-1)/1 だが、実用的には差が大きいほどスコア高
        gap_ratio = (top_prob - avg_prob) / avg_prob

        # gap_ratioを0-1に変換。gap_ratio >= (n-1) のとき1.0
        # 実用的なスケーリング: gap_ratio / (n - 1) でクリップ
        max_gap = n - 1  # 一頭に全確率が集中した場合の理論最大
        strength = gap_ratio / max_gap if max_gap > 0 else 0.0

        return float(np.clip(strength, 0.0, 1.0))

    def _calculate_data_completeness(self, race: RaceData) -> float:
        """出走馬のデータ充実度を算出する。

        オッズデータと馬体重データの有無を基に算出する。

        Args:
            race: レースデータ

        Returns:
            0-1 のデータ充実度指標。1に近いほどデータが充実。
        """
        if not race.entries:
            return 0.0

        total_entries = len(race.entries)
        odds_count = sum(
            1 for entry in race.entries if entry.win_odds is not None
        )
        weight_count = sum(
            1 for entry in race.entries if entry.weight is not None
        )

        # オッズと馬体重の充実度を均等に重み付け
        odds_ratio = odds_count / total_entries
        weight_ratio = weight_count / total_entries

        return float((odds_ratio + weight_ratio) / 2.0)

    def _generate_skip_reason(
        self,
        unpredictability: float,
        strength_gap: float,
        data_completeness: float,
    ) -> str:
        """見送り理由を生成する。

        各要素の値に基づいて、どの要素が見送り判定に寄与したかを説明する。

        Args:
            unpredictability: 荒れやすさ (0-1)
            strength_gap: 実力差の明確さ (0-1)
            data_completeness: データ充実度 (0-1)

        Returns:
            見送り理由を説明する文字列
        """
        reasons: list[str] = []

        if unpredictability > 0.7:
            reasons.append(
                f"荒れやすさが高い(unpredictability={unpredictability:.2f})"
            )
        if strength_gap < 0.3:
            reasons.append(
                f"実力差が不明確(strength_gap={strength_gap:.2f})"
            )
        if data_completeness < 0.5:
            reasons.append(
                f"データが不足(data_completeness={data_completeness:.2f})"
            )

        if not reasons:
            reasons.append("総合スコアが閾値未満")

        return "見送り: " + "、".join(reasons)
