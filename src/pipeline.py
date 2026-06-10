"""予測パイプライン - 全コンポーネントを統合する。

データ取得 → 特徴量抽出 → 予測 → レース評価 → 買い目選択 → 資金配分 → 出力
のフルパイプラインを接続し、エラー発生時のグレースフルデグラデーション
（一部レース失敗時も他レースは処理継続）を実装する。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from src.backtest.backtester import Backtester
from src.betting.bet_selector import BetSelector
from src.betting.fund_allocator import FundAllocator
from src.data.models import (
    AllocatedBet,
    BacktestResult,
    Config,
    RaceData,
    RaceEvaluation,
)
from src.evaluation.race_evaluator import RaceEvaluator
from src.features.engineer import FeatureEngineer
from src.output.formatter import OutputFormatter, RacePrediction
from src.prediction.model import PredictionModel

logger = logging.getLogger(__name__)


@dataclass
class RaceProcessingError:
    """レース処理エラー情報"""

    race_id: str
    race_name: str
    error_message: str


class PredictionPipeline:
    """予測パイプライン - 全コンポーネントを統合する。

    データ取得 → 特徴量抽出 → 予測 → レース評価 → 買い目選択 → 資金配分 → 出力
    のフルパイプラインを接続する。
    """

    def __init__(
        self,
        config: Config,
        model: PredictionModel,
        historical_races: list[RaceData] | None = None,
    ) -> None:
        """Initialize all pipeline components from config.

        Args:
            config: システム設定
            model: 学習済み予測モデル
            historical_races: 特徴量抽出に使用する過去レースデータ
        """
        self.config = config
        self.model = model
        self.feature_engineer = FeatureEngineer(historical_races=historical_races)
        self.bet_selector = BetSelector(
            min_expected_value=config.min_expected_value,
            target_bet_types=config.target_bet_types,
        )
        self.fund_allocator = FundAllocator(
            max_single_bet_ratio=config.max_single_bet_ratio,
        )
        self.race_evaluator = RaceEvaluator(
            confidence_threshold=config.confidence_threshold,
        )
        self.output_formatter = OutputFormatter()

    def predict_race_day(self, races: list[RaceData]) -> str:
        """Run the full pipeline for a day of races.

        For each race:
        1. Extract features
        2. Predict probabilities
        3. Evaluate race
        4. If should_bet: select bets, allocate funds
        5. Collect results

        On error for a single race, skip it and continue with remaining races.
        Return formatted output.

        Args:
            races: 当日のレースデータリスト

        Returns:
            フォーマット済みの予測結果文字列
        """
        predictions: list[RacePrediction] = []
        errors: list[RaceProcessingError] = []

        for race in races:
            try:
                prediction = self._process_single_race(race)
                predictions.append(prediction)
            except Exception as e:
                error_info = RaceProcessingError(
                    race_id=race.race_id,
                    race_name=race.race_name,
                    error_message=str(e),
                )
                errors.append(error_info)
                logger.warning(
                    "レース処理エラー [%s] %s: %s",
                    race.race_id,
                    race.race_name,
                    e,
                )
                continue

        # Apply day-level evaluation (e.g., all races below threshold)
        if predictions:
            evaluations = [p.evaluation for p in predictions]
            updated_evaluations = self.race_evaluator.evaluate_race_day(evaluations)

            # Update predictions with day-level evaluations
            for i, updated_eval in enumerate(updated_evaluations):
                if not updated_eval.should_bet and predictions[i].evaluation.should_bet:
                    # Day-level override: clear allocated bets
                    predictions[i] = RacePrediction(
                        race=predictions[i].race,
                        evaluation=updated_eval,
                        allocated_bets=[],
                    )

        # Format output
        output = self.output_formatter.format_predictions(predictions)

        # Append error information if any
        if errors:
            error_lines = [
                "",
                "=" * 60,
                "【処理エラー】",
                "=" * 60,
            ]
            for err in errors:
                error_lines.append(
                    f"  ⚠ [{err.race_id}] {err.race_name}: {err.error_message}"
                )
            error_lines.append(
                f"  （{len(errors)}レースで処理エラーが発生しました）"
            )
            error_lines.append("=" * 60)
            output += "\n" + "\n".join(error_lines)

        return output

    def run_backtest(self, races: list[RaceData]) -> BacktestResult:
        """Run backtester on provided races.

        Args:
            races: バックテスト対象レースリスト

        Returns:
            BacktestResult: バックテスト結果
        """
        backtester = Backtester(
            feature_engineer=self.feature_engineer,
            bet_selector=self.bet_selector,
            fund_allocator=self.fund_allocator,
            race_evaluator=self.race_evaluator,
        )
        return backtester.run(races, self.model, self.config)

    def _process_single_race(self, race: RaceData) -> RacePrediction:
        """Process a single race through the full pipeline.

        Args:
            race: レースデータ

        Returns:
            RacePrediction: 1レースの予測結果

        Raises:
            Exception: 特徴量抽出や予測で発生した例外
        """
        # Step 1: Extract features for each horse
        features = self._extract_race_features(race)

        # Step 2: Predict probabilities
        probabilities = self.model.predict_probabilities(features)

        # Step 3: Evaluate race confidence
        evaluation = self.race_evaluator.evaluate(race, probabilities)

        # Step 4: If should_bet, select bets and allocate funds
        allocated_bets: list[AllocatedBet] = []
        if evaluation.should_bet:
            bets = self.bet_selector.select_bets(
                race, probabilities, self.config.max_bets_per_race
            )
            if bets:
                allocated_bets = self.fund_allocator.allocate(
                    bets, self.config.daily_budget
                )

            # If no bets found after selection, mark as skip
            if not allocated_bets:
                evaluation = RaceEvaluation(
                    race_id=evaluation.race_id,
                    confidence_score=evaluation.confidence_score,
                    should_bet=False,
                    skip_reason="期待値が基準を超える買い目が見つかりませんでした",
                    factors=evaluation.factors,
                )

        return RacePrediction(
            race=race,
            evaluation=evaluation,
            allocated_bets=allocated_bets,
        )

    def _extract_race_features(self, race: RaceData) -> np.ndarray:
        """Extract features for all horses in a race.

        Args:
            race: レースデータ

        Returns:
            shape (n_horses, n_features) の特徴量配列

        Raises:
            ValueError: 出走馬がいない場合
        """
        if not race.entries:
            raise ValueError(f"レース {race.race_id} に出走馬がいません")

        features_list = []
        for entry in race.entries:
            fv = self.feature_engineer.extract_features(race, entry)
            features_list.append(fv.values)

        return np.array(features_list)
