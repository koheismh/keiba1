"""予測パイプラインのユニットテスト"""

from datetime import date, time
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.data.models import (
    AllocatedBet,
    BetRecommendation,
    BetType,
    Config,
    HorseEntry,
    RaceData,
    RaceEvaluation,
    TrackCondition,
)
from src.output.formatter import RacePrediction
from src.pipeline import PredictionPipeline, RaceProcessingError


def _make_race(race_id: str = "202401010101", n_horses: int = 6) -> RaceData:
    """テスト用レースデータを生成する。"""
    entries = [
        HorseEntry(
            horse_name=f"Horse{i}",
            jockey_name=f"Jockey{i}",
            gate_number=min(i, 8),
            horse_number=i,
            weight=480 + i * 2,
            weight_change=i - 3,
            win_odds=float(3 + i * 2),
        )
        for i in range(1, n_horses + 1)
    ]
    return RaceData(
        race_id=race_id,
        race_name=f"テストレース{race_id[-2:]}",
        race_date=date(2024, 1, 1),
        post_time=time(10, 0),
        venue="東京",
        course_type="芝",
        distance=1600,
        track_condition=TrackCondition.FIRM,
        weather="晴",
        entries=entries,
        results=None,
        payouts=None,
    )


def _make_config() -> Config:
    """テスト用設定を生成する。"""
    return Config(
        confidence_threshold=50,
        max_bets_per_race=10,
        min_expected_value=1.0,
        daily_budget=10000,
        target_bet_types=[BetType.WIN],
        max_single_bet_ratio=0.3,
    )


class TestPredictionPipelineInit:
    """PredictionPipeline初期化テスト"""

    def test_initializes_all_components(self) -> None:
        config = _make_config()
        model = MagicMock()

        pipeline = PredictionPipeline(config=config, model=model)

        assert pipeline.config is config
        assert pipeline.model is model
        assert pipeline.feature_engineer is not None
        assert pipeline.bet_selector is not None
        assert pipeline.fund_allocator is not None
        assert pipeline.race_evaluator is not None
        assert pipeline.output_formatter is not None

    def test_passes_config_to_components(self) -> None:
        config = _make_config()
        model = MagicMock()

        pipeline = PredictionPipeline(config=config, model=model)

        assert pipeline.bet_selector.min_expected_value == config.min_expected_value
        assert pipeline.fund_allocator.max_single_bet_ratio == config.max_single_bet_ratio
        assert pipeline.race_evaluator.confidence_threshold == config.confidence_threshold

    def test_accepts_historical_races(self) -> None:
        config = _make_config()
        model = MagicMock()
        historical = [_make_race("hist01")]

        pipeline = PredictionPipeline(
            config=config, model=model, historical_races=historical
        )

        assert pipeline.feature_engineer._historical_races == historical


class TestPredictRaceDay:
    """predict_race_day テスト"""

    def test_processes_all_races(self) -> None:
        config = _make_config()
        model = MagicMock()
        # Model returns uniform probabilities
        model.predict_probabilities = MagicMock(
            return_value=np.array([0.3, 0.2, 0.2, 0.15, 0.1, 0.05])
        )

        pipeline = PredictionPipeline(config=config, model=model)
        races = [_make_race(f"2024010101{i:02d}") for i in range(1, 4)]

        output = pipeline.predict_race_day(races)

        # Should have called predict_probabilities for each race
        assert model.predict_probabilities.call_count == 3
        # Output should contain text
        assert len(output) > 0

    def test_graceful_degradation_on_error(self) -> None:
        """一部レースでエラーが発生しても他レースは処理を継続する"""
        config = _make_config()
        model = MagicMock()

        # First call succeeds, second raises, third succeeds
        call_count = [0]

        def side_effect(features: np.ndarray) -> np.ndarray:
            call_count[0] += 1
            if call_count[0] == 2:
                raise RuntimeError("予測エラー")
            n = len(features)
            probs = np.ones(n) / n
            return probs

        model.predict_probabilities = MagicMock(side_effect=side_effect)

        pipeline = PredictionPipeline(config=config, model=model)
        races = [_make_race(f"2024010101{i:02d}") for i in range(1, 4)]

        output = pipeline.predict_race_day(races)

        # Should continue despite error in second race
        assert model.predict_probabilities.call_count == 3
        # Should contain error information
        assert "処理エラー" in output
        assert "予測エラー" in output

    def test_empty_races_returns_no_predictions_message(self) -> None:
        config = _make_config()
        model = MagicMock()

        pipeline = PredictionPipeline(config=config, model=model)

        output = pipeline.predict_race_day([])

        assert "予測対象レースがありません" in output

    def test_race_with_no_entries_raises_and_is_handled(self) -> None:
        """出走馬がないレースはエラーとして処理される"""
        config = _make_config()
        model = MagicMock()

        pipeline = PredictionPipeline(config=config, model=model)
        # Race with no entries
        race = RaceData(
            race_id="empty_race",
            race_name="空レース",
            race_date=date(2024, 1, 1),
            post_time=time(10, 0),
            venue="東京",
            course_type="芝",
            distance=1600,
            track_condition=TrackCondition.FIRM,
            weather="晴",
            entries=[],
            results=None,
            payouts=None,
        )

        output = pipeline.predict_race_day([race])

        # Should report the error gracefully
        assert "処理エラー" in output

    def test_skipped_race_shows_reason(self) -> None:
        """信頼度が低いレースは見送り理由が出力される"""
        config = Config(
            confidence_threshold=99,  # Very high threshold → all races skipped
            max_bets_per_race=10,
            min_expected_value=1.0,
            daily_budget=10000,
            target_bet_types=[BetType.WIN],
            max_single_bet_ratio=0.3,
        )
        model = MagicMock()
        # Nearly uniform distribution → low confidence
        model.predict_probabilities = MagicMock(
            return_value=np.array([0.17, 0.17, 0.17, 0.17, 0.16, 0.16])
        )

        pipeline = PredictionPipeline(config=config, model=model)
        races = [_make_race()]

        output = pipeline.predict_race_day(races)

        assert "見送り" in output


class TestRunBacktest:
    """run_backtest テスト"""

    def test_delegates_to_backtester(self) -> None:
        config = _make_config()
        model = MagicMock()
        model.predict_probabilities = MagicMock(
            return_value=np.array([0.3, 0.2, 0.2, 0.15, 0.1, 0.05])
        )

        pipeline = PredictionPipeline(config=config, model=model)
        races = [_make_race()]

        result = pipeline.run_backtest(races)

        # Should return a BacktestResult
        assert result.total_races == 1


class TestExtractRaceFeatures:
    """_extract_race_features テスト"""

    def test_extracts_features_for_all_horses(self) -> None:
        config = _make_config()
        model = MagicMock()
        pipeline = PredictionPipeline(config=config, model=model)

        race = _make_race(n_horses=4)
        features = pipeline._extract_race_features(race)

        assert features.shape[0] == 4
        assert features.shape[1] == 8  # 8 features per engineer

    def test_raises_for_empty_entries(self) -> None:
        config = _make_config()
        model = MagicMock()
        pipeline = PredictionPipeline(config=config, model=model)

        race = RaceData(
            race_id="empty",
            race_name="空",
            race_date=date(2024, 1, 1),
            post_time=None,
            venue="東京",
            course_type="芝",
            distance=1600,
            track_condition=TrackCondition.FIRM,
            weather=None,
            entries=[],
            results=None,
            payouts=None,
        )

        with pytest.raises(ValueError, match="出走馬がいません"):
            pipeline._extract_race_features(race)
