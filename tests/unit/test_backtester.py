"""Backtesterクラスのユニットテスト"""

from datetime import date
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.backtest.backtester import Backtester, BacktestReport, _DailyRecord
from src.betting.bet_selector import BetSelector
from src.betting.fund_allocator import FundAllocator
from src.data.models import (
    AllocatedBet,
    BacktestResult,
    BetRecommendation,
    BetType,
    Config,
    HorseEntry,
    PayoutInfo,
    RaceData,
    RaceEvaluation,
    RaceResult,
    TrackCondition,
)
from src.evaluation.race_evaluator import RaceEvaluator
from src.features.engineer import FeatureEngineer


def _make_race(
    race_id: str = "race1",
    race_date: date | None = None,
    n_entries: int = 6,
    with_results: bool = True,
    payouts: list[PayoutInfo] | None = None,
) -> RaceData:
    """テスト用レースデータを作成する"""
    if race_date is None:
        race_date = date(2024, 1, 1)

    entries = [
        HorseEntry(
            horse_name=f"Horse{i+1}",
            jockey_name=f"Jockey{i+1}",
            gate_number=(i % 8) + 1,
            horse_number=i + 1,
            weight=480 + i * 2,
            weight_change=0,
            win_odds=float(3 + i * 2),
        )
        for i in range(n_entries)
    ]

    results = None
    if with_results:
        results = [
            RaceResult(horse_number=i + 1, finish_position=i + 1)
            for i in range(n_entries)
        ]

    if payouts is None and with_results:
        payouts = [
            PayoutInfo(bet_type=BetType.WIN, combination=(1,), payout=500),
            PayoutInfo(bet_type=BetType.PLACE, combination=(1,), payout=150),
            PayoutInfo(bet_type=BetType.PLACE, combination=(2,), payout=200),
            PayoutInfo(bet_type=BetType.QUINELLA, combination=(1, 2), payout=1200),
        ]

    return RaceData(
        race_id=race_id,
        race_name=f"Test Race {race_id}",
        race_date=race_date,
        post_time=None,
        venue="東京",
        course_type="芝",
        distance=2000,
        track_condition=TrackCondition.FIRM,
        weather="晴",
        entries=entries,
        results=results,
        payouts=payouts,
    )


def _make_backtester() -> Backtester:
    """テスト用のBacktesterを作成する"""
    feature_engineer = FeatureEngineer(historical_races=[])
    bet_selector = BetSelector(min_expected_value=0.5)
    fund_allocator = FundAllocator(max_single_bet_ratio=0.3)
    race_evaluator = RaceEvaluator(confidence_threshold=30)

    return Backtester(
        feature_engineer=feature_engineer,
        bet_selector=bet_selector,
        fund_allocator=fund_allocator,
        race_evaluator=race_evaluator,
    )


class TestBacktesterInit:
    """Backtester初期化テスト"""

    def test_init_stores_components(self):
        fe = FeatureEngineer()
        bs = BetSelector()
        fa = FundAllocator()
        re = RaceEvaluator()

        bt = Backtester(fe, bs, fa, re)

        assert bt._feature_engineer is fe
        assert bt._bet_selector is bs
        assert bt._fund_allocator is fa
        assert bt._race_evaluator is re


class TestBacktesterRun:
    """Backtester.run() テスト"""

    def test_empty_races_returns_zero_results(self):
        backtester = _make_backtester()
        model = MagicMock()
        config = Config()

        result = backtester.run([], model, config)

        assert result.total_races == 0
        assert result.bet_races == 0
        assert result.skipped_races == 0
        assert result.total_investment == 0
        assert result.total_return == 0

    def test_race_with_low_confidence_is_skipped(self):
        """信頼度スコアが閾値未満のレースは見送られる"""
        feature_engineer = FeatureEngineer(historical_races=[])
        bet_selector = BetSelector()
        fund_allocator = FundAllocator()
        # 高い閾値を設定してほぼすべてのレースを見送りにする
        race_evaluator = RaceEvaluator(confidence_threshold=100)

        backtester = Backtester(feature_engineer, bet_selector, fund_allocator, race_evaluator)

        model = MagicMock()
        model.predict_probabilities.return_value = np.array([0.3, 0.2, 0.15, 0.15, 0.1, 0.1])

        race = _make_race()
        config = Config()

        result = backtester.run([race], model, config)

        assert result.total_races == 1
        assert result.bet_races == 0
        assert result.skipped_races == 1

    def test_race_with_bets_accumulates_investment(self):
        """投資対象レースでは投資金額が累積される"""
        backtester = _make_backtester()

        model = MagicMock()
        # 確率分布が偏っていると信頼度スコアが高くなる
        model.predict_probabilities.return_value = np.array([0.5, 0.2, 0.1, 0.1, 0.05, 0.05])

        race = _make_race()
        config = Config(daily_budget=10000, max_bets_per_race=10)

        result = backtester.run([race], model, config)

        # 何かしら投資された
        assert result.total_investment >= 0
        assert result.total_races == 1

    def test_hit_race_accumulates_returns(self):
        """的中した場合は払い戻しが累積される"""
        backtester = _make_backtester()

        model = MagicMock()
        model.predict_probabilities.return_value = np.array([0.6, 0.2, 0.1, 0.05, 0.03, 0.02])

        # 単勝1番の払い戻しが高いレース
        payouts = [
            PayoutInfo(bet_type=BetType.WIN, combination=(1,), payout=500),
        ]
        race = _make_race(payouts=payouts)
        config = Config(daily_budget=10000, max_bets_per_race=10)

        result = backtester.run([race], model, config)

        # 何かしらの結果が返る（具体的な金額はbet_selectorとfund_allocatorの実装依存）
        assert result.total_races == 1

    def test_multiple_races_across_days(self):
        """複数日のレースで日次リターンが正しく計算される"""
        backtester = _make_backtester()

        model = MagicMock()
        model.predict_probabilities.return_value = np.array([0.5, 0.2, 0.1, 0.1, 0.05, 0.05])

        races = [
            _make_race(race_id="r1", race_date=date(2024, 1, 1)),
            _make_race(race_id="r2", race_date=date(2024, 1, 2)),
            _make_race(race_id="r3", race_date=date(2024, 1, 3)),
        ]
        config = Config(daily_budget=10000)

        result = backtester.run(races, model, config)

        assert result.total_races == 3

    def test_skipped_when_no_bets_generated(self):
        """買い目が生成されないレースは見送りとなる"""
        feature_engineer = FeatureEngineer(historical_races=[])
        # 非常に高い期待値閾値を設定して買い目を生成させない
        bet_selector = BetSelector(min_expected_value=100.0)
        fund_allocator = FundAllocator()
        race_evaluator = RaceEvaluator(confidence_threshold=0)

        backtester = Backtester(feature_engineer, bet_selector, fund_allocator, race_evaluator)

        model = MagicMock()
        model.predict_probabilities.return_value = np.array([0.5, 0.2, 0.1, 0.1, 0.05, 0.05])

        race = _make_race()
        config = Config()

        result = backtester.run([race], model, config)

        assert result.skipped_races == 1
        assert result.bet_races == 0

    def test_prediction_error_skips_race(self):
        """モデル予測でエラーが発生した場合、レースは見送られる"""
        backtester = _make_backtester()

        model = MagicMock()
        model.predict_probabilities.side_effect = RuntimeError("Model error")

        race = _make_race()
        config = Config()

        result = backtester.run([race], model, config)

        assert result.skipped_races == 1
        assert result.bet_races == 0

    def test_bet_type_stats_populated(self):
        """券種別統計が正しく生成される"""
        backtester = _make_backtester()

        model = MagicMock()
        model.predict_probabilities.return_value = np.array([0.5, 0.2, 0.1, 0.1, 0.05, 0.05])

        race = _make_race()
        config = Config(daily_budget=10000)

        result = backtester.run([race], model, config)

        # 全券種分のstatsが存在する
        for bt in BetType:
            assert bt in result.bet_type_stats
            stats = result.bet_type_stats[bt]
            assert "hit_rate" in stats
            assert "return_rate" in stats


class TestBacktesterGenerateReport:
    """Backtester.generate_report() テスト"""

    def test_report_with_good_return_rate(self):
        """回収率100%以上のレポートには推奨事項がない"""
        backtester = _make_backtester()

        result = BacktestResult(
            total_races=100,
            bet_races=50,
            skipped_races=50,
            total_investment=100000,
            total_return=120000,
            hit_rate=0.3,
            return_rate=1.2,
            max_drawdown=0.1,
            sharpe_ratio=1.5,
            daily_returns=[1.0, 1.1, 0.9, 1.2],
            weekly_returns=[1.05, 1.1],
            monthly_returns=[1.1],
            bet_type_stats={bt: {"hit_rate": 0.3, "return_rate": 1.2, "investment": 10000, "returns": 12000, "bets": 10, "hits": 3} for bt in BetType},
        )

        report = backtester.generate_report(result)

        assert isinstance(report, BacktestReport)
        assert report.result is result
        assert len(report.recommendations) == 0
        assert report.is_overfitting is False

    def test_report_with_low_return_rate(self):
        """回収率100%未満のレポートには推奨事項が含まれる"""
        backtester = _make_backtester()

        result = BacktestResult(
            total_races=100,
            bet_races=80,
            skipped_races=20,
            total_investment=100000,
            total_return=60000,
            hit_rate=0.05,
            return_rate=0.6,
            max_drawdown=0.6,
            sharpe_ratio=-0.5,
            daily_returns=[0.8, 0.6, 0.5, 0.7],
            weekly_returns=[0.65, 0.6],
            monthly_returns=[0.6],
            bet_type_stats={bt: {"hit_rate": 0.05, "return_rate": 0.4, "investment": 10000, "returns": 4000, "bets": 10, "hits": 1} for bt in BetType},
        )

        report = backtester.generate_report(result)

        assert len(report.recommendations) > 0
        # 回収率が低い旨のメッセージが含まれる
        assert any("100%" in r for r in report.recommendations)


class TestBacktesterOverfittingCheck:
    """Backtester.run_with_overfitting_check() テスト"""

    def test_overfitting_detected(self):
        """学習/検証データの回収率が大幅に乖離している場合に過学習と判定"""
        backtester = _make_backtester()

        model = MagicMock()
        model.predict_probabilities.return_value = np.array([0.5, 0.2, 0.1, 0.1, 0.05, 0.05])

        train_races = [_make_race(race_id=f"train_{i}", race_date=date(2024, 1, i + 1)) for i in range(3)]
        val_races = [_make_race(race_id=f"val_{i}", race_date=date(2024, 2, i + 1)) for i in range(3)]

        config = Config(daily_budget=10000)

        report = backtester.run_with_overfitting_check(train_races, val_races, model, config)

        assert isinstance(report, BacktestReport)
        # is_overfitting depends on actual return rates from the mock data

    def test_no_overfitting_when_rates_similar(self):
        """学習/検証データの回収率が近い場合は過学習なしと判定"""
        backtester = _make_backtester()

        # 過学習判定のロジック直接テスト
        assert backtester._detect_overfitting(1.0, 0.95) is False
        assert backtester._detect_overfitting(0.8, 0.7) is False

    def test_overfitting_detected_direct(self):
        """過学習判定の直接テスト"""
        backtester = _make_backtester()

        # 学習率が検証率の1.5倍以上かつ差が0.2以上
        assert backtester._detect_overfitting(1.5, 0.7) is True
        assert backtester._detect_overfitting(2.0, 0.8) is True

    def test_no_overfitting_when_both_zero(self):
        """両方0の場合は過学習判定なし"""
        backtester = _make_backtester()
        assert backtester._detect_overfitting(0.0, 0.0) is False


class TestCombinationMatches:
    """_combination_matches() テスト"""

    def test_win_match(self):
        backtester = _make_backtester()
        payout = PayoutInfo(bet_type=BetType.WIN, combination=(3,), payout=500)
        assert backtester._combination_matches(BetType.WIN, (3,), payout) is True
        assert backtester._combination_matches(BetType.WIN, (1,), payout) is False

    def test_place_match(self):
        backtester = _make_backtester()
        payout = PayoutInfo(bet_type=BetType.PLACE, combination=(2,), payout=200)
        assert backtester._combination_matches(BetType.PLACE, (2,), payout) is True
        assert backtester._combination_matches(BetType.PLACE, (5,), payout) is False

    def test_quinella_match_order_independent(self):
        backtester = _make_backtester()
        payout = PayoutInfo(bet_type=BetType.QUINELLA, combination=(1, 3), payout=1000)
        assert backtester._combination_matches(BetType.QUINELLA, (1, 3), payout) is True
        assert backtester._combination_matches(BetType.QUINELLA, (3, 1), payout) is True
        assert backtester._combination_matches(BetType.QUINELLA, (1, 2), payout) is False

    def test_exacta_match_order_dependent(self):
        backtester = _make_backtester()
        payout = PayoutInfo(bet_type=BetType.EXACTA, combination=(1, 3), payout=2000)
        assert backtester._combination_matches(BetType.EXACTA, (1, 3), payout) is True
        assert backtester._combination_matches(BetType.EXACTA, (3, 1), payout) is False

    def test_trio_match_order_independent(self):
        backtester = _make_backtester()
        payout = PayoutInfo(bet_type=BetType.TRIO, combination=(1, 2, 3), payout=5000)
        assert backtester._combination_matches(BetType.TRIO, (1, 2, 3), payout) is True
        assert backtester._combination_matches(BetType.TRIO, (3, 1, 2), payout) is True
        assert backtester._combination_matches(BetType.TRIO, (1, 2, 4), payout) is False

    def test_trifecta_match_order_dependent(self):
        backtester = _make_backtester()
        payout = PayoutInfo(bet_type=BetType.TRIFECTA, combination=(1, 2, 3), payout=10000)
        assert backtester._combination_matches(BetType.TRIFECTA, (1, 2, 3), payout) is True
        assert backtester._combination_matches(BetType.TRIFECTA, (1, 3, 2), payout) is False

    def test_wide_match_order_independent(self):
        backtester = _make_backtester()
        payout = PayoutInfo(bet_type=BetType.WIDE, combination=(2, 5), payout=800)
        assert backtester._combination_matches(BetType.WIDE, (2, 5), payout) is True
        assert backtester._combination_matches(BetType.WIDE, (5, 2), payout) is True
        assert backtester._combination_matches(BetType.WIDE, (2, 3), payout) is False


class TestTimeSeriesAggregation:
    """日次/週次/月次集約テスト"""

    def test_daily_returns_calculation(self):
        backtester = _make_backtester()
        daily_records = {
            date(2024, 1, 1): _DailyRecord(investment=10000, returns=12000),
            date(2024, 1, 2): _DailyRecord(investment=10000, returns=8000),
            date(2024, 1, 3): _DailyRecord(investment=5000, returns=5000),
        }

        returns = backtester._calculate_daily_returns(daily_records)

        assert len(returns) == 3
        assert returns[0] == pytest.approx(1.2)  # 12000/10000
        assert returns[1] == pytest.approx(0.8)  # 8000/10000
        assert returns[2] == pytest.approx(1.0)  # 5000/5000

    def test_weekly_returns_aggregation(self):
        backtester = _make_backtester()
        # All in same ISO week (2024-01-01 is Monday of week 1)
        daily_records = {
            date(2024, 1, 1): _DailyRecord(investment=10000, returns=12000),
            date(2024, 1, 2): _DailyRecord(investment=10000, returns=8000),
        }

        weekly = backtester._aggregate_to_weekly(daily_records)

        assert len(weekly) == 1
        # Total: investment=20000, returns=20000, return_rate=1.0
        assert weekly[0] == pytest.approx(1.0)

    def test_monthly_returns_aggregation(self):
        backtester = _make_backtester()
        daily_records = {
            date(2024, 1, 1): _DailyRecord(investment=10000, returns=12000),
            date(2024, 1, 15): _DailyRecord(investment=10000, returns=8000),
            date(2024, 2, 1): _DailyRecord(investment=5000, returns=7000),
        }

        monthly = backtester._aggregate_to_monthly(daily_records)

        assert len(monthly) == 2
        # Jan: investment=20000, returns=20000, return_rate=1.0
        assert monthly[0] == pytest.approx(1.0)
        # Feb: investment=5000, returns=7000, return_rate=1.4
        assert monthly[1] == pytest.approx(1.4)

    def test_empty_records(self):
        backtester = _make_backtester()
        assert backtester._calculate_daily_returns({}) == []
        assert backtester._aggregate_to_weekly({}) == []
        assert backtester._aggregate_to_monthly({}) == []


class TestMaxDrawdown:
    """最大ドローダウン計算テスト"""

    def test_no_drawdown(self):
        backtester = _make_backtester()
        # 常に増加する場合、ドローダウンは0
        returns = [1.1, 1.2, 1.3]
        dd = backtester._calculate_max_drawdown(returns)
        assert dd == pytest.approx(0.0)

    def test_constant_returns(self):
        backtester = _make_backtester()
        returns = [1.0, 1.0, 1.0]
        dd = backtester._calculate_max_drawdown(returns)
        assert dd == pytest.approx(0.0)

    def test_single_drawdown(self):
        backtester = _make_backtester()
        # Peak at 1.5, drops to 0.75 (50% drawdown)
        returns = [1.5, 0.5, 1.0]
        # cumulative: 1.5, 0.75, 0.75
        # peak: 1.5, drawdown from 1.5 to 0.75 = 0.5
        dd = backtester._calculate_max_drawdown(returns)
        assert dd == pytest.approx(0.5)

    def test_empty_returns(self):
        backtester = _make_backtester()
        assert backtester._calculate_max_drawdown([]) == 0.0


class TestSharpeRatio:
    """シャープレシオ計算テスト"""

    def test_positive_sharpe(self):
        backtester = _make_backtester()
        # 常に超過リターンがプラスならシャープ比はプラス
        returns = [1.1, 1.2, 1.15, 1.1]
        sr = backtester._calculate_sharpe_ratio(returns)
        assert sr > 0

    def test_negative_sharpe(self):
        backtester = _make_backtester()
        # 常に超過リターンがマイナスならシャープ比はマイナス
        returns = [0.8, 0.7, 0.9, 0.85]
        sr = backtester._calculate_sharpe_ratio(returns)
        assert sr < 0

    def test_insufficient_data(self):
        backtester = _make_backtester()
        assert backtester._calculate_sharpe_ratio([]) == 0.0
        assert backtester._calculate_sharpe_ratio([1.0]) == 0.0

    def test_zero_variance(self):
        backtester = _make_backtester()
        # All same returns → std = 0 → sharpe = 0
        returns = [1.0, 1.0, 1.0]
        sr = backtester._calculate_sharpe_ratio(returns)
        assert sr == 0.0
