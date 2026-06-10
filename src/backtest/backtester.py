"""バックテスト実行エンジン。

過去のレースデータに対してモデルの性能を検証するモジュール。
全パイプラインコンポーネント（特徴量抽出→予測→レース評価→買い目選択→資金配分）を
統合し、仮想的に馬券を購入したシミュレーションを実行する。
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date

import numpy as np

from src.betting.bet_selector import BetSelector
from src.betting.fund_allocator import FundAllocator
from src.data.models import (
    AllocatedBet,
    BacktestResult,
    BetType,
    Config,
    PayoutInfo,
    RaceData,
)
from src.evaluation.race_evaluator import RaceEvaluator
from src.features.engineer import FeatureEngineer
from src.prediction.model import PredictionModel


@dataclass
class BacktestReport:
    """バックテストレポート"""

    result: BacktestResult
    recommendations: list[str]  # パラメータ調整推奨事項
    is_overfitting: bool  # 過学習判定
    train_return_rate: float  # 学習用データでの回収率


@dataclass
class _DailyRecord:
    """日次の投資・払い戻し記録"""

    investment: int = 0
    returns: int = 0


class Backtester:
    """バックテスト実行エンジン。

    全パイプラインコンポーネントを統合し、指定期間のレースデータに対して
    仮想馬券購入シミュレーションを実行する。
    """

    def __init__(
        self,
        feature_engineer: FeatureEngineer,
        bet_selector: BetSelector,
        fund_allocator: FundAllocator,
        race_evaluator: RaceEvaluator,
    ) -> None:
        """Initialize with all pipeline components.

        Args:
            feature_engineer: 特徴量抽出器
            bet_selector: 買い目選択器
            fund_allocator: 資金配分器
            race_evaluator: レース評価器
        """
        self._feature_engineer = feature_engineer
        self._bet_selector = bet_selector
        self._fund_allocator = fund_allocator
        self._race_evaluator = race_evaluator

    def run(
        self, races: list[RaceData], model: PredictionModel, config: Config
    ) -> BacktestResult:
        """バックテストを実行する。

        指定されたレースリストに対してモデルを適用し、
        仮想的に馬券を購入したシミュレーションを実行する。

        For each race:
        1. Extract features for each horse
        2. Predict probabilities
        3. Evaluate race confidence
        4. If should_bet: select bets, allocate funds
        5. Check actual results to determine if bets hit
        6. Accumulate investment and returns

        Args:
            races: バックテスト対象レースリスト
            model: 学習済み予測モデル
            config: システム設定

        Returns:
            BacktestResult: バックテスト結果
        """
        total_races = len(races)
        bet_races = 0
        skipped_races = 0
        total_investment = 0
        total_return = 0
        total_bets_placed = 0
        total_hits = 0

        # 日次記録（日付 → 投資/払い戻し）
        daily_records: dict[date, _DailyRecord] = defaultdict(_DailyRecord)

        # 券種別統計
        bet_type_stats: dict[BetType, dict[str, float]] = {
            bt: {"investment": 0, "returns": 0, "bets": 0, "hits": 0}
            for bt in BetType
        }

        for race in races:
            # Step 1: Extract features for each horse
            race_features = self._extract_race_features(race)
            if race_features is None or len(race_features) == 0:
                skipped_races += 1
                continue

            # Step 2: Predict probabilities
            try:
                probabilities = model.predict_probabilities(race_features)
            except Exception:
                skipped_races += 1
                continue

            # Step 3: Evaluate race confidence
            evaluation = self._race_evaluator.evaluate(race, probabilities)

            if not evaluation.should_bet:
                skipped_races += 1
                continue

            # Step 4: Select bets and allocate funds
            bets = self._bet_selector.select_bets(
                race, probabilities, config.max_bets_per_race
            )

            if not bets:
                skipped_races += 1
                continue

            allocated_bets = self._fund_allocator.allocate(bets, config.daily_budget)

            if not allocated_bets:
                skipped_races += 1
                continue

            # Step 5: Check actual results to determine hits
            bet_races += 1
            race_investment = sum(ab.amount for ab in allocated_bets)
            race_return = 0
            race_hits = 0

            for allocated_bet in allocated_bets:
                bet_amount = allocated_bet.amount
                total_investment += bet_amount
                total_bets_placed += 1

                bt = allocated_bet.recommendation.bet_type
                bet_type_stats[bt]["investment"] += bet_amount
                bet_type_stats[bt]["bets"] += 1

                # 的中判定
                payout = self._check_hit(allocated_bet, race)
                if payout > 0:
                    total_return += payout
                    race_return += payout
                    total_hits += 1
                    race_hits += 1
                    bet_type_stats[bt]["returns"] += payout
                    bet_type_stats[bt]["hits"] += 1

            # 日次記録の更新
            record = daily_records[race.race_date]
            record.investment += race_investment
            record.returns += race_return

        # 結果計算
        hit_rate = total_hits / total_bets_placed if total_bets_placed > 0 else 0.0
        return_rate = total_return / total_investment if total_investment > 0 else 0.0

        # 日次リターン計算
        daily_returns = self._calculate_daily_returns(daily_records)

        # 週次・月次リターン計算
        weekly_returns = self._aggregate_to_weekly(daily_records)
        monthly_returns = self._aggregate_to_monthly(daily_records)

        # 最大ドローダウン計算
        max_drawdown = self._calculate_max_drawdown(daily_returns)

        # シャープレシオ計算
        sharpe_ratio = self._calculate_sharpe_ratio(daily_returns)

        # 券種別統計の最終形式
        final_bet_type_stats: dict[BetType, dict[str, float]] = {}
        for bt, stats in bet_type_stats.items():
            bets_count = stats["bets"]
            final_bet_type_stats[bt] = {
                "hit_rate": stats["hits"] / bets_count if bets_count > 0 else 0.0,
                "return_rate": stats["returns"] / stats["investment"]
                if stats["investment"] > 0
                else 0.0,
                "investment": stats["investment"],
                "returns": stats["returns"],
                "bets": bets_count,
                "hits": stats["hits"],
            }

        return BacktestResult(
            total_races=total_races,
            bet_races=bet_races,
            skipped_races=skipped_races,
            total_investment=total_investment,
            total_return=total_return,
            hit_rate=hit_rate,
            return_rate=return_rate,
            max_drawdown=max_drawdown,
            sharpe_ratio=sharpe_ratio,
            daily_returns=daily_returns,
            weekly_returns=weekly_returns,
            monthly_returns=monthly_returns,
            bet_type_stats=final_bet_type_stats,
        )

    def generate_report(self, result: BacktestResult) -> BacktestReport:
        """バックテスト結果からレポートを生成する。

        - 回収率100%未満時のパラメータ調整推奨事項を提示する
        - 検証用vs学習用の回収率比較による過学習判定を行う

        Args:
            result: バックテスト結果

        Returns:
            BacktestReport: レポート（推奨事項、過学習判定含む）
        """
        recommendations = self._generate_recommendations(result)
        train_return_rate = result.return_rate  # 単体呼び出し時はresult自身の値を使用
        is_overfitting = False

        return BacktestReport(
            result=result,
            recommendations=recommendations,
            is_overfitting=is_overfitting,
            train_return_rate=train_return_rate,
        )

    def run_with_overfitting_check(
        self,
        train_races: list[RaceData],
        validation_races: list[RaceData],
        model: PredictionModel,
        config: Config,
    ) -> BacktestReport:
        """学習用と検証用で別々にバックテストを実行し、過学習判定を行う。

        Args:
            train_races: 学習用レースデータ
            validation_races: 検証用レースデータ
            model: 学習済み予測モデル
            config: システム設定

        Returns:
            BacktestReport: 過学習判定を含むレポート
        """
        train_result = self.run(train_races, model, config)
        validation_result = self.run(validation_races, model, config)

        recommendations = self._generate_recommendations(validation_result)
        is_overfitting = self._detect_overfitting(
            train_result.return_rate, validation_result.return_rate
        )

        if is_overfitting:
            recommendations.append(
                "過学習の兆候あり: 学習データ回収率"
                f"({train_result.return_rate:.1%})と検証データ回収率"
                f"({validation_result.return_rate:.1%})に大きな乖離があります。"
                "正則化の強化またはデータ量の増加を検討してください。"
            )

        return BacktestReport(
            result=validation_result,
            recommendations=recommendations,
            is_overfitting=is_overfitting,
            train_return_rate=train_result.return_rate,
        )

    # --- Private methods ---

    def _extract_race_features(self, race: RaceData) -> np.ndarray | None:
        """レース全馬の特徴量を抽出する。

        Args:
            race: レースデータ

        Returns:
            shape (n_horses, n_features) の特徴量配列。
            出走馬がいない場合はNone。
        """
        if not race.entries:
            return None

        features_list = []
        for entry in race.entries:
            fv = self._feature_engineer.extract_features(race, entry)
            features_list.append(fv.values)

        return np.array(features_list)

    def _check_hit(self, allocated_bet: AllocatedBet, race: RaceData) -> int:
        """買い目が的中したかどうかを判定し、払い戻し金額を返す。

        実際の払い戻し情報（PayoutInfo）が存在すれば、券種・組み合わせが一致する
        払い戻しを検索する。的中時は投資金額に対する払い戻し額を返す。

        Args:
            allocated_bet: 配分済み買い目
            race: レースデータ（結果含む）

        Returns:
            払い戻し金額（円）。不的中の場合は0。
        """
        if race.payouts is None or race.results is None:
            return 0

        bet = allocated_bet.recommendation
        bet_type = bet.bet_type
        bet_combination = bet.combination

        for payout_info in race.payouts:
            if payout_info.bet_type != bet_type:
                continue

            if self._combination_matches(bet_type, bet_combination, payout_info):
                # 払い戻し金額は100円あたりの金額として記録されている
                # 実際の払い戻し = (投資金額 / 100) * 払い戻し金額
                return (allocated_bet.amount // 100) * payout_info.payout

        return 0

    def _combination_matches(
        self, bet_type: BetType, bet_combination: tuple[int, ...], payout: PayoutInfo
    ) -> bool:
        """買い目の組み合わせが払い戻し情報と一致するか判定する。

        券種に応じたマッチングルール:
        - 単勝/複勝: 馬番が一致
        - 馬連/ワイド/三連複: 馬番のセットが一致（順不同）
        - 馬単/三連単: 馬番の順序も一致

        Args:
            bet_type: 券種
            bet_combination: 買い目の馬番組み合わせ
            payout: 払い戻し情報

        Returns:
            一致する場合True
        """
        payout_combination = payout.combination

        if bet_type in (BetType.WIN, BetType.PLACE):
            # 単勝・複勝: 馬番が一致
            return bet_combination == payout_combination

        elif bet_type in (BetType.QUINELLA, BetType.WIDE, BetType.TRIO):
            # 馬連・ワイド・三連複: 順不同で馬番セットが一致
            return set(bet_combination) == set(payout_combination)

        elif bet_type in (BetType.EXACTA, BetType.TRIFECTA):
            # 馬単・三連単: 順序も含めて一致
            return bet_combination == payout_combination

        return False

    def _calculate_daily_returns(
        self, daily_records: dict[date, _DailyRecord]
    ) -> list[float]:
        """日次リターン（回収率）のリストを算出する。

        Args:
            daily_records: 日付ごとの投資/払い戻し記録

        Returns:
            日次リターン率のリスト（投資がある日のみ）
        """
        if not daily_records:
            return []

        sorted_dates = sorted(daily_records.keys())
        returns = []
        for d in sorted_dates:
            record = daily_records[d]
            if record.investment > 0:
                returns.append(record.returns / record.investment)
            else:
                returns.append(0.0)

        return returns

    def _aggregate_to_weekly(
        self, daily_records: dict[date, _DailyRecord]
    ) -> list[float]:
        """日次記録を週次リターンに集約する。

        ISO weekを使用して週単位に集約。

        Args:
            daily_records: 日付ごとの投資/払い戻し記録

        Returns:
            週次リターン率のリスト
        """
        if not daily_records:
            return []

        weekly: dict[tuple[int, int], _DailyRecord] = defaultdict(_DailyRecord)

        for d, record in daily_records.items():
            iso_year, iso_week, _ = d.isocalendar()
            key = (iso_year, iso_week)
            weekly[key].investment += record.investment
            weekly[key].returns += record.returns

        sorted_keys = sorted(weekly.keys())
        returns = []
        for key in sorted_keys:
            record = weekly[key]
            if record.investment > 0:
                returns.append(record.returns / record.investment)
            else:
                returns.append(0.0)

        return returns

    def _aggregate_to_monthly(
        self, daily_records: dict[date, _DailyRecord]
    ) -> list[float]:
        """日次記録を月次リターンに集約する。

        Args:
            daily_records: 日付ごとの投資/払い戻し記録

        Returns:
            月次リターン率のリスト
        """
        if not daily_records:
            return []

        monthly: dict[tuple[int, int], _DailyRecord] = defaultdict(_DailyRecord)

        for d, record in daily_records.items():
            key = (d.year, d.month)
            monthly[key].investment += record.investment
            monthly[key].returns += record.returns

        sorted_keys = sorted(monthly.keys())
        returns = []
        for key in sorted_keys:
            record = monthly[key]
            if record.investment > 0:
                returns.append(record.returns / record.investment)
            else:
                returns.append(0.0)

        return returns

    def _calculate_max_drawdown(self, daily_returns: list[float]) -> float:
        """最大ドローダウンを算出する。

        累積リターンの最高値からの最大下落幅を計算する。

        Args:
            daily_returns: 日次リターン率のリスト

        Returns:
            最大ドローダウン（0〜1の正の値、下落幅）
        """
        if not daily_returns:
            return 0.0

        # 累積リターンの計算（1ベース）
        cumulative = []
        value = 1.0
        for r in daily_returns:
            value *= r  # r は回収率（1.0 = 100%）
            cumulative.append(value)

        if not cumulative:
            return 0.0

        # 最大ドローダウン計算
        peak = cumulative[0]
        max_dd = 0.0

        for val in cumulative:
            if val > peak:
                peak = val
            drawdown = (peak - val) / peak if peak > 0 else 0.0
            max_dd = max(max_dd, drawdown)

        return max_dd

    def _calculate_sharpe_ratio(self, daily_returns: list[float]) -> float:
        """シャープレシオを算出する。

        日次リターンの平均と標準偏差から計算する。
        リスクフリーレートは0%と仮定する。

        Args:
            daily_returns: 日次リターン率のリスト

        Returns:
            シャープレシオ
        """
        if len(daily_returns) < 2:
            return 0.0

        # リターン率を超過リターンに変換（1.0 = break even → 0.0 excess）
        excess_returns = [r - 1.0 for r in daily_returns]

        mean_return = np.mean(excess_returns)
        std_return = np.std(excess_returns, ddof=1)

        if std_return == 0:
            return 0.0

        return float(mean_return / std_return)

    def _generate_recommendations(self, result: BacktestResult) -> list[str]:
        """回収率が100%未満の場合のパラメータ調整推奨事項を生成する。

        Args:
            result: バックテスト結果

        Returns:
            推奨事項のリスト
        """
        recommendations: list[str] = []

        if result.return_rate >= 1.0:
            return recommendations

        recommendations.append(
            f"回収率が{result.return_rate:.1%}で100%未満です。以下の調整を検討してください:"
        )

        # 的中率が低い場合
        if result.hit_rate < 0.1:
            recommendations.append(
                "的中率が低い: 信頼度スコア閾値を引き上げて、"
                "より確実性の高いレースに絞ることを推奨します。"
            )

        # 券種別に成績が悪いものを特定
        worst_bet_types: list[str] = []
        for bt, stats in result.bet_type_stats.items():
            if stats.get("bets", 0) > 0 and stats.get("return_rate", 0) < 0.5:
                worst_bet_types.append(bt.value)

        if worst_bet_types:
            recommendations.append(
                f"回収率の低い券種({', '.join(worst_bet_types)})を"
                "対象から除外することを検討してください。"
            )

        # 最大ドローダウンが大きい場合
        if result.max_drawdown > 0.5:
            recommendations.append(
                f"最大ドローダウンが{result.max_drawdown:.1%}と大きい: "
                "資金配分の集中度を下げるか、1レースあたりの予算上限を"
                "引き下げることを推奨します。"
            )

        # 見送りレース数が少ない場合
        if result.total_races > 0:
            skip_ratio = result.skipped_races / result.total_races
            if skip_ratio < 0.3:
                recommendations.append(
                    "見送りレース数が少ない: 信頼度スコア閾値を引き上げて、"
                    "より厳選したレースのみ投資することを推奨します。"
                )

        return recommendations

    def _detect_overfitting(
        self, train_return_rate: float, validation_return_rate: float
    ) -> bool:
        """過学習判定を行う。

        学習用データの回収率が検証用データの回収率を大幅に上回る場合、
        過学習と判定する。

        基準: 学習用回収率が検証用回収率の1.5倍以上、かつ差が20%ポイント以上

        Args:
            train_return_rate: 学習用データでの回収率
            validation_return_rate: 検証用データでの回収率

        Returns:
            過学習と判定される場合True
        """
        if train_return_rate <= 0 or validation_return_rate <= 0:
            return False

        rate_ratio = train_return_rate / validation_return_rate if validation_return_rate > 0 else float('inf')
        rate_diff = train_return_rate - validation_return_rate

        # 学習データの回収率が検証データの1.5倍以上、かつ差が20%ポイント以上
        return rate_ratio >= 1.5 and rate_diff >= 0.2
