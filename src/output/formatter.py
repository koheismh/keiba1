"""出力フォーマッタ"""

from dataclasses import dataclass

from src.data.models import AllocatedBet, RaceData, RaceEvaluation


@dataclass
class RacePrediction:
    """1レースの予測結果"""

    race: RaceData
    evaluation: RaceEvaluation
    allocated_bets: list[AllocatedBet]  # Empty if skipped


@dataclass
class DaySummary:
    """1日の全体サマリー"""

    total_races: int
    bet_races: int
    skipped_races: int
    total_investment: int
    expected_return_rate: float


class OutputFormatter:
    """予測結果を読みやすいテキスト形式にフォーマットする"""

    def format_predictions(self, predictions: list[RacePrediction]) -> str:
        """Format all race predictions for a day, sorted by race number/id.

        Output includes:
        - Each race: name, post_time, bet list or skip reason
        - Day summary at the end
        """
        if not predictions:
            return "予測対象レースがありません。"

        # Sort predictions by race_id
        sorted_predictions = sorted(predictions, key=lambda p: p.race.race_id)

        lines: list[str] = []
        lines.append("=" * 60)
        lines.append("競馬レース予測結果")
        lines.append("=" * 60)

        for prediction in sorted_predictions:
            lines.append("")
            lines.append(self.format_race(prediction))

        # Calculate and append summary
        summary = self._calculate_summary(sorted_predictions)
        lines.append("")
        lines.append(self.format_summary(summary))

        return "\n".join(lines)

    def format_race(self, prediction: RacePrediction) -> str:
        """Format a single race prediction."""
        lines: list[str] = []

        # Race header
        race = prediction.race
        post_time_str = race.post_time.strftime("%H:%M") if race.post_time else "未定"
        lines.append("-" * 60)
        lines.append(f"【{race.race_name}】 発走時刻: {post_time_str}")
        lines.append(f"  {race.venue} {race.course_type}{race.distance}m "
                     f"馬場: {race.track_condition.value}")
        lines.append("-" * 60)

        evaluation = prediction.evaluation

        if evaluation.should_bet and prediction.allocated_bets:
            # Show bet list
            lines.append(f"  信頼度スコア: {evaluation.confidence_score}/100")
            lines.append(f"  買い目数: {len(prediction.allocated_bets)}点")
            lines.append("")
            lines.append(f"  {'券種':<8} {'馬番':<12} {'金額':>8} {'期待値':>6}")
            lines.append(f"  {'─' * 40}")

            for bet in prediction.allocated_bets:
                rec = bet.recommendation
                bet_type_name = rec.bet_type.value
                combination_str = "-".join(str(n) for n in rec.combination)
                amount_str = f"¥{bet.amount:,}"
                ev_str = f"{rec.expected_value:.2f}"
                lines.append(
                    f"  {bet_type_name:<8} {combination_str:<12} "
                    f"{amount_str:>8} {ev_str:>6}"
                )

            total_amount = sum(b.amount for b in prediction.allocated_bets)
            lines.append(f"  {'─' * 40}")
            lines.append(f"  合計投資金額: ¥{total_amount:,}")
        else:
            # Show skip reason
            lines.append(f"  ▶ 見送り")
            lines.append(f"  信頼度スコア: {evaluation.confidence_score}/100")
            if evaluation.skip_reason:
                lines.append(f"  見送り理由: {evaluation.skip_reason}")

        return "\n".join(lines)

    def format_summary(self, summary: DaySummary) -> str:
        """Format the day summary."""
        lines: list[str] = []
        lines.append("=" * 60)
        lines.append("【1日のサマリー】")
        lines.append("=" * 60)
        lines.append(f"  全レース数:       {summary.total_races}")
        lines.append(f"  投資対象レース数: {summary.bet_races}")
        lines.append(f"  見送りレース数:   {summary.skipped_races}")
        lines.append(f"  合計投資金額:     ¥{summary.total_investment:,}")
        lines.append(f"  期待回収率:       {summary.expected_return_rate:.1f}%")
        lines.append("=" * 60)
        return "\n".join(lines)

    def _calculate_summary(
        self, predictions: list[RacePrediction]
    ) -> DaySummary:
        """Calculate the day summary from predictions."""
        total_races = len(predictions)
        bet_races = sum(
            1 for p in predictions
            if p.evaluation.should_bet and p.allocated_bets
        )
        skipped_races = total_races - bet_races

        total_investment = sum(
            sum(b.amount for b in p.allocated_bets)
            for p in predictions
            if p.evaluation.should_bet and p.allocated_bets
        )

        # Calculate expected return rate
        if total_investment > 0:
            total_expected_return = sum(
                sum(
                    b.amount * b.recommendation.expected_value
                    for b in p.allocated_bets
                )
                for p in predictions
                if p.evaluation.should_bet and p.allocated_bets
            )
            expected_return_rate = (total_expected_return / total_investment) * 100
        else:
            expected_return_rate = 0.0

        return DaySummary(
            total_races=total_races,
            bet_races=bet_races,
            skipped_races=skipped_races,
            total_investment=total_investment,
            expected_return_rate=expected_return_rate,
        )
