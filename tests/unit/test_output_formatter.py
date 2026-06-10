"""OutputFormatter のユニットテスト"""

from datetime import date, time

import pytest

from src.data.models import (
    AllocatedBet,
    BetRecommendation,
    BetType,
    HorseEntry,
    RaceData,
    RaceEvaluation,
    TrackCondition,
)
from src.output.formatter import DaySummary, OutputFormatter, RacePrediction


@pytest.fixture
def formatter():
    return OutputFormatter()


@pytest.fixture
def sample_race():
    return RaceData(
        race_id="202401010101",
        race_name="東京1R",
        race_date=date(2024, 1, 1),
        post_time=time(10, 0),
        venue="東京",
        course_type="芝",
        distance=1600,
        track_condition=TrackCondition.FIRM,
        weather="晴",
        entries=[
            HorseEntry(
                horse_name="テスト馬A",
                jockey_name="テスト騎手A",
                gate_number=1,
                horse_number=1,
                weight=480,
                weight_change=-2,
                win_odds=3.5,
            ),
        ],
        results=None,
        payouts=None,
    )


@pytest.fixture
def sample_race_b():
    return RaceData(
        race_id="202401010102",
        race_name="東京2R",
        race_date=date(2024, 1, 1),
        post_time=time(10, 30),
        venue="東京",
        course_type="ダート",
        distance=1200,
        track_condition=TrackCondition.GOOD,
        weather="曇",
        entries=[],
        results=None,
        payouts=None,
    )


@pytest.fixture
def bet_evaluation():
    return RaceEvaluation(
        race_id="202401010101",
        confidence_score=75,
        should_bet=True,
        skip_reason=None,
        factors={"荒れやすさ": 0.3, "実力差": 0.7, "データ充実度": 0.8},
    )


@pytest.fixture
def skip_evaluation():
    return RaceEvaluation(
        race_id="202401010102",
        confidence_score=30,
        should_bet=False,
        skip_reason="信頼度スコアが閾値未満（荒れやすさが高い）",
        factors={"荒れやすさ": 0.9, "実力差": 0.2, "データ充実度": 0.4},
    )


@pytest.fixture
def sample_bets():
    return [
        AllocatedBet(
            recommendation=BetRecommendation(
                bet_type=BetType.WIN,
                combination=(3,),
                estimated_probability=0.35,
                estimated_odds=4.0,
                expected_value=1.4,
            ),
            amount=2000,
        ),
        AllocatedBet(
            recommendation=BetRecommendation(
                bet_type=BetType.QUINELLA,
                combination=(3, 5),
                estimated_probability=0.15,
                estimated_odds=12.0,
                expected_value=1.8,
            ),
            amount=1000,
        ),
    ]


class TestFormatRace:
    def test_format_race_with_bets(
        self, formatter, sample_race, bet_evaluation, sample_bets
    ):
        prediction = RacePrediction(
            race=sample_race,
            evaluation=bet_evaluation,
            allocated_bets=sample_bets,
        )
        result = formatter.format_race(prediction)

        assert "東京1R" in result
        assert "10:00" in result
        assert "信頼度スコア: 75/100" in result
        assert "単勝" in result
        assert "3" in result
        assert "馬連" in result
        assert "3-5" in result
        assert "¥2,000" in result
        assert "¥1,000" in result
        assert "1.40" in result
        assert "1.80" in result
        assert "¥3,000" in result  # total

    def test_format_race_skipped(
        self, formatter, sample_race_b, skip_evaluation
    ):
        prediction = RacePrediction(
            race=sample_race_b,
            evaluation=skip_evaluation,
            allocated_bets=[],
        )
        result = formatter.format_race(prediction)

        assert "東京2R" in result
        assert "10:30" in result
        assert "見送り" in result
        assert "信頼度スコア: 30/100" in result
        assert "信頼度スコアが閾値未満" in result

    def test_format_race_no_post_time(self, formatter, bet_evaluation, sample_bets):
        race = RaceData(
            race_id="202401010103",
            race_name="東京3R",
            race_date=date(2024, 1, 1),
            post_time=None,
            venue="東京",
            course_type="芝",
            distance=2000,
            track_condition=TrackCondition.YIELDING,
            weather=None,
            entries=[],
            results=None,
            payouts=None,
        )
        prediction = RacePrediction(
            race=race,
            evaluation=bet_evaluation,
            allocated_bets=sample_bets,
        )
        result = formatter.format_race(prediction)

        assert "未定" in result


class TestFormatSummary:
    def test_format_summary(self, formatter):
        summary = DaySummary(
            total_races=12,
            bet_races=5,
            skipped_races=7,
            total_investment=15000,
            expected_return_rate=125.3,
        )
        result = formatter.format_summary(summary)

        assert "全レース数:       12" in result
        assert "投資対象レース数: 5" in result
        assert "見送りレース数:   7" in result
        assert "¥15,000" in result
        assert "125.3%" in result

    def test_format_summary_no_investment(self, formatter):
        summary = DaySummary(
            total_races=6,
            bet_races=0,
            skipped_races=6,
            total_investment=0,
            expected_return_rate=0.0,
        )
        result = formatter.format_summary(summary)

        assert "投資対象レース数: 0" in result
        assert "見送りレース数:   6" in result
        assert "¥0" in result


class TestFormatPredictions:
    def test_empty_predictions(self, formatter):
        result = formatter.format_predictions([])
        assert "予測対象レースがありません" in result

    def test_predictions_sorted_by_race_id(
        self,
        formatter,
        sample_race,
        sample_race_b,
        bet_evaluation,
        skip_evaluation,
        sample_bets,
    ):
        # Put race_b (id ending 02) first, race (id ending 01) second
        predictions = [
            RacePrediction(
                race=sample_race_b,
                evaluation=skip_evaluation,
                allocated_bets=[],
            ),
            RacePrediction(
                race=sample_race,
                evaluation=bet_evaluation,
                allocated_bets=sample_bets,
            ),
        ]
        result = formatter.format_predictions(predictions)

        # Race 01 should appear before Race 02 in the output
        pos_race_a = result.find("東京1R")
        pos_race_b = result.find("東京2R")
        assert pos_race_a < pos_race_b

    def test_predictions_include_summary(
        self, formatter, sample_race, bet_evaluation, sample_bets
    ):
        predictions = [
            RacePrediction(
                race=sample_race,
                evaluation=bet_evaluation,
                allocated_bets=sample_bets,
            ),
        ]
        result = formatter.format_predictions(predictions)

        assert "1日のサマリー" in result
        assert "全レース数:       1" in result
        assert "投資対象レース数: 1" in result
        assert "見送りレース数:   0" in result

    def test_summary_calculations(
        self,
        formatter,
        sample_race,
        sample_race_b,
        bet_evaluation,
        skip_evaluation,
        sample_bets,
    ):
        predictions = [
            RacePrediction(
                race=sample_race,
                evaluation=bet_evaluation,
                allocated_bets=sample_bets,
            ),
            RacePrediction(
                race=sample_race_b,
                evaluation=skip_evaluation,
                allocated_bets=[],
            ),
        ]
        result = formatter.format_predictions(predictions)

        assert "全レース数:       2" in result
        assert "投資対象レース数: 1" in result
        assert "見送りレース数:   1" in result
        assert "¥3,000" in result  # total investment
