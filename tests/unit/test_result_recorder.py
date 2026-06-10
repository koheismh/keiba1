"""ResultRecorder のユニットテスト"""

from datetime import date

import pytest

from src.data.models import BetType
from src.output.result_recorder import BetRecord, ResultRecorder


@pytest.fixture
def recorder(tmp_path) -> ResultRecorder:
    """空のResultRecorderを返す。"""
    return ResultRecorder(storage_path=tmp_path / "test_results.json")


@pytest.fixture
def sample_records() -> list[BetRecord]:
    """テスト用のサンプルレコードを返す。"""
    return [
        BetRecord(
            date=date(2024, 1, 15),
            race_id="202401010101",
            bet_type=BetType.WIN,
            combination=(3,),
            amount=1000,
            is_hit=True,
            payout=3500,
        ),
        BetRecord(
            date=date(2024, 1, 15),
            race_id="202401010102",
            bet_type=BetType.QUINELLA,
            combination=(1, 5),
            amount=2000,
            is_hit=False,
            payout=0,
        ),
        BetRecord(
            date=date(2024, 1, 22),
            race_id="202401020101",
            bet_type=BetType.TRIFECTA,
            combination=(3, 1, 7),
            amount=500,
            is_hit=True,
            payout=15000,
        ),
        BetRecord(
            date=date(2024, 2, 5),
            race_id="202402010101",
            bet_type=BetType.WIN,
            combination=(5,),
            amount=1000,
            is_hit=False,
            payout=0,
        ),
    ]


class TestRecordResult:
    """record_result メソッドのテスト"""

    def test_record_single(self, recorder: ResultRecorder) -> None:
        record = BetRecord(
            date=date(2024, 1, 15),
            race_id="202401010101",
            bet_type=BetType.WIN,
            combination=(3,),
            amount=1000,
            is_hit=True,
            payout=3500,
        )
        recorder.record_result(record)
        assert len(recorder.get_records()) == 1
        assert recorder.get_records()[0] == record

    def test_record_multiple(
        self, recorder: ResultRecorder, sample_records: list[BetRecord]
    ) -> None:
        for record in sample_records:
            recorder.record_result(record)
        assert len(recorder.get_records()) == 4


class TestGetDailyReturnRate:
    """get_daily_return_rate メソッドのテスト"""

    def test_with_records(
        self, recorder: ResultRecorder, sample_records: list[BetRecord]
    ) -> None:
        for record in sample_records:
            recorder.record_result(record)
        # 2024-01-15: investment=3000, payout=3500, rate=3500/3000
        rate = recorder.get_daily_return_rate(date(2024, 1, 15))
        assert rate == pytest.approx(3500 / 3000)

    def test_no_records_for_date(self, recorder: ResultRecorder) -> None:
        rate = recorder.get_daily_return_rate(date(2024, 1, 1))
        assert rate == 0.0

    def test_all_miss_day(self, recorder: ResultRecorder) -> None:
        recorder.record_result(
            BetRecord(
                date=date(2024, 3, 1),
                race_id="R001",
                bet_type=BetType.WIN,
                combination=(1,),
                amount=1000,
                is_hit=False,
                payout=0,
            )
        )
        rate = recorder.get_daily_return_rate(date(2024, 3, 1))
        assert rate == 0.0


class TestGetWeeklyReturnRate:
    """get_weekly_return_rate メソッドのテスト"""

    def test_with_records(
        self, recorder: ResultRecorder, sample_records: list[BetRecord]
    ) -> None:
        for record in sample_records:
            recorder.record_result(record)
        # 2024-01-15 is ISO week 3 of 2024
        year, week, _ = date(2024, 1, 15).isocalendar()
        rate = recorder.get_weekly_return_rate(year, week)
        # investment=3000, payout=3500
        assert rate == pytest.approx(3500 / 3000)

    def test_no_records_for_week(self, recorder: ResultRecorder) -> None:
        rate = recorder.get_weekly_return_rate(2024, 52)
        assert rate == 0.0


class TestGetMonthlyReturnRate:
    """get_monthly_return_rate メソッドのテスト"""

    def test_with_records(
        self, recorder: ResultRecorder, sample_records: list[BetRecord]
    ) -> None:
        for record in sample_records:
            recorder.record_result(record)
        # January 2024: investment=3000+500=3500, payout=3500+15000=18500
        rate = recorder.get_monthly_return_rate(2024, 1)
        assert rate == pytest.approx(18500 / 3500)

    def test_no_records_for_month(self, recorder: ResultRecorder) -> None:
        rate = recorder.get_monthly_return_rate(2024, 12)
        assert rate == 0.0


class TestGetRecentReturnRate:
    """get_recent_return_rate メソッドのテスト"""

    def test_no_records(self, recorder: ResultRecorder) -> None:
        rate = recorder.get_recent_return_rate()
        assert rate == 0.0

    def test_all_within_30_days(self, recorder: ResultRecorder) -> None:
        # All records within same date
        recorder.record_result(
            BetRecord(
                date=date(2024, 3, 1),
                race_id="R001",
                bet_type=BetType.WIN,
                combination=(1,),
                amount=1000,
                is_hit=True,
                payout=2000,
            )
        )
        recorder.record_result(
            BetRecord(
                date=date(2024, 3, 10),
                race_id="R002",
                bet_type=BetType.WIN,
                combination=(2,),
                amount=1000,
                is_hit=False,
                payout=0,
            )
        )
        # latest=2024-03-10, cutoff=2024-02-10, both within range
        rate = recorder.get_recent_return_rate(days=30)
        assert rate == pytest.approx(2000 / 2000)

    def test_older_records_excluded(self, recorder: ResultRecorder) -> None:
        # Old record outside 30 days
        recorder.record_result(
            BetRecord(
                date=date(2024, 1, 1),
                race_id="R001",
                bet_type=BetType.WIN,
                combination=(1,),
                amount=5000,
                is_hit=True,
                payout=50000,
            )
        )
        # Recent record
        recorder.record_result(
            BetRecord(
                date=date(2024, 3, 10),
                race_id="R002",
                bet_type=BetType.WIN,
                combination=(2,),
                amount=1000,
                is_hit=False,
                payout=0,
            )
        )
        # latest=2024-03-10, cutoff=2024-02-10, only R002 in range
        rate = recorder.get_recent_return_rate(days=30)
        assert rate == 0.0

    def test_custom_days(self, recorder: ResultRecorder) -> None:
        recorder.record_result(
            BetRecord(
                date=date(2024, 3, 1),
                race_id="R001",
                bet_type=BetType.WIN,
                combination=(1,),
                amount=1000,
                is_hit=True,
                payout=1500,
            )
        )
        recorder.record_result(
            BetRecord(
                date=date(2024, 3, 10),
                race_id="R002",
                bet_type=BetType.WIN,
                combination=(2,),
                amount=1000,
                is_hit=False,
                payout=0,
            )
        )
        # 7 days: latest=2024-03-10, cutoff=2024-03-04. Only R002 in range.
        rate = recorder.get_recent_return_rate(days=7)
        assert rate == 0.0


class TestShouldRetrain:
    """should_retrain メソッドのテスト"""

    def test_no_records(self, recorder: ResultRecorder) -> None:
        assert recorder.should_retrain() is False

    def test_high_return_rate(self, recorder: ResultRecorder) -> None:
        # 回収率 > 80% → 再学習不要
        recorder.record_result(
            BetRecord(
                date=date(2024, 3, 1),
                race_id="R001",
                bet_type=BetType.WIN,
                combination=(1,),
                amount=1000,
                is_hit=True,
                payout=1500,
            )
        )
        assert recorder.should_retrain() is False

    def test_low_return_rate(self, recorder: ResultRecorder) -> None:
        # 回収率 < 80% → 再学習推奨
        recorder.record_result(
            BetRecord(
                date=date(2024, 3, 1),
                race_id="R001",
                bet_type=BetType.WIN,
                combination=(1,),
                amount=1000,
                is_hit=True,
                payout=500,
            )
        )
        assert recorder.should_retrain() is True

    def test_exactly_80_percent(self, recorder: ResultRecorder) -> None:
        # 回収率 = 80% → 再学習不要（80%未満が条件）
        recorder.record_result(
            BetRecord(
                date=date(2024, 3, 1),
                race_id="R001",
                bet_type=BetType.WIN,
                combination=(1,),
                amount=1000,
                is_hit=True,
                payout=800,
            )
        )
        assert recorder.should_retrain() is False


class TestGetRetrainNotification:
    """get_retrain_notification メソッドのテスト"""

    def test_no_notification_when_healthy(self, recorder: ResultRecorder) -> None:
        recorder.record_result(
            BetRecord(
                date=date(2024, 3, 1),
                race_id="R001",
                bet_type=BetType.WIN,
                combination=(1,),
                amount=1000,
                is_hit=True,
                payout=1500,
            )
        )
        assert recorder.get_retrain_notification() is None

    def test_notification_when_low(self, recorder: ResultRecorder) -> None:
        recorder.record_result(
            BetRecord(
                date=date(2024, 3, 1),
                race_id="R001",
                bet_type=BetType.WIN,
                combination=(1,),
                amount=1000,
                is_hit=True,
                payout=500,
            )
        )
        notification = recorder.get_retrain_notification()
        assert notification is not None
        assert "50.0%" in notification
        assert "再学習" in notification


class TestGenerateReport:
    """generate_report メソッドのテスト"""

    def test_empty_records(self, recorder: ResultRecorder) -> None:
        report = recorder.generate_report()
        assert "実績データがありません" in report

    def test_report_contains_summary(
        self, recorder: ResultRecorder, sample_records: list[BetRecord]
    ) -> None:
        for record in sample_records:
            recorder.record_result(record)
        report = recorder.generate_report()
        assert "全体サマリー" in report
        assert "総買い目数: 4" in report
        assert "的中数: 2" in report

    def test_report_contains_bet_type_stats(
        self, recorder: ResultRecorder, sample_records: list[BetRecord]
    ) -> None:
        for record in sample_records:
            recorder.record_result(record)
        report = recorder.generate_report()
        assert "券種別成績" in report
        assert "単勝" in report

    def test_report_contains_recent_rate(
        self, recorder: ResultRecorder, sample_records: list[BetRecord]
    ) -> None:
        for record in sample_records:
            recorder.record_result(record)
        report = recorder.generate_report()
        assert "直近30日間" in report
