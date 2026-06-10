"""ResultRecorder の永続化機能テスト"""

from datetime import date

import pytest

from src.data.models import BetType
from src.output.result_recorder import BetRecord, ResultRecorder


@pytest.fixture
def sample_record() -> BetRecord:
    return BetRecord(
        date=date(2024, 1, 15),
        race_id="202401010101",
        bet_type=BetType.WIN,
        combination=(3,),
        amount=1000,
        is_hit=True,
        payout=3500,
    )


class TestPersistence:
    """永続化のテスト。"""

    def test_save_and_reload(self, tmp_path, sample_record: BetRecord) -> None:
        storage_path = tmp_path / "results.json"

        # 記録して保存
        recorder1 = ResultRecorder(storage_path=storage_path)
        recorder1.record_result(sample_record)
        assert len(recorder1.get_records()) == 1

        # 新しいインスタンスで読み込み
        recorder2 = ResultRecorder(storage_path=storage_path)
        records = recorder2.get_records()
        assert len(records) == 1
        assert records[0].race_id == "202401010101"
        assert records[0].bet_type == BetType.WIN
        assert records[0].combination == (3,)
        assert records[0].amount == 1000
        assert records[0].is_hit is True
        assert records[0].payout == 3500

    def test_multiple_records_persist(self, tmp_path) -> None:
        storage_path = tmp_path / "results.json"

        recorder1 = ResultRecorder(storage_path=storage_path)
        recorder1.record_result(
            BetRecord(
                date=date(2024, 1, 15),
                race_id="race1",
                bet_type=BetType.WIN,
                combination=(1,),
                amount=1000,
                is_hit=True,
                payout=2000,
            )
        )
        recorder1.record_result(
            BetRecord(
                date=date(2024, 1, 16),
                race_id="race2",
                bet_type=BetType.QUINELLA,
                combination=(1, 3),
                amount=500,
                is_hit=False,
                payout=0,
            )
        )

        recorder2 = ResultRecorder(storage_path=storage_path)
        assert len(recorder2.get_records()) == 2

    def test_empty_file_no_error(self, tmp_path) -> None:
        storage_path = tmp_path / "results.json"
        recorder = ResultRecorder(storage_path=storage_path)
        assert len(recorder.get_records()) == 0

    def test_file_created_on_first_record(
        self, tmp_path, sample_record: BetRecord
    ) -> None:
        storage_path = tmp_path / "results.json"
        assert not storage_path.exists()

        recorder = ResultRecorder(storage_path=storage_path)
        recorder.record_result(sample_record)
        assert storage_path.exists()
