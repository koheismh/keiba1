"""例外クラス階層のユニットテスト。"""

import pytest

from src.exceptions import (
    ConfigError,
    DataFetchError,
    DataValidationError,
    HorseRacePredictorError,
    ModelError,
)


class TestHorseRacePredictorError:
    """基底例外クラスのテスト。"""

    def test_inherits_from_exception(self) -> None:
        assert issubclass(HorseRacePredictorError, Exception)

    def test_can_be_raised_and_caught(self) -> None:
        with pytest.raises(HorseRacePredictorError):
            raise HorseRacePredictorError("test error")

    def test_message(self) -> None:
        e = HorseRacePredictorError("something went wrong")
        assert str(e) == "something went wrong"


class TestDataFetchError:
    """データ取得エラーのテスト。"""

    def test_inherits_from_base(self) -> None:
        assert issubclass(DataFetchError, HorseRacePredictorError)

    def test_attributes(self) -> None:
        e = DataFetchError(race_id="202401010101", message="connection timeout")
        assert e.race_id == "202401010101"

    def test_message_format(self) -> None:
        e = DataFetchError(race_id="202401010101", message="connection timeout")
        assert str(e) == "Race 202401010101: connection timeout"

    def test_caught_by_base_class(self) -> None:
        with pytest.raises(HorseRacePredictorError):
            raise DataFetchError("R001", "server error")


class TestDataValidationError:
    """データバリデーションエラーのテスト。"""

    def test_inherits_from_base(self) -> None:
        assert issubclass(DataValidationError, HorseRacePredictorError)

    def test_attributes(self) -> None:
        e = DataValidationError(field="horse_number", reason="must be positive")
        assert e.field == "horse_number"

    def test_message_format(self) -> None:
        e = DataValidationError(field="horse_number", reason="must be positive")
        assert str(e) == "Validation failed for horse_number: must be positive"

    def test_caught_by_base_class(self) -> None:
        with pytest.raises(HorseRacePredictorError):
            raise DataValidationError("weight", "negative value")


class TestConfigError:
    """設定エラーのテスト。"""

    def test_inherits_from_base(self) -> None:
        assert issubclass(ConfigError, HorseRacePredictorError)

    def test_attributes(self) -> None:
        e = ConfigError(key="confidence_threshold", value=150, valid_range="0-100")
        assert e.key == "confidence_threshold"
        assert e.value == 150
        assert e.valid_range == "0-100"

    def test_message_format(self) -> None:
        e = ConfigError(key="confidence_threshold", value=150, valid_range="0-100")
        assert str(e) == "Invalid config confidence_threshold=150. Valid range: 0-100"

    def test_various_value_types(self) -> None:
        e = ConfigError(key="daily_budget", value=-1000, valid_range="> 0")
        assert str(e) == "Invalid config daily_budget=-1000. Valid range: > 0"

        e = ConfigError(key="min_expected_value", value=-0.5, valid_range=">= 0")
        assert str(e) == "Invalid config min_expected_value=-0.5. Valid range: >= 0"

    def test_caught_by_base_class(self) -> None:
        with pytest.raises(HorseRacePredictorError):
            raise ConfigError("key", "val", "range")


class TestModelError:
    """モデルエラーのテスト。"""

    def test_inherits_from_base(self) -> None:
        assert issubclass(ModelError, HorseRacePredictorError)

    def test_can_be_raised_with_message(self) -> None:
        e = ModelError("model file not found")
        assert str(e) == "model file not found"

    def test_caught_by_base_class(self) -> None:
        with pytest.raises(HorseRacePredictorError):
            raise ModelError("prediction failed")
