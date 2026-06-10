"""ConfigManagerのユニットテスト。"""

import tempfile
from pathlib import Path

import pytest
import yaml

from src.config import ConfigManager
from src.data.models import BetType, Config
from src.exceptions import ConfigError


@pytest.fixture
def config_manager() -> ConfigManager:
    return ConfigManager()


@pytest.fixture
def default_config_path() -> Path:
    return Path("config/default.yaml")


class TestConfigManagerLoad:
    """ConfigManager.load()のテスト"""

    def test_load_default_config(
        self, config_manager: ConfigManager, default_config_path: Path
    ) -> None:
        """デフォルト設定ファイルが正しく読み込めること"""
        config = config_manager.load(default_config_path)
        assert isinstance(config, Config)
        assert config.confidence_threshold == 50
        assert config.max_bets_per_race == 10
        assert config.min_expected_value == 1.0
        assert config.daily_budget == 10000
        assert config.max_single_bet_ratio == 0.3
        assert config.target_bet_types == list(BetType)

    def test_load_file_not_found(self, config_manager: ConfigManager) -> None:
        """存在しないファイルを読み込むとFileNotFoundErrorが発生すること"""
        with pytest.raises(FileNotFoundError):
            config_manager.load(Path("nonexistent.yaml"))

    def test_load_custom_values(self, config_manager: ConfigManager) -> None:
        """カスタム設定値が正しく読み込めること"""
        config_data = {
            "prediction": {
                "confidence_threshold": 70,
                "max_bets_per_race": 5,
                "min_expected_value": 1.5,
            },
            "budget": {
                "daily_budget": 20000,
                "max_single_bet_ratio": 0.2,
            },
            "bet_types": ["WIN", "PLACE"],
        }
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            yaml.dump(config_data, f)
            tmp_path = Path(f.name)

        try:
            config = config_manager.load(tmp_path)
            assert config.confidence_threshold == 70
            assert config.max_bets_per_race == 5
            assert config.min_expected_value == 1.5
            assert config.daily_budget == 20000
            assert config.max_single_bet_ratio == 0.2
            assert config.target_bet_types == [BetType.WIN, BetType.PLACE]
        finally:
            tmp_path.unlink()

    def test_load_empty_yaml(self, config_manager: ConfigManager) -> None:
        """空のYAMLファイルを読み込むとデフォルト値が使用されること"""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            f.write("")
            tmp_path = Path(f.name)

        try:
            config = config_manager.load(tmp_path)
            assert config.confidence_threshold == 50
            assert config.max_bets_per_race == 10
        finally:
            tmp_path.unlink()


class TestConfigManagerValidate:
    """ConfigManager.validate()のテスト"""

    def test_validate_valid_config(self, config_manager: ConfigManager) -> None:
        """有効な設定値はバリデーションを通過すること"""
        config_dict = {
            "prediction": {
                "confidence_threshold": 50,
                "max_bets_per_race": 10,
                "min_expected_value": 1.0,
            },
            "budget": {
                "daily_budget": 10000,
                "max_single_bet_ratio": 0.3,
            },
            "bet_types": ["WIN", "PLACE"],
        }
        # Should not raise
        config_manager.validate(config_dict)

    def test_validate_confidence_threshold_too_low(
        self, config_manager: ConfigManager
    ) -> None:
        """信頼度閾値が0未満のときConfigErrorが発生すること"""
        config_dict = {"prediction": {"confidence_threshold": -1}}
        with pytest.raises(ConfigError) as exc_info:
            config_manager.validate(config_dict)
        assert exc_info.value.key == "prediction.confidence_threshold"
        assert exc_info.value.value == -1

    def test_validate_confidence_threshold_too_high(
        self, config_manager: ConfigManager
    ) -> None:
        """信頼度閾値が100を超えるときConfigErrorが発生すること"""
        config_dict = {"prediction": {"confidence_threshold": 101}}
        with pytest.raises(ConfigError) as exc_info:
            config_manager.validate(config_dict)
        assert exc_info.value.key == "prediction.confidence_threshold"
        assert exc_info.value.value == 101

    def test_validate_confidence_threshold_not_int(
        self, config_manager: ConfigManager
    ) -> None:
        """信頼度閾値が整数でないときConfigErrorが発生すること"""
        config_dict = {"prediction": {"confidence_threshold": 50.5}}
        with pytest.raises(ConfigError) as exc_info:
            config_manager.validate(config_dict)
        assert exc_info.value.key == "prediction.confidence_threshold"

    def test_validate_max_bets_per_race_zero(
        self, config_manager: ConfigManager
    ) -> None:
        """最大買い目数が0のときConfigErrorが発生すること"""
        config_dict = {"prediction": {"max_bets_per_race": 0}}
        with pytest.raises(ConfigError) as exc_info:
            config_manager.validate(config_dict)
        assert exc_info.value.key == "prediction.max_bets_per_race"

    def test_validate_max_bets_per_race_negative(
        self, config_manager: ConfigManager
    ) -> None:
        """最大買い目数が負のときConfigErrorが発生すること"""
        config_dict = {"prediction": {"max_bets_per_race": -5}}
        with pytest.raises(ConfigError) as exc_info:
            config_manager.validate(config_dict)
        assert exc_info.value.key == "prediction.max_bets_per_race"

    def test_validate_min_expected_value_negative(
        self, config_manager: ConfigManager
    ) -> None:
        """期待値最低基準が負のときConfigErrorが発生すること"""
        config_dict = {"prediction": {"min_expected_value": -0.5}}
        with pytest.raises(ConfigError) as exc_info:
            config_manager.validate(config_dict)
        assert exc_info.value.key == "prediction.min_expected_value"

    def test_validate_daily_budget_zero(
        self, config_manager: ConfigManager
    ) -> None:
        """日次予算が0のときConfigErrorが発生すること"""
        config_dict = {"budget": {"daily_budget": 0}}
        with pytest.raises(ConfigError) as exc_info:
            config_manager.validate(config_dict)
        assert exc_info.value.key == "budget.daily_budget"

    def test_validate_max_single_bet_ratio_zero(
        self, config_manager: ConfigManager
    ) -> None:
        """単一買い目比率が0のときConfigErrorが発生すること"""
        config_dict = {"budget": {"max_single_bet_ratio": 0.0}}
        with pytest.raises(ConfigError) as exc_info:
            config_manager.validate(config_dict)
        assert exc_info.value.key == "budget.max_single_bet_ratio"

    def test_validate_max_single_bet_ratio_over_one(
        self, config_manager: ConfigManager
    ) -> None:
        """単一買い目比率が1.0を超えるときConfigErrorが発生すること"""
        config_dict = {"budget": {"max_single_bet_ratio": 1.5}}
        with pytest.raises(ConfigError) as exc_info:
            config_manager.validate(config_dict)
        assert exc_info.value.key == "budget.max_single_bet_ratio"

    def test_validate_invalid_bet_type(
        self, config_manager: ConfigManager
    ) -> None:
        """無効な券種名のときConfigErrorが発生すること"""
        config_dict = {"bet_types": ["WIN", "INVALID_TYPE"]}
        with pytest.raises(ConfigError) as exc_info:
            config_manager.validate(config_dict)
        assert exc_info.value.key == "bet_types"
        assert exc_info.value.value == "INVALID_TYPE"

    def test_validate_boundary_confidence_threshold_zero(
        self, config_manager: ConfigManager
    ) -> None:
        """信頼度閾値が0はバリデーションを通過すること"""
        config_dict = {"prediction": {"confidence_threshold": 0}}
        config_manager.validate(config_dict)

    def test_validate_boundary_confidence_threshold_hundred(
        self, config_manager: ConfigManager
    ) -> None:
        """信頼度閾値が100はバリデーションを通過すること"""
        config_dict = {"prediction": {"confidence_threshold": 100}}
        config_manager.validate(config_dict)

    def test_validate_boundary_max_single_bet_ratio_one(
        self, config_manager: ConfigManager
    ) -> None:
        """単一買い目比率が1.0はバリデーションを通過すること"""
        config_dict = {"budget": {"max_single_bet_ratio": 1.0}}
        config_manager.validate(config_dict)

    def test_validate_min_expected_value_zero(
        self, config_manager: ConfigManager
    ) -> None:
        """期待値最低基準が0はバリデーションを通過すること"""
        config_dict = {"prediction": {"min_expected_value": 0}}
        config_manager.validate(config_dict)


class TestConfigErrorMessage:
    """ConfigErrorのエラーメッセージが有効範囲を含むことのテスト"""

    def test_error_message_includes_key_and_value(
        self, config_manager: ConfigManager
    ) -> None:
        """エラーメッセージに無効なキーと値が含まれること"""
        config_dict = {"prediction": {"confidence_threshold": -1}}
        with pytest.raises(ConfigError) as exc_info:
            config_manager.validate(config_dict)
        error_msg = str(exc_info.value)
        assert "prediction.confidence_threshold" in error_msg
        assert "-1" in error_msg

    def test_error_message_includes_valid_range(
        self, config_manager: ConfigManager
    ) -> None:
        """エラーメッセージに有効範囲が含まれること"""
        config_dict = {"prediction": {"confidence_threshold": 200}}
        with pytest.raises(ConfigError) as exc_info:
            config_manager.validate(config_dict)
        assert "Valid range" in str(exc_info.value)
        assert exc_info.value.valid_range is not None
