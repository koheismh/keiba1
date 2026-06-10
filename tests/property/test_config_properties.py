"""設定値バリデーションのプロパティベーステスト。

Feature: horse-race-predictor, Property 15: 設定値バリデーション

For any 有効範囲外の設定値（信頼度閾値 < 0 or > 100、最大買い目数 ≤ 0、
期待値最低基準 < 0、予算 ≤ 0など）に対して、バリデーションは失敗し
有効範囲を含むエラーメッセージを返すこと。

Validates: Requirements 10.6
"""

import string

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from src.config import ConfigManager
from src.exceptions import ConfigError


class TestConfidenceThresholdValidation:
    """Property 15: confidence_threshold のバリデーション。

    Feature: horse-race-predictor, Property 15: 設定値バリデーション
    Validates: Requirements 10.6
    """

    @settings(max_examples=100)
    @given(
        value=st.one_of(
            st.integers(max_value=-1),
            st.integers(min_value=101),
        )
    )
    def test_invalid_int_confidence_threshold_raises_config_error(self, value: int) -> None:
        """有効範囲外の整数値（< 0 or > 100）はConfigErrorを送出する。"""
        config_dict = {"prediction": {"confidence_threshold": value}}
        manager = ConfigManager()

        with pytest.raises(ConfigError) as exc_info:
            manager.validate(config_dict)

        assert exc_info.value.key == "prediction.confidence_threshold"
        assert exc_info.value.value == value
        assert exc_info.value.valid_range is not None
        assert len(exc_info.value.valid_range) > 0

    @settings(max_examples=100)
    @given(
        value=st.floats(allow_nan=False, allow_infinity=False)
    )
    def test_float_confidence_threshold_raises_config_error(self, value: float) -> None:
        """float型の値はConfigErrorを送出する（intのみ有効）。"""
        config_dict = {"prediction": {"confidence_threshold": value}}
        manager = ConfigManager()

        with pytest.raises(ConfigError) as exc_info:
            manager.validate(config_dict)

        assert exc_info.value.key == "prediction.confidence_threshold"
        assert exc_info.value.value == value
        assert exc_info.value.valid_range is not None
        assert len(exc_info.value.valid_range) > 0


class TestMaxBetsPerRaceValidation:
    """Property 15: max_bets_per_race のバリデーション。

    Feature: horse-race-predictor, Property 15: 設定値バリデーション
    Validates: Requirements 10.6
    """

    @settings(max_examples=100)
    @given(value=st.integers(max_value=0))
    def test_invalid_max_bets_per_race_raises_config_error(self, value: int) -> None:
        """max_bets_per_race ≤ 0 はConfigErrorを送出する。"""
        config_dict = {"prediction": {"max_bets_per_race": value}}
        manager = ConfigManager()

        with pytest.raises(ConfigError) as exc_info:
            manager.validate(config_dict)

        assert exc_info.value.key == "prediction.max_bets_per_race"
        assert exc_info.value.value == value
        assert exc_info.value.valid_range is not None
        assert len(exc_info.value.valid_range) > 0


class TestMinExpectedValueValidation:
    """Property 15: min_expected_value のバリデーション。

    Feature: horse-race-predictor, Property 15: 設定値バリデーション
    Validates: Requirements 10.6
    """

    @settings(max_examples=100)
    @given(
        value=st.one_of(
            st.floats(max_value=-0.01, allow_nan=False, allow_infinity=False),
            st.integers(max_value=-1),
        )
    )
    def test_negative_min_expected_value_raises_config_error(self, value) -> None:
        """min_expected_value < 0 はConfigErrorを送出する。"""
        config_dict = {"prediction": {"min_expected_value": value}}
        manager = ConfigManager()

        with pytest.raises(ConfigError) as exc_info:
            manager.validate(config_dict)

        assert exc_info.value.key == "prediction.min_expected_value"
        assert exc_info.value.value == value
        assert exc_info.value.valid_range is not None
        assert len(exc_info.value.valid_range) > 0


class TestDailyBudgetValidation:
    """Property 15: daily_budget のバリデーション。

    Feature: horse-race-predictor, Property 15: 設定値バリデーション
    Validates: Requirements 10.6
    """

    @settings(max_examples=100)
    @given(value=st.integers(max_value=0))
    def test_invalid_daily_budget_raises_config_error(self, value: int) -> None:
        """daily_budget ≤ 0 はConfigErrorを送出する。"""
        config_dict = {"budget": {"daily_budget": value}}
        manager = ConfigManager()

        with pytest.raises(ConfigError) as exc_info:
            manager.validate(config_dict)

        assert exc_info.value.key == "budget.daily_budget"
        assert exc_info.value.value == value
        assert exc_info.value.valid_range is not None
        assert len(exc_info.value.valid_range) > 0


class TestMaxSingleBetRatioValidation:
    """Property 15: max_single_bet_ratio のバリデーション。

    Feature: horse-race-predictor, Property 15: 設定値バリデーション
    Validates: Requirements 10.6
    """

    @settings(max_examples=100)
    @given(
        value=st.one_of(
            st.floats(max_value=0.0, allow_nan=False, allow_infinity=False),
            st.floats(min_value=1.01, allow_nan=False, allow_infinity=False),
        )
    )
    def test_invalid_max_single_bet_ratio_raises_config_error(self, value: float) -> None:
        """max_single_bet_ratio ≤ 0.0 or > 1.0 はConfigErrorを送出する。"""
        config_dict = {"budget": {"max_single_bet_ratio": value}}
        manager = ConfigManager()

        with pytest.raises(ConfigError) as exc_info:
            manager.validate(config_dict)

        assert exc_info.value.key == "budget.max_single_bet_ratio"
        assert exc_info.value.value == value
        assert exc_info.value.valid_range is not None
        assert len(exc_info.value.valid_range) > 0


class TestBetTypeValidation:
    """Property 15: bet_types のバリデーション。

    Feature: horse-race-predictor, Property 15: 設定値バリデーション
    Validates: Requirements 10.6
    """

    # 有効なBetType名
    _VALID_BET_TYPE_NAMES = {"WIN", "PLACE", "QUINELLA", "EXACTA", "WIDE", "TRIO", "TRIFECTA"}

    @settings(max_examples=100)
    @given(
        value=st.text(
            alphabet=string.ascii_letters + string.digits + "_",
            min_size=1,
            max_size=20,
        ).filter(lambda x: x not in {"WIN", "PLACE", "QUINELLA", "EXACTA", "WIDE", "TRIO", "TRIFECTA"})
    )
    def test_invalid_bet_type_raises_config_error(self, value: str) -> None:
        """有効なenum名以外の文字列はConfigErrorを送出する。"""
        config_dict = {"bet_types": [value]}
        manager = ConfigManager()

        with pytest.raises(ConfigError) as exc_info:
            manager.validate(config_dict)

        assert exc_info.value.key == "bet_types"
        assert exc_info.value.value == value
        assert exc_info.value.valid_range is not None
        assert len(exc_info.value.valid_range) > 0
