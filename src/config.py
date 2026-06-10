"""設定管理モジュール。

YAMLファイルから設定を読み込み、バリデーションを行い、
Configデータクラスに変換する。
"""

from pathlib import Path
from typing import Any

import yaml

from src.data.models import BetType, Config
from src.exceptions import ConfigError


class ConfigManager:
    """設定管理クラス。

    YAML設定ファイルの読み込み、バリデーション、Configデータクラスへの変換を行う。
    """

    # 有効なBetType名のセット
    _VALID_BET_TYPE_NAMES: set[str] = {bt.name for bt in BetType}

    def load(self, path: Path) -> Config:
        """設定ファイルを読み込み、バリデーション後にConfigデータクラスを返す。

        Args:
            path: YAML設定ファイルのパス

        Returns:
            Config: バリデーション済みの設定データクラス

        Raises:
            FileNotFoundError: 設定ファイルが存在しない場合
            ConfigError: 設定値が有効範囲外の場合
        """
        if not path.exists():
            raise FileNotFoundError(f"設定ファイルが見つかりません: {path}")

        with open(path, "r", encoding="utf-8") as f:
            config_dict = yaml.safe_load(f)

        if config_dict is None:
            config_dict = {}

        self.validate(config_dict)
        return self._to_config(config_dict)

    def validate(self, config_dict: dict[str, Any]) -> None:
        """設定値のバリデーションを行う。

        すべての設定値が有効範囲内であることを確認する。
        有効範囲外の値が見つかった場合、ConfigErrorを送出する。

        Args:
            config_dict: YAML読み込み後の設定辞書

        Raises:
            ConfigError: 設定値が有効範囲外の場合
        """
        prediction = config_dict.get("prediction", {})
        budget = config_dict.get("budget", {})
        bet_types = config_dict.get("bet_types", [])

        # confidence_threshold: 0-100 (int)
        if "confidence_threshold" in prediction:
            value = prediction["confidence_threshold"]
            if not isinstance(value, int) or value < 0 or value > 100:
                raise ConfigError(
                    "prediction.confidence_threshold",
                    value,
                    "0 <= confidence_threshold <= 100 (int)",
                )

        # max_bets_per_race: 1+ (int)
        if "max_bets_per_race" in prediction:
            value = prediction["max_bets_per_race"]
            if not isinstance(value, int) or value < 1:
                raise ConfigError(
                    "prediction.max_bets_per_race",
                    value,
                    "max_bets_per_race >= 1 (int)",
                )

        # min_expected_value: 0+ (float)
        if "min_expected_value" in prediction:
            value = prediction["min_expected_value"]
            if not isinstance(value, (int, float)) or value < 0:
                raise ConfigError(
                    "prediction.min_expected_value",
                    value,
                    "min_expected_value >= 0 (float)",
                )

        # daily_budget: 1+ (int)
        if "daily_budget" in budget:
            value = budget["daily_budget"]
            if not isinstance(value, int) or value < 1:
                raise ConfigError(
                    "budget.daily_budget",
                    value,
                    "daily_budget >= 1 (int)",
                )

        # max_single_bet_ratio: 0.0 < x <= 1.0 (float)
        if "max_single_bet_ratio" in budget:
            value = budget["max_single_bet_ratio"]
            if not isinstance(value, (int, float)) or value <= 0.0 or value > 1.0:
                raise ConfigError(
                    "budget.max_single_bet_ratio",
                    value,
                    "0.0 < max_single_bet_ratio <= 1.0 (float)",
                )

        # bet_types: valid BetType enum values
        if bet_types:
            for bt in bet_types:
                if bt not in self._VALID_BET_TYPE_NAMES:
                    raise ConfigError(
                        "bet_types",
                        bt,
                        f"有効な券種: {sorted(self._VALID_BET_TYPE_NAMES)}",
                    )

    def _to_config(self, config_dict: dict[str, Any]) -> Config:
        """バリデーション済み辞書をConfigデータクラスに変換する。

        Args:
            config_dict: バリデーション済みの設定辞書

        Returns:
            Config: 設定データクラス
        """
        prediction = config_dict.get("prediction", {})
        budget = config_dict.get("budget", {})
        bet_types_raw = config_dict.get("bet_types", [])

        # BetType名からenumに変換
        target_bet_types = [BetType[name] for name in bet_types_raw] if bet_types_raw else list(BetType)

        return Config(
            confidence_threshold=prediction.get("confidence_threshold", 50),
            max_bets_per_race=prediction.get("max_bets_per_race", 10),
            min_expected_value=float(prediction.get("min_expected_value", 1.0)),
            daily_budget=budget.get("daily_budget", 10000),
            max_single_bet_ratio=float(budget.get("max_single_bet_ratio", 0.3)),
            target_bet_types=target_bet_types,
        )
