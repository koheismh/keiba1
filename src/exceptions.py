"""競馬予測システムの例外クラス階層。

このモジュールはシステム全体で使用される例外クラスを定義する。
すべてのカスタム例外は基底クラス HorseRacePredictorError を継承する。
"""

from typing import Any


class HorseRacePredictorError(Exception):
    """基底例外クラス。

    競馬予測システムで発生するすべてのカスタム例外の基底クラス。
    """

    pass


class DataFetchError(HorseRacePredictorError):
    """データ取得エラー。

    外部ソースからのデータ取得時に発生するエラー。
    リトライ（最大3回）後も失敗した場合に送出される。
    """

    def __init__(self, race_id: str, message: str) -> None:
        self.race_id = race_id
        super().__init__(f"Race {race_id}: {message}")


class DataValidationError(HorseRacePredictorError):
    """データバリデーションエラー。

    データのフォーマット不正や必須フィールド欠損時に発生するエラー。
    """

    def __init__(self, field: str, reason: str) -> None:
        self.field = field
        super().__init__(f"Validation failed for {field}: {reason}")


class ConfigError(HorseRacePredictorError):
    """設定エラー。

    設定値が有効範囲外の場合に発生するエラー。
    エラーメッセージには無効な設定キー、値、および有効範囲が含まれる。
    """

    def __init__(self, key: str, value: Any, valid_range: str) -> None:
        self.key = key
        self.value = value
        self.valid_range = valid_range
        super().__init__(f"Invalid config {key}={value}. Valid range: {valid_range}")


class ModelError(HorseRacePredictorError):
    """モデルエラー。

    予測モデルの読み込み、学習、推論時に発生するエラー。
    モデルファイル不在や予測確率異常などが含まれる。
    """

    pass
