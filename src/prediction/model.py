"""予測モデル - LightGBMベースの着順確率推定モデル"""

from pathlib import Path

import lightgbm as lgb
import numpy as np

from src.exceptions import ModelError


class PredictionModel:
    """LightGBMベースの着順確率推定モデル。

    LightGBMの二値分類器を使用して、各馬の勝利確率を推定する。
    predict_probabilities ではsoftmax正規化を行い、
    確率が[0,1]範囲かつ合計≈1.0となるように調整する。
    """

    DEFAULT_PARAMS: dict = {
        "objective": "binary",
        "num_leaves": 31,
        "learning_rate": 0.05,
        "n_estimators": 100,
        "verbose": -1,
    }

    def __init__(self, params: dict | None = None) -> None:
        """初期化。

        Args:
            params: LightGBMのパラメータ辞書。Noneの場合はデフォルトパラメータを使用。
        """
        self._params = params if params is not None else self.DEFAULT_PARAMS.copy()
        self._model: lgb.Booster | None = None

    def train(self, features: np.ndarray, labels: np.ndarray) -> None:
        """モデルを学習する。

        二値分類として学習する。ラベルは着順（1=1着, 2=2着, ...）を
        1着かどうか（1=勝利, 0=非勝利）に変換して使用する。

        Args:
            features: shape (n_samples, n_features) の特徴量配列
            labels: shape (n_samples,) の着順配列（1=1着, 2=2着, ...）
        """
        # 着順を二値ラベルに変換（1着=1, それ以外=0）
        binary_labels = (labels == 1).astype(int)

        # パラメータからn_estimatorsを取り出す（LightGBMのnum_boost_roundに対応）
        params = {k: v for k, v in self._params.items() if k != "n_estimators"}
        num_boost_round = self._params.get("n_estimators", 100)

        # LightGBMデータセットを作成
        train_data = lgb.Dataset(features, label=binary_labels)

        # モデルを学習
        self._model = lgb.train(
            params,
            train_data,
            num_boost_round=num_boost_round,
        )

    def predict_probabilities(self, race_features: np.ndarray) -> np.ndarray:
        """各馬の着順確率を推定する。

        生の予測値を取得し、softmax正規化を適用して
        確率が[0,1]範囲かつ合計≈1.0となるようにする。

        Args:
            race_features: shape (n_horses_in_race, n_features) の特徴量配列

        Returns:
            shape (n_horses,) の確率配列。各値は[0,1]範囲、合計≈1.0。

        Raises:
            ModelError: モデルが未学習の場合
        """
        if self._model is None:
            raise ModelError("モデルが学習されていません。先にtrain()を呼び出してください。")

        # 生の予測値（勝利確率）を取得
        raw_predictions = self._model.predict(race_features)

        # softmax正規化で確率分布に変換
        probabilities = self._softmax(raw_predictions)

        return probabilities

    def save(self, path: Path) -> None:
        """モデルをファイルに保存する。

        Args:
            path: 保存先のファイルパス

        Raises:
            ModelError: モデルが未学習の場合
        """
        if self._model is None:
            raise ModelError("モデルが学習されていません。保存するモデルがありません。")

        # 親ディレクトリを作成
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        self._model.save_model(str(path))

    def load(self, path: Path) -> None:
        """モデルをファイルから読み込む。

        Args:
            path: モデルファイルのパス

        Raises:
            ModelError: ファイルが存在しない場合
        """
        path = Path(path)
        if not path.exists():
            raise ModelError(f"モデルファイルが見つかりません: {path}")

        self._model = lgb.Booster(model_file=str(path))

    @staticmethod
    def _softmax(x: np.ndarray) -> np.ndarray:
        """数値安定性を考慮したsoftmax正規化。

        Args:
            x: 入力配列

        Returns:
            softmax正規化された確率配列
        """
        # 数値安定性のために最大値を引く
        exp_x = np.exp(x - np.max(x))
        return exp_x / exp_x.sum()
