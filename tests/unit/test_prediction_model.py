"""PredictionModel のユニットテスト"""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from src.exceptions import ModelError
from src.prediction.model import PredictionModel


@pytest.fixture
def sample_training_data():
    """学習用のサンプルデータを生成する。"""
    np.random.seed(42)
    n_samples = 200
    n_features = 5

    features = np.random.randn(n_samples, n_features)
    # 着順ラベル: 1着〜8着
    labels = np.random.randint(1, 9, size=n_samples)
    # 確実に1着のサンプルを含める
    labels[:25] = 1

    return features, labels


@pytest.fixture
def trained_model(sample_training_data):
    """学習済みのモデルを返す。"""
    features, labels = sample_training_data
    model = PredictionModel()
    model.train(features, labels)
    return model


class TestPredictionModelInit:
    """初期化に関するテスト"""

    def test_default_params(self):
        """デフォルトパラメータで初期化できること"""
        model = PredictionModel()
        assert model._params == PredictionModel.DEFAULT_PARAMS

    def test_custom_params(self):
        """カスタムパラメータで初期化できること"""
        custom_params = {
            "objective": "binary",
            "num_leaves": 63,
            "learning_rate": 0.1,
            "n_estimators": 200,
            "verbose": -1,
        }
        model = PredictionModel(params=custom_params)
        assert model._params == custom_params

    def test_model_is_none_initially(self):
        """初期状態ではモデルがNoneであること"""
        model = PredictionModel()
        assert model._model is None


class TestPredictionModelTrain:
    """train() メソッドに関するテスト"""

    def test_train_sets_model(self, sample_training_data):
        """学習後にモデルが設定されること"""
        features, labels = sample_training_data
        model = PredictionModel()
        model.train(features, labels)
        assert model._model is not None

    def test_train_with_all_same_labels(self):
        """全サンプルが同じ着順でも学習できること"""
        features = np.random.randn(50, 3)
        labels = np.ones(50)  # 全て1着
        model = PredictionModel()
        model.train(features, labels)
        assert model._model is not None


class TestPredictionModelPredict:
    """predict_probabilities() メソッドに関するテスト"""

    def test_predict_raises_without_training(self):
        """未学習時にModelErrorが送出されること"""
        model = PredictionModel()
        race_features = np.random.randn(8, 5)
        with pytest.raises(ModelError):
            model.predict_probabilities(race_features)

    def test_predict_returns_correct_shape(self, trained_model, sample_training_data):
        """予測結果のshapeが入力馬数と一致すること"""
        features, _ = sample_training_data
        n_horses = 8
        race_features = features[:n_horses]
        probs = trained_model.predict_probabilities(race_features)
        assert probs.shape == (n_horses,)

    def test_predict_probabilities_sum_to_one(self, trained_model, sample_training_data):
        """確率の合計が≈1.0であること"""
        features, _ = sample_training_data
        race_features = features[:10]
        probs = trained_model.predict_probabilities(race_features)
        assert abs(probs.sum() - 1.0) < 1e-6

    def test_predict_probabilities_in_range(self, trained_model, sample_training_data):
        """すべての確率が[0,1]範囲であること"""
        features, _ = sample_training_data
        race_features = features[:12]
        probs = trained_model.predict_probabilities(race_features)
        assert np.all(probs >= 0.0)
        assert np.all(probs <= 1.0)

    def test_predict_single_horse(self, trained_model, sample_training_data):
        """1頭のレースでも予測できること（確率=1.0）"""
        features, _ = sample_training_data
        race_features = features[:1]
        probs = trained_model.predict_probabilities(race_features)
        assert probs.shape == (1,)
        assert abs(probs[0] - 1.0) < 1e-6


class TestPredictionModelSaveLoad:
    """save() / load() メソッドに関するテスト"""

    def test_save_raises_without_training(self):
        """未学習時にsave()でModelErrorが送出されること"""
        model = PredictionModel()
        with tempfile.NamedTemporaryFile(suffix=".txt") as f:
            with pytest.raises(ModelError):
                model.save(Path(f.name))

    def test_load_raises_with_nonexistent_file(self):
        """存在しないファイルでload()時にModelErrorが送出されること"""
        model = PredictionModel()
        with pytest.raises(ModelError):
            model.load(Path("/nonexistent/path/model.txt"))

    def test_save_and_load_roundtrip(self, trained_model, sample_training_data):
        """保存→読み込みで予測結果が一致すること"""
        features, _ = sample_training_data
        race_features = features[:8]

        # 保存前の予測
        probs_before = trained_model.predict_probabilities(race_features)

        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "model.txt"
            trained_model.save(model_path)

            # 新しいモデルに読み込み
            loaded_model = PredictionModel()
            loaded_model.load(model_path)

            # 読み込み後の予測
            probs_after = loaded_model.predict_probabilities(race_features)

        np.testing.assert_array_almost_equal(probs_before, probs_after)

    def test_save_creates_parent_directories(self, trained_model):
        """save()が親ディレクトリを自動作成すること"""
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "subdir" / "nested" / "model.txt"
            trained_model.save(model_path)
            assert model_path.exists()
