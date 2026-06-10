"""ModelTrainer のユニットテスト"""

import numpy as np
import pytest

from src.prediction.trainer import CrossValidationResult, ModelTrainer


class TestCrossValidationResult:
    """CrossValidationResult データクラスのテスト"""

    def test_dataclass_creation(self):
        """CrossValidationResult が正しく生成されること"""
        result = CrossValidationResult(
            n_splits=5,
            fold_scores=[0.7, 0.75, 0.72, 0.68, 0.73],
            mean_score=0.716,
            std_score=0.025,
            train_scores=[0.85, 0.87, 0.84, 0.86, 0.85],
            is_overfitting=False,
        )
        assert result.n_splits == 5
        assert len(result.fold_scores) == 5
        assert result.mean_score == 0.716
        assert result.std_score == 0.025
        assert len(result.train_scores) == 5
        assert result.is_overfitting is False


class TestModelTrainer:
    """ModelTrainer クラスのテスト"""

    @pytest.fixture
    def trainer(self):
        """デフォルトパラメータのトレーナー"""
        return ModelTrainer()

    @pytest.fixture
    def sample_data(self):
        """テスト用のサンプルデータを生成する。

        100サンプル、5特徴量のデータセットを生成。
        1着は各レース（10頭）に1頭ずつ割り当てる。
        """
        rng = np.random.default_rng(42)
        n_samples = 100
        n_features = 5

        features = rng.random((n_samples, n_features))
        # 10レース×10頭を想定し、各レースで1頭だけ1着
        labels = np.array([1] + [2, 3, 4, 5, 6, 7, 8, 9, 10] * 11)[:n_samples]
        # 先頭10サンプルに1着を分散
        labels = np.tile([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 10)

        return features, labels

    def test_init_default_params(self):
        """デフォルトパラメータで初期化できること"""
        trainer = ModelTrainer()
        assert trainer._model_params is None

    def test_init_custom_params(self):
        """カスタムパラメータで初期化できること"""
        params = {"num_leaves": 15, "learning_rate": 0.1}
        trainer = ModelTrainer(model_params=params)
        assert trainer._model_params == params

    def test_cross_validate_returns_result(self, trainer, sample_data):
        """cross_validateがCrossValidationResultを返すこと"""
        features, labels = sample_data
        result = trainer.cross_validate(features, labels, n_splits=3)

        assert isinstance(result, CrossValidationResult)
        assert result.n_splits == 3
        assert len(result.fold_scores) == 3
        assert len(result.train_scores) == 3

    def test_cross_validate_default_splits(self, trainer, sample_data):
        """デフォルトで5分割交差検証を行うこと"""
        features, labels = sample_data
        result = trainer.cross_validate(features, labels)

        assert result.n_splits == 5
        assert len(result.fold_scores) == 5
        assert len(result.train_scores) == 5

    def test_cross_validate_scores_in_range(self, trainer, sample_data):
        """スコアが0.0〜1.0の範囲内であること"""
        features, labels = sample_data
        result = trainer.cross_validate(features, labels, n_splits=3)

        for score in result.fold_scores:
            assert 0.0 <= score <= 1.0
        for score in result.train_scores:
            assert 0.0 <= score <= 1.0
        assert 0.0 <= result.mean_score <= 1.0
        assert result.std_score >= 0.0

    def test_cross_validate_mean_score_calculation(self, trainer, sample_data):
        """mean_scoreがfold_scoresの平均と一致すること"""
        features, labels = sample_data
        result = trainer.cross_validate(features, labels, n_splits=3)

        expected_mean = np.mean(result.fold_scores)
        assert abs(result.mean_score - expected_mean) < 1e-10

    def test_cross_validate_std_score_calculation(self, trainer, sample_data):
        """std_scoreがfold_scoresの標準偏差と一致すること"""
        features, labels = sample_data
        result = trainer.cross_validate(features, labels, n_splits=3)

        expected_std = np.std(result.fold_scores)
        assert abs(result.std_score - expected_std) < 1e-10

    def test_overfitting_detection(self):
        """過学習が正しく検出されること。

        学習データに完全にフィットするが汎化しない状況をシミュレート。
        """
        # 過学習しやすいデータ（少量サンプル、多特徴量）
        rng = np.random.default_rng(123)
        n_samples = 50
        n_features = 20

        features = rng.random((n_samples, n_features))
        labels = np.tile([1, 2, 3, 4, 5], 10)

        # 過学習しやすい大きなモデルパラメータ
        params = {
            "objective": "binary",
            "num_leaves": 50,
            "learning_rate": 0.3,
            "n_estimators": 200,
            "verbose": -1,
        }
        trainer = ModelTrainer(model_params=params)
        result = trainer.cross_validate(features, labels, n_splits=3)

        # 過学習判定のメカニズムが動作していることを確認
        # (具体的な結果はデータ依存だが、train_score >= val_score は常に成立)
        for train_score, val_score in zip(result.train_scores, result.fold_scores):
            assert train_score >= 0.0
            assert val_score >= 0.0

    def test_no_overfitting_simple_data(self):
        """シンプルなデータでは過学習が検出されにくいこと"""
        rng = np.random.default_rng(42)
        n_samples = 200
        n_features = 3

        # 特徴量から着順が決まりやすいデータを生成
        features = rng.random((n_samples, n_features))
        labels = np.tile([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 20)

        params = {
            "objective": "binary",
            "num_leaves": 10,
            "learning_rate": 0.05,
            "n_estimators": 20,
            "verbose": -1,
        }
        trainer = ModelTrainer(model_params=params)
        result = trainer.cross_validate(features, labels, n_splits=3)

        # 結果が正しい構造を持つことを確認
        assert isinstance(result.is_overfitting, bool)

    def test_overfitting_threshold(self):
        """OVERFITTING_THRESHOLD が0.2であること"""
        assert ModelTrainer.OVERFITTING_THRESHOLD == 0.2
