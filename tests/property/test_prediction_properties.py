"""予測モデルのプロパティベーステスト。

Feature: horse-race-predictor, Property 4: 確率分布の妥当性
Feature: horse-race-predictor, Property 12: 交差検証の網羅性

Validates: Requirements 2.1, 2.5
"""

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from sklearn.model_selection import KFold

from src.prediction.model import PredictionModel

# --- Module-level fixture: モデルを一度だけ学習する ---

# 学習は計算コストが高いため、モジュールレベルでモデルを事前学習しておく
_NUM_FEATURES = 8  # 特徴量数（設計書に記載の8特徴量に対応）
_TRAINED_MODEL: PredictionModel | None = None


def _get_trained_model() -> PredictionModel:
    """学習済みモデルを返す（初回のみ学習を実行）。"""
    global _TRAINED_MODEL
    if _TRAINED_MODEL is None:
        rng = np.random.default_rng(42)

        # 学習データを生成（各レース10頭、100レース分）
        n_races = 100
        n_horses_per_race = 10
        n_samples = n_races * n_horses_per_race

        features = rng.standard_normal((n_samples, _NUM_FEATURES))
        # 各レースで1頭だけ1着（ラベル=1）、残りは非1着（ラベル=2以上）
        labels = np.zeros(n_samples, dtype=int)
        for i in range(n_races):
            start_idx = i * n_horses_per_race
            # ランダムに1頭を1着にする
            winner_offset = rng.integers(0, n_horses_per_race)
            labels[start_idx : start_idx + n_horses_per_race] = 2  # 非勝利
            labels[start_idx + winner_offset] = 1  # 勝利

        model = PredictionModel()
        model.train(features, labels)
        _TRAINED_MODEL = model

    return _TRAINED_MODEL


@pytest.fixture(scope="module")
def trained_model() -> PredictionModel:
    """学習済みPredictionModelのフィクスチャ（モジュールスコープ）。"""
    return _get_trained_model()


# --- Strategies ---


def race_features_strategy(
    n_features: int = _NUM_FEATURES,
) -> st.SearchStrategy[np.ndarray]:
    """ランダムなレース特徴量配列を生成するストラテジ。

    2〜18頭のレースに対して、各馬の特徴量ベクトルを生成する。
    """
    return st.integers(min_value=2, max_value=18).flatmap(
        lambda n_horses: st.builds(
            lambda data: np.array(data).reshape(n_horses, n_features),
            data=st.lists(
                st.floats(min_value=-5.0, max_value=5.0, allow_nan=False, allow_infinity=False),
                min_size=n_horses * n_features,
                max_size=n_horses * n_features,
            ),
        )
    )


# --- Property Tests ---


class TestProbabilityDistributionValidity:
    """Property 4: 確率分布の妥当性

    Feature: horse-race-predictor, Property 4: 確率分布の妥当性

    For any レースの特徴量入力に対して、予測モデルが出力する各馬の着順確率はすべて
    [0, 1]の範囲内であり、全馬の確率の合計は1.0に近似すること。

    Validates: Requirements 2.1
    """

    @settings(max_examples=100)
    @given(race_features=race_features_strategy())
    def test_all_probabilities_in_valid_range(
        self, race_features: np.ndarray, trained_model: PredictionModel
    ) -> None:
        """各馬の着順確率がすべて[0, 1]の範囲内であることを検証する。

        Validates: Requirements 2.1
        """
        probabilities = trained_model.predict_probabilities(race_features)

        # すべての確率が[0, 1]の範囲内であること
        assert np.all(probabilities >= 0.0), (
            f"確率に負の値が含まれている: min={probabilities.min()}, "
            f"values={probabilities}"
        )
        assert np.all(probabilities <= 1.0), (
            f"確率に1.0を超える値が含まれている: max={probabilities.max()}, "
            f"values={probabilities}"
        )

    @settings(max_examples=100)
    @given(race_features=race_features_strategy())
    def test_probabilities_sum_approximately_one(
        self, race_features: np.ndarray, trained_model: PredictionModel
    ) -> None:
        """全馬の確率の合計が1.0に近似（許容誤差1e-6以内）であることを検証する。

        Validates: Requirements 2.1
        """
        probabilities = trained_model.predict_probabilities(race_features)

        # 確率の合計が1.0に近似すること（許容誤差: 1e-6）
        prob_sum = probabilities.sum()
        assert abs(prob_sum - 1.0) < 1e-6, (
            f"確率の合計が1.0から離れている: sum={prob_sum}, "
            f"差分={abs(prob_sum - 1.0)}, "
            f"values={probabilities}"
        )

    @settings(max_examples=100)
    @given(race_features=race_features_strategy())
    def test_probability_count_matches_horse_count(
        self, race_features: np.ndarray, trained_model: PredictionModel
    ) -> None:
        """出力される確率の数がレースの出走馬数と一致することを検証する。

        Validates: Requirements 2.1
        """
        n_horses = race_features.shape[0]
        probabilities = trained_model.predict_probabilities(race_features)

        assert len(probabilities) == n_horses, (
            f"確率の数が出走馬数と不一致: "
            f"probabilities={len(probabilities)}, horses={n_horses}"
        )



# --- Property 12: 交差検証の網羅性 ---


class TestCrossValidationCoverage:
    """Property 12: 交差検証の網羅性

    Feature: horse-race-predictor, Property 12: 交差検証の網羅性

    For any データセットと分割数kに対して、交差検証の各フォールドにおいて各データポイントは
    ちょうど1回だけ検証データとして使用され、k-1回学習データとして使用されること。

    Validates: Requirements 2.5
    """

    @settings(max_examples=100)
    @given(
        dataset_size=st.integers(min_value=20, max_value=100),
        data=st.data(),
    )
    def test_each_data_point_used_exactly_once_as_validation(
        self, dataset_size: int, data: st.DataObject
    ) -> None:
        """各データポイントがちょうど1回だけ検証データとして使用されることを検証する。

        Union of all validation indices covers all data points exactly once.

        Validates: Requirements 2.5
        """
        n_splits = data.draw(
            st.integers(min_value=2, max_value=min(10, dataset_size)),
            label="n_splits",
        )

        X = np.arange(dataset_size)
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

        # 全フォールドの検証インデックスを収集
        all_val_indices: list[int] = []
        for _, val_index in kf.split(X):
            all_val_indices.extend(val_index.tolist())

        # 全データポイントがちょうど1回ずつ検証に使われることを確認
        all_val_sorted = sorted(all_val_indices)
        expected = list(range(dataset_size))

        assert all_val_sorted == expected, (
            f"検証データのカバレッジが不完全: "
            f"dataset_size={dataset_size}, n_splits={n_splits}, "
            f"validation_count={len(all_val_indices)}, expected={dataset_size}"
        )

    @settings(max_examples=100)
    @given(
        dataset_size=st.integers(min_value=20, max_value=100),
        data=st.data(),
    )
    def test_each_data_point_in_training_k_minus_1_times(
        self, dataset_size: int, data: st.DataObject
    ) -> None:
        """各データポイントがちょうど(k-1)回学習データとして使用されることを検証する。

        Each data point appears in exactly (n_splits - 1) training folds.

        Validates: Requirements 2.5
        """
        n_splits = data.draw(
            st.integers(min_value=2, max_value=min(10, dataset_size)),
            label="n_splits",
        )

        X = np.arange(dataset_size)
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

        # 各データポイントが学習データに含まれる回数を集計
        train_counts = np.zeros(dataset_size, dtype=int)
        for train_index, _ in kf.split(X):
            for idx in train_index:
                train_counts[idx] += 1

        # 各データポイントがちょうど(n_splits - 1)回学習データに使われること
        expected_count = n_splits - 1
        for i in range(dataset_size):
            assert train_counts[i] == expected_count, (
                f"データポイント{i}の学習使用回数が不正: "
                f"actual={train_counts[i]}, expected={expected_count}, "
                f"dataset_size={dataset_size}, n_splits={n_splits}"
            )
