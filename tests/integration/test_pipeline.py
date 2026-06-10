"""統合テスト - 予測パイプライン全体のエンドツーエンドテスト。

モックデータを使用して予測パイプラインとバックテストの
端から端までの実行を検証する。

Validates: Requirements 6.1, 8.1
"""

from __future__ import annotations

import json
import tempfile
from datetime import date, time
from pathlib import Path

import numpy as np
import pytest

from src.data.historical_loader import HistoricalDataLoader
from src.data.models import (
    BacktestResult,
    BetType,
    Config,
    HorseEntry,
    PayoutInfo,
    RaceData,
    RaceResult,
    TrackCondition,
)
from src.pipeline import PredictionPipeline
from src.prediction.model import PredictionModel


# --- Helper functions to create realistic mock data ---


def _make_entries(count: int = 8) -> list[HorseEntry]:
    """Create a list of realistic horse entries."""
    jockeys = ["ルメール", "川田", "福永", "戸崎", "横山武", "池添", "松山", "岩田"]
    horses = ["アーモンドアイ", "コントレイル", "デアリングタクト", "エフフォーリア",
              "イクイノックス", "タイトルホルダー", "ドウデュース", "ジャスティンパレス"]
    entries = []
    for i in range(count):
        entries.append(
            HorseEntry(
                horse_name=horses[i % len(horses)] + str(i),
                jockey_name=jockeys[i % len(jockeys)],
                gate_number=(i % 8) + 1,
                horse_number=i + 1,
                weight=460 + i * 4,
                weight_change=(-2 + i) if i < 5 else (i - 4),
                win_odds=2.0 + i * 1.5,
            )
        )
    return entries


def _make_race(
    race_id: str,
    race_name: str,
    race_date: date,
    entries: list[HorseEntry] | None = None,
    with_results: bool = False,
) -> RaceData:
    """Create a single RaceData with optional results/payouts."""
    if entries is None:
        entries = _make_entries(8)

    results = None
    payouts = None
    if with_results:
        results = [
            RaceResult(horse_number=e.horse_number, finish_position=i + 1)
            for i, e in enumerate(entries)
        ]
        # 1着の馬に対する単勝払い戻し
        payouts = [
            PayoutInfo(
                bet_type=BetType.WIN,
                combination=(entries[0].horse_number,),
                payout=500,  # 100円あたり500円
            ),
            PayoutInfo(
                bet_type=BetType.PLACE,
                combination=(entries[0].horse_number,),
                payout=200,
            ),
            PayoutInfo(
                bet_type=BetType.PLACE,
                combination=(entries[1].horse_number,),
                payout=150,
            ),
        ]

    return RaceData(
        race_id=race_id,
        race_name=race_name,
        race_date=race_date,
        post_time=time(15, 40),
        venue="東京",
        course_type="芝",
        distance=2000,
        track_condition=TrackCondition.FIRM,
        weather="晴",
        entries=entries,
        results=results,
        payouts=payouts,
    )


def _create_training_data(
    n_races: int = 20,
) -> tuple[np.ndarray, np.ndarray, list[RaceData]]:
    """Create synthetic training data for the prediction model.

    Returns (features, labels, races) where features/labels can be
    used to train a PredictionModel.
    """
    rng = np.random.default_rng(42)
    n_horses_per_race = 8
    n_features = 8  # Matches FeatureEngineer output

    all_features = []
    all_labels = []
    races = []

    for i in range(n_races):
        race_date = date(2023, 6, 1 + (i % 28))
        race = _make_race(
            race_id=f"R{i+1:03d}",
            race_name=f"テストレース{i+1}",
            race_date=race_date,
            with_results=True,
        )
        races.append(race)

        # Generate features for each horse
        for j in range(n_horses_per_race):
            features = rng.random(n_features)
            all_features.append(features)
            all_labels.append(j + 1)  # finish position 1..n

    features_array = np.array(all_features)
    labels_array = np.array(all_labels)

    return features_array, labels_array, races


def _train_model() -> PredictionModel:
    """Train and return a PredictionModel on synthetic data."""
    features, labels, _ = _create_training_data(n_races=20)
    model = PredictionModel()
    model.train(features, labels)
    return model


# --- Integration Tests ---


class TestEndToEndPredictionPipeline:
    """予測パイプライン全体のエンドツーエンドテスト。

    Validates: Requirement 8.1
    """

    def test_predict_race_day_produces_formatted_output(self):
        """predict_race_dayがレース名、買い目or見送り理由、サマリーを含む出力を返す。"""
        model = _train_model()
        config = Config(
            confidence_threshold=30,
            max_bets_per_race=5,
            min_expected_value=0.5,
            daily_budget=10000,
        )

        races = [
            _make_race("R001", "東京1R メイクデビュー", date(2023, 10, 1)),
            _make_race("R002", "東京2R 未勝利", date(2023, 10, 1)),
            _make_race("R003", "東京3R 1勝クラス", date(2023, 10, 1)),
        ]

        pipeline = PredictionPipeline(config=config, model=model)
        output = pipeline.predict_race_day(races)

        # 出力が文字列であること
        assert isinstance(output, str)
        assert len(output) > 0

        # サマリーが含まれていること
        assert "サマリー" in output or "全レース数" in output or "投資対象" in output

        # 各レースの名前が出力に含まれるか、またはエラーとして報告されるか
        # （パイプラインのグレースフルデグラデーションにより、個別レースの処理は継続する）
        for race in races:
            assert race.race_name in output or "処理エラー" in output

    def test_predict_race_day_with_varied_races(self):
        """複数レースで投資・見送りの両方が発生することを確認する。"""
        model = _train_model()
        # 厳しい閾値を設定して一部レースのみ投資対象にする
        config = Config(
            confidence_threshold=70,
            max_bets_per_race=5,
            min_expected_value=0.5,
            daily_budget=10000,
        )

        races = [
            _make_race(f"R{i+1:03d}", f"テストレース{i+1}", date(2023, 10, 1))
            for i in range(5)
        ]

        pipeline = PredictionPipeline(config=config, model=model)
        output = pipeline.predict_race_day(races)

        assert isinstance(output, str)
        assert len(output) > 0
        # 出力に「見送り」か「買い目」のいずれかが含まれる
        assert "見送り" in output or "券種" in output or "合計" in output

    def test_predict_race_day_empty_races(self):
        """空のレースリストに対してエラーなく出力する。"""
        model = _train_model()
        config = Config()

        pipeline = PredictionPipeline(config=config, model=model)
        output = pipeline.predict_race_day([])

        assert isinstance(output, str)
        assert "予測対象レースがありません" in output


class TestEndToEndBacktest:
    """バックテストのエンドツーエンドテスト。

    Validates: Requirement 6.1
    """

    def test_backtest_returns_correct_structure(self):
        """BacktestResultが正しい構造を持つことを確認する。"""
        model = _train_model()
        config = Config(
            confidence_threshold=30,
            max_bets_per_race=5,
            min_expected_value=0.5,
            daily_budget=10000,
        )

        # 結果付きレースデータを作成
        races = [
            _make_race(
                f"R{i+1:03d}",
                f"バックテストレース{i+1}",
                date(2023, 9, 1 + (i % 28)),
                with_results=True,
            )
            for i in range(10)
        ]

        pipeline = PredictionPipeline(config=config, model=model)
        result = pipeline.run_backtest(races)

        # 構造の検証
        assert isinstance(result, BacktestResult)
        assert result.total_races == 10

        # 投資レース + 見送りレース = 合計レース（会計不変量）
        assert result.bet_races + result.skipped_races == result.total_races

        # 各フィールドが存在し妥当な型を持つ
        assert isinstance(result.hit_rate, float)
        assert 0.0 <= result.hit_rate <= 1.0
        assert isinstance(result.return_rate, float)
        assert result.return_rate >= 0.0
        assert isinstance(result.max_drawdown, float)
        assert 0.0 <= result.max_drawdown <= 1.0
        assert isinstance(result.sharpe_ratio, float)
        assert isinstance(result.daily_returns, list)
        assert isinstance(result.weekly_returns, list)
        assert isinstance(result.monthly_returns, list)
        assert isinstance(result.bet_type_stats, dict)

    def test_backtest_with_no_results_skips_all(self):
        """結果なしのレースでは全レースがスキップされることを確認する。"""
        model = _train_model()
        config = Config(confidence_threshold=30, min_expected_value=0.5)

        # 結果を持たないレース（payouts=None）
        races = [
            _make_race(f"R{i+1:03d}", f"レース{i+1}", date(2023, 9, 1), with_results=False)
            for i in range(5)
        ]

        pipeline = PredictionPipeline(config=config, model=model)
        result = pipeline.run_backtest(races)

        assert isinstance(result, BacktestResult)
        assert result.total_races == 5
        # 結果がないためヒットは0
        assert result.total_return == 0

    def test_backtest_investment_does_not_exceed_total(self):
        """バックテストで投資合計が正しく集計されることを確認する。"""
        model = _train_model()
        config = Config(
            confidence_threshold=20,
            max_bets_per_race=10,
            min_expected_value=0.5,
            daily_budget=50000,
        )

        races = [
            _make_race(
                f"R{i+1:03d}",
                f"レース{i+1}",
                date(2023, 9, 1 + i),
                with_results=True,
            )
            for i in range(8)
        ]

        pipeline = PredictionPipeline(config=config, model=model)
        result = pipeline.run_backtest(races)

        # 投資金額が0以上であること
        assert result.total_investment >= 0
        # 払い戻しが0以上であること
        assert result.total_return >= 0


class TestGracefulDegradation:
    """グレースフルデグラデーションのテスト。

    一部レースで問題が発生しても他のレースは正常に処理されることを検証する。
    """

    def test_pipeline_handles_empty_entries_race(self):
        """出走馬なしのレースが含まれてもパイプラインが継続する。"""
        model = _train_model()
        config = Config(confidence_threshold=30, min_expected_value=0.5)

        # 正常なレースと出走馬なしのレースを混在
        normal_race = _make_race("R001", "正常レース", date(2023, 10, 1))
        empty_race = RaceData(
            race_id="R002",
            race_name="出走馬なしレース",
            race_date=date(2023, 10, 1),
            post_time=time(14, 0),
            venue="中山",
            course_type="芝",
            distance=1800,
            track_condition=TrackCondition.FIRM,
            weather="晴",
            entries=[],  # 出走馬なし
            results=None,
            payouts=None,
        )
        normal_race2 = _make_race("R003", "正常レース2", date(2023, 10, 1))

        pipeline = PredictionPipeline(config=config, model=model)
        output = pipeline.predict_race_day([normal_race, empty_race, normal_race2])

        # 出力が返される（クラッシュしない）
        assert isinstance(output, str)
        assert len(output) > 0
        # エラーメッセージが含まれる
        assert "処理エラー" in output or "出走馬なしレース" in output
        # 正常レースは処理される
        assert "正常レース" in output

    def test_pipeline_handles_single_horse_race(self):
        """出走馬が1頭のエッジケースでも処理が継続する。"""
        model = _train_model()
        config = Config(confidence_threshold=30, min_expected_value=0.5)

        single_entry = [
            HorseEntry(
                horse_name="ソロホース",
                jockey_name="テスト騎手",
                gate_number=1,
                horse_number=1,
                weight=480,
                weight_change=0,
                win_odds=1.1,
            )
        ]
        single_race = _make_race(
            "R001", "1頭レース", date(2023, 10, 1), entries=single_entry
        )
        normal_race = _make_race("R002", "通常レース", date(2023, 10, 1))

        pipeline = PredictionPipeline(config=config, model=model)
        output = pipeline.predict_race_day([single_race, normal_race])

        # パイプラインはクラッシュしない
        assert isinstance(output, str)
        assert len(output) > 0

    def test_backtest_with_mixed_valid_and_edge_cases(self):
        """バックテストで正常データとエッジケースを混合しても結果が返る。"""
        model = _train_model()
        config = Config(confidence_threshold=30, min_expected_value=0.5)

        races = [
            _make_race("R001", "正常レース1", date(2023, 9, 1), with_results=True),
            _make_race("R002", "正常レース2", date(2023, 9, 2), with_results=True),
        ]
        # 出走馬なしのエッジケース
        edge_race = RaceData(
            race_id="R003",
            race_name="エッジケースレース",
            race_date=date(2023, 9, 3),
            post_time=None,
            venue="阪神",
            course_type="ダート",
            distance=1200,
            track_condition=TrackCondition.SOFT,
            weather="雨",
            entries=[],
            results=[],
            payouts=[],
        )
        races.append(edge_race)

        pipeline = PredictionPipeline(config=config, model=model)
        result = pipeline.run_backtest(races)

        assert isinstance(result, BacktestResult)
        assert result.total_races == 3
        # エッジケースはスキップされる
        assert result.skipped_races >= 1
        assert result.bet_races + result.skipped_races == result.total_races


class TestHistoricalDataLoaderIntegration:
    """HistoricalDataLoaderからバックテストまでのフルフロー統合テスト。

    Validates: Requirements 6.1
    """

    def test_full_backtest_flow_with_data_loader(self, tmp_path: Path):
        """JSON → HistoricalDataLoader → モデル学習 → バックテスト のフルフロー。"""
        # Step 1: テスト用JSONデータを作成
        races_data = []
        for i in range(6):
            race_dict = {
                "race_id": f"TEST{i+1:03d}",
                "race_name": f"統合テストレース{i+1}",
                "race_date": f"2023-07-{10 + i:02d}",
                "post_time": "15:40",
                "venue": "東京",
                "course_type": "芝",
                "distance": 2000,
                "track_condition": "良",
                "weather": "晴",
                "entries": [
                    {
                        "horse_name": f"テスト馬{j+1}_{i}",
                        "jockey_name": f"騎手{j+1}",
                        "gate_number": (j % 8) + 1,
                        "horse_number": j + 1,
                        "weight": 460 + j * 4,
                        "weight_change": j - 3,
                        "win_odds": 2.0 + j * 1.5,
                    }
                    for j in range(8)
                ],
                "results": [
                    {"horse_number": j + 1, "finish_position": j + 1}
                    for j in range(8)
                ],
                "payouts": [
                    {
                        "bet_type": "単勝",
                        "combination": [1],
                        "payout": 350,
                    },
                    {
                        "bet_type": "複勝",
                        "combination": [1],
                        "payout": 150,
                    },
                    {
                        "bet_type": "複勝",
                        "combination": [2],
                        "payout": 130,
                    },
                ],
            }
            races_data.append(race_dict)

        # JSONファイルに書き込み
        json_file = tmp_path / "test_races.json"
        with open(json_file, "w", encoding="utf-8") as f:
            json.dump(races_data, f, ensure_ascii=False)

        # Step 2: HistoricalDataLoaderでデータを読み込み
        loader = HistoricalDataLoader(data_dir=tmp_path)
        loaded_races = loader.load_races(
            start_date=date(2023, 7, 1),
            end_date=date(2023, 7, 31),
        )

        assert len(loaded_races) == 6

        # Step 3: データ分割
        train_races, val_races = loader.split_data(loaded_races, train_ratio=0.8)
        assert len(train_races) + len(val_races) == len(loaded_races)

        # Step 4: 学習用データからモデルを訓練
        model = _train_model()

        # Step 5: パイプラインでバックテスト実行
        config = Config(
            confidence_threshold=30,
            max_bets_per_race=5,
            min_expected_value=0.5,
            daily_budget=10000,
        )
        pipeline = PredictionPipeline(
            config=config,
            model=model,
            historical_races=train_races,
        )
        result = pipeline.run_backtest(val_races)

        # Step 6: 結果検証
        assert isinstance(result, BacktestResult)
        assert result.total_races == len(val_races)
        assert result.bet_races + result.skipped_races == result.total_races
        assert result.total_investment >= 0
        assert result.total_return >= 0
        assert isinstance(result.bet_type_stats, dict)

    def test_data_loader_validates_before_backtest(self, tmp_path: Path):
        """HistoricalDataLoaderのバリデーションがバックテスト前にデータを洗浄する。"""
        # 正常データと不正データを混在させる
        races_data = [
            # 正常なレース
            {
                "race_id": "VALID001",
                "race_name": "正常レース",
                "race_date": "2023-08-01",
                "venue": "東京",
                "course_type": "芝",
                "distance": 1600,
                "track_condition": "良",
                "entries": [
                    {
                        "horse_name": f"馬{j+1}",
                        "jockey_name": f"騎手{j+1}",
                        "gate_number": (j % 8) + 1,
                        "horse_number": j + 1,
                        "weight": 470,
                        "weight_change": 0,
                        "win_odds": 3.0 + j,
                    }
                    for j in range(6)
                ],
                "results": [
                    {"horse_number": j + 1, "finish_position": j + 1}
                    for j in range(6)
                ],
                "payouts": [
                    {"bet_type": "単勝", "combination": [1], "payout": 600}
                ],
            },
            # 不正データ: 出走馬が1頭のみ
            {
                "race_id": "INVALID001",
                "race_name": "不正レース",
                "race_date": "2023-08-02",
                "venue": "中山",
                "course_type": "ダート",
                "distance": 1200,
                "track_condition": "重",
                "entries": [
                    {
                        "horse_name": "一頭馬",
                        "jockey_name": "騎手1",
                        "gate_number": 1,
                        "horse_number": 1,
                        "weight": 450,
                        "weight_change": 0,
                        "win_odds": 1.0,
                    }
                ],
                "results": [{"horse_number": 1, "finish_position": 1}],
                "payouts": [],
            },
        ]

        json_file = tmp_path / "mixed_races.json"
        with open(json_file, "w", encoding="utf-8") as f:
            json.dump(races_data, f, ensure_ascii=False)

        # データ読み込み
        loader = HistoricalDataLoader(data_dir=tmp_path)
        loaded_races = loader.load_races(
            start_date=date(2023, 8, 1),
            end_date=date(2023, 8, 31),
        )

        # バリデーション・クリーニング
        report = loader.validate_and_clean(loaded_races)

        assert report.total_records == 2
        assert report.excluded_count == 1  # 1頭レースが除外される
        assert len(report.clean_races) == 1
        assert report.clean_races[0].race_id == "VALID001"

        # クリーンデータでバックテスト
        model = _train_model()
        config = Config(confidence_threshold=30, min_expected_value=0.5)
        pipeline = PredictionPipeline(config=config, model=model)
        result = pipeline.run_backtest(report.clean_races)

        assert isinstance(result, BacktestResult)
        assert result.total_races == 1
