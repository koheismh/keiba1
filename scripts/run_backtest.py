"""バックテスト実行スクリプト。

data/raw/ のレースデータを読み込み、モデルを学習し、
バックテストを実行して結果をレポートする。

使用方法:
    python3 scripts/run_backtest.py
    python3 scripts/run_backtest.py --config config/default.yaml
"""

import argparse
import logging
import sys
from pathlib import Path

# プロジェクトルートをパスに追加
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.backtest.backtester import Backtester
from src.betting.bet_selector import BetSelector
from src.betting.fund_allocator import FundAllocator
from src.config import ConfigManager
from src.data.historical_loader import HistoricalDataLoader
from src.evaluation.race_evaluator import RaceEvaluator
from src.features.engineer import FeatureEngineer
from src.prediction.model import PredictionModel
from src.prediction.trainer import ModelTrainer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="バックテスト実行")
    parser.add_argument(
        "--config",
        type=str,
        default="config/default.yaml",
        help="設定ファイルパス",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/raw",
        help="データディレクトリ",
    )
    args = parser.parse_args()

    # 設定読み込み
    config_manager = ConfigManager()
    config = config_manager.load(Path(args.config))
    logger.info(f"設定: 信頼度閾値={config.confidence_threshold}, "
                f"最大買い目数={config.max_bets_per_race}, "
                f"予算={config.daily_budget}円")

    # データ読み込み
    loader = HistoricalDataLoader(data_dir=Path(args.data_dir))
    from datetime import date
    all_races = loader.load_races(start_date=date(2020, 1, 1), end_date=date(2025, 12, 31))
    logger.info(f"読み込みレース数: {len(all_races)}")

    # データクリーニング
    report = loader.validate_and_clean(all_races)
    clean_races = report.clean_races
    logger.info(f"クリーニング後: {len(clean_races)}レース "
                f"(除外: {report.excluded_count}件)")

    if len(clean_races) < 10:
        logger.error("有効なレースデータが不足しています（10件未満）")
        sys.exit(1)

    # データ分割
    train_races, test_races = loader.split_data(clean_races, train_ratio=0.7)
    logger.info(f"学習用: {len(train_races)}レース, 検証用: {len(test_races)}レース")

    # 特徴量エンジニアリング
    feature_engineer = FeatureEngineer()

    # 学習データの準備
    logger.info("特徴量を抽出中...")
    import numpy as np
    train_features = []
    train_labels = []

    for race in train_races:
        if race.results is None:
            continue
        # 着順マップを作成
        result_map = {r.horse_number: r.finish_position for r in race.results}

        for entry in race.entries:
            if entry.horse_number not in result_map:
                continue
            fv = feature_engineer.extract_features(race, entry)
            train_features.append(fv.values)
            # ラベル: 1着=1, それ以外=0 (二値分類として学習)
            label = 1.0 if result_map[entry.horse_number] == 1 else 0.0
            train_labels.append(label)

    if not train_features:
        logger.error("学習用の特徴量データが生成できませんでした")
        sys.exit(1)

    X_train = np.array(train_features)
    y_train = np.array(train_labels)
    logger.info(f"学習データ: {X_train.shape[0]}サンプル, {X_train.shape[1]}特徴量")
    logger.info(f"1着ラベル比率: {y_train.mean():.3f}")

    # モデル学習
    logger.info("モデルを学習中...")
    model = PredictionModel()
    model.train(X_train, y_train)

    # モデル保存
    model_path = Path("data/models/model.txt")
    model.save(model_path)
    logger.info(f"モデルを保存: {model_path}")

    # バックテスト実行
    logger.info("")
    logger.info("=" * 60)
    logger.info("バックテスト実行")
    logger.info("=" * 60)

    backtester = Backtester(
        feature_engineer=feature_engineer,
        race_evaluator=RaceEvaluator(
            confidence_threshold=config.confidence_threshold
        ),
        bet_selector=BetSelector(
            min_expected_value=config.min_expected_value,
            target_bet_types=config.target_bet_types,
        ),
        fund_allocator=FundAllocator(
            max_single_bet_ratio=config.max_single_bet_ratio
        ),
    )

    result = backtester.run(test_races, model, config)

    # 結果レポート
    print()
    print("=" * 60)
    print("バックテスト結果")
    print("=" * 60)
    print()
    print(f"【対象期間】")
    if test_races:
        print(f"  開始: {test_races[0].race_date}")
        print(f"  終了: {test_races[-1].race_date}")
    print(f"  全レース数: {result.total_races}")
    print(f"  投資レース数: {result.bet_races}")
    print(f"  見送りレース数: {result.skipped_races}")
    print()
    print(f"【収支】")
    print(f"  合計投資金額: {result.total_investment:,}円")
    print(f"  合計払い戻し: {result.total_return:,}円")
    print(f"  損益: {result.total_return - result.total_investment:+,}円")
    print()
    print(f"【パフォーマンス】")
    print(f"  回収率: {result.return_rate * 100:.1f}%")
    print(f"  的中率: {result.hit_rate * 100:.1f}%")
    print(f"  最大ドローダウン: {result.max_drawdown * 100:.1f}%")
    print(f"  シャープレシオ: {result.sharpe_ratio:.3f}")
    print()
    print(f"【券種別成績】")
    for bet_type, stats in result.bet_type_stats.items():
        bets_count = stats.get("bets", stats.get("count", 0))
        if bets_count > 0:
            print(f"  {bet_type.value}: "
                  f"的中率={stats['hit_rate']*100:.1f}%, "
                  f"回収率={stats['return_rate']*100:.1f}%, "
                  f"件数={int(bets_count)}")
    print()
    print("=" * 60)

    # パラメータ調整推奨
    if result.return_rate < 1.0:
        print()
        print("【パラメータ調整推奨】")
        print(f"  回収率が100%を下回っています ({result.return_rate*100:.1f}%)。")
        print(f"  以下の調整を検討してください：")
        if result.skipped_races < result.total_races * 0.3:
            print(f"  - 信頼度閾値を上げる（現在: {config.confidence_threshold}）")
        if config.min_expected_value <= 1.0:
            print(f"  - 期待値基準を上げる（現在: {config.min_expected_value}）")
        print(f"  - 対象券種を限定する")
        print(f"  - より多くの学習データを追加する")


if __name__ == "__main__":
    main()
