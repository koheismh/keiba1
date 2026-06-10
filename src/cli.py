"""CLIエントリーポイント

競馬レース予測システムのCLIインターフェース。
サブコマンド: backtest, predict, record, retrain
"""

import argparse
import sys
from datetime import date, datetime
from pathlib import Path


def _parse_date(date_str: str) -> date:
    """日付文字列をdateオブジェクトに変換する。

    Args:
        date_str: YYYY-MM-DD形式の日付文字列

    Returns:
        dateオブジェクト

    Raises:
        argparse.ArgumentTypeError: 日付形式が不正な場合
    """
    try:
        return datetime.strptime(date_str, "%Y-%m-%d").date()
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"日付の形式が不正です: '{date_str}'。YYYY-MM-DD形式で指定してください。"
        )


def handle_backtest(args: argparse.Namespace) -> None:
    """バックテストサブコマンドのハンドラ。

    指定期間の過去レースデータに対してモデルを適用し、
    仮想的に馬券を購入したシミュレーションを実行する。

    Args:
        args: パース済みコマンドライン引数
    """
    from src.config import ConfigManager
    from src.exceptions import ModelError
    from src.pipeline import PredictionPipeline
    from src.prediction.model import PredictionModel

    config_path = Path(args.config)
    print(f"[backtest] 設定ファイル: {config_path}")
    print(f"[backtest] 期間: {args.start_date} 〜 {args.end_date}")

    # 設定読み込み
    config_manager = ConfigManager()
    try:
        config = config_manager.load(config_path)
    except FileNotFoundError as e:
        print(f"エラー: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"[backtest] 信頼度閾値: {config.confidence_threshold}")
    print(f"[backtest] 最大買い目数: {config.max_bets_per_race}")
    print(f"[backtest] 日次予算: {config.daily_budget:,}円")
    print()

    # モデル読み込み
    if args.model:
        model_path = Path(args.model)
        print(f"[backtest] モデル: {model_path}")
        model = PredictionModel()
        try:
            model.load(model_path)
        except ModelError as e:
            print(f"エラー: {e}", file=sys.stderr)
            sys.exit(1)

        # パイプライン構築
        pipeline = PredictionPipeline(config=config, model=model)

        print("[backtest] バックテストを開始します...")
        # Note: In a full implementation, races would be loaded from
        # HistoricalDataLoader for the given date range.
        # For now, we show that the pipeline is connected.
        print("[backtest] ※レースデータの読み込みにはHistoricalDataLoaderが必要です。")
    else:
        print("[backtest] バックテストを開始します...")
        print("[backtest] ※モデルが指定されていません。--model オプションで学習済みモデルを指定してください。")


def handle_predict(args: argparse.Namespace) -> None:
    """当日予測サブコマンドのハンドラ。

    指定日のレースデータを取得し、モデルで予測を行い、
    推奨買い目を出力する。

    Args:
        args: パース済みコマンドライン引数
    """
    from src.config import ConfigManager
    from src.exceptions import ModelError
    from src.pipeline import PredictionPipeline
    from src.prediction.model import PredictionModel

    config_path = Path(args.config)
    model_path = Path(args.model)

    print(f"[predict] 設定ファイル: {config_path}")
    print(f"[predict] 予測対象日: {args.date}")
    print(f"[predict] モデル: {model_path}")

    # 設定読み込み
    config_manager = ConfigManager()
    try:
        config = config_manager.load(config_path)
    except FileNotFoundError as e:
        print(f"エラー: {e}", file=sys.stderr)
        sys.exit(1)

    # モデルファイル存在チェックと読み込み
    model = PredictionModel()
    try:
        model.load(model_path)
    except ModelError as e:
        print(f"エラー: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"[predict] 信頼度閾値: {config.confidence_threshold}")
    print(f"[predict] 最大買い目数: {config.max_bets_per_race}")
    print(f"[predict] 日次予算: {config.daily_budget:,}円")
    print()

    # パイプライン構築
    pipeline = PredictionPipeline(config=config, model=model)

    print("[predict] 当日予測を開始します...")
    # Note: In a full implementation, races would be fetched from
    # RaceDataFetcher for the given date.
    # For now, we show that the pipeline is connected.
    print("[predict] ※レースデータの取得にはRaceDataFetcherが必要です。")


def handle_record(args: argparse.Namespace) -> None:
    """実績記録サブコマンドのハンドラ。

    指定日のレース結果を取得し、予測結果と比較して
    的中・不的中を記録する。

    Args:
        args: パース済みコマンドライン引数
    """
    print(f"[record] 記録対象日: {args.date}")
    print()
    print("[record] 実績記録を開始します...")
    print("[record] ※パイプライン統合は次タスクで実装されます。")


def handle_retrain(args: argparse.Namespace) -> None:
    """モデル再学習サブコマンドのハンドラ。

    指定期間のデータを使用してモデルの再学習を実行する。

    Args:
        args: パース済みコマンドライン引数
    """
    from src.config import ConfigManager

    config_path = Path(args.config)
    output_path = Path(args.output)

    print(f"[retrain] 設定ファイル: {config_path}")
    print(f"[retrain] 期間: {args.start_date} 〜 {args.end_date}")
    print(f"[retrain] 出力先: {output_path}")

    # 設定読み込み
    config_manager = ConfigManager()
    try:
        config = config_manager.load(config_path)
    except FileNotFoundError as e:
        print(f"エラー: {e}", file=sys.stderr)
        sys.exit(1)

    print()
    print("[retrain] モデル再学習を開始します...")
    print("[retrain] ※パイプライン統合は次タスクで実装されます。")


def create_parser() -> argparse.ArgumentParser:
    """CLIパーサーを構築する。

    Returns:
        設定済みのArgumentParser
    """
    parser = argparse.ArgumentParser(
        prog="horse-race-predictor",
        description="競馬レース予測システム - 回収率最適化アドバイスツール",
    )

    subparsers = parser.add_subparsers(dest="command")

    # backtest サブコマンド
    backtest_parser = subparsers.add_parser(
        "backtest",
        help="バックテスト実行 - 過去データでモデル性能を検証",
    )
    backtest_parser.add_argument(
        "--start-date",
        required=True,
        type=_parse_date,
        help="開始日 (YYYY-MM-DD)",
    )
    backtest_parser.add_argument(
        "--end-date",
        required=True,
        type=_parse_date,
        help="終了日 (YYYY-MM-DD)",
    )
    backtest_parser.add_argument(
        "--config",
        default="config/default.yaml",
        help="設定ファイルパス (デフォルト: config/default.yaml)",
    )
    backtest_parser.add_argument(
        "--model",
        help="学習済みモデルのパス",
    )

    # predict サブコマンド
    predict_parser = subparsers.add_parser(
        "predict",
        help="当日予測 - 指定日のレース予測を実行",
    )
    predict_parser.add_argument(
        "--date",
        required=True,
        type=_parse_date,
        help="予測対象日 (YYYY-MM-DD)",
    )
    predict_parser.add_argument(
        "--config",
        default="config/default.yaml",
        help="設定ファイルパス (デフォルト: config/default.yaml)",
    )
    predict_parser.add_argument(
        "--model",
        required=True,
        help="学習済みモデルのパス",
    )

    # record サブコマンド
    record_parser = subparsers.add_parser(
        "record",
        help="実績記録 - レース結果を記録し的中判定を行う",
    )
    record_parser.add_argument(
        "--date",
        required=True,
        type=_parse_date,
        help="記録対象日 (YYYY-MM-DD)",
    )

    # retrain サブコマンド
    retrain_parser = subparsers.add_parser(
        "retrain",
        help="モデル再学習 - 蓄積データでモデルを更新",
    )
    retrain_parser.add_argument(
        "--start-date",
        required=True,
        type=_parse_date,
        help="学習データ開始日 (YYYY-MM-DD)",
    )
    retrain_parser.add_argument(
        "--end-date",
        required=True,
        type=_parse_date,
        help="学習データ終了日 (YYYY-MM-DD)",
    )
    retrain_parser.add_argument(
        "--config",
        default="config/default.yaml",
        help="設定ファイルパス (デフォルト: config/default.yaml)",
    )
    retrain_parser.add_argument(
        "--output",
        default="data/models/model.txt",
        help="学習済みモデル出力先 (デフォルト: data/models/model.txt)",
    )

    return parser


def main() -> None:
    """メインエントリーポイント。

    コマンドライン引数をパースし、対応するサブコマンドハンドラを実行する。
    サブコマンドが指定されない場合はヘルプを表示する。
    """
    parser = create_parser()
    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    handlers = {
        "backtest": handle_backtest,
        "predict": handle_predict,
        "record": handle_record,
        "retrain": handle_retrain,
    }

    handler = handlers[args.command]
    handler(args)


if __name__ == "__main__":
    main()
