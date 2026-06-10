"""CLIエントリーポイントのユニットテスト"""

import sys
from datetime import date
from pathlib import Path
from unittest.mock import patch

import pytest

from src.cli import create_parser, handle_backtest, handle_record, handle_retrain, main


class TestCreateParser:
    """create_parser関数のテスト"""

    def test_parser_has_four_subcommands(self):
        """4つのサブコマンドが登録されていること"""
        parser = create_parser()
        # サブコマンドの存在確認（ヘルプテキストで確認）
        help_text = parser.format_help()
        assert "backtest" in help_text
        assert "predict" in help_text
        assert "record" in help_text
        assert "retrain" in help_text

    def test_backtest_parses_required_args(self):
        """backtestサブコマンドが必須引数をパースできること"""
        parser = create_parser()
        args = parser.parse_args(
            ["backtest", "--start-date", "2024-01-01", "--end-date", "2024-03-31"]
        )
        assert args.command == "backtest"
        assert args.start_date == date(2024, 1, 1)
        assert args.end_date == date(2024, 3, 31)
        assert args.config == "config/default.yaml"
        assert args.model is None

    def test_backtest_parses_optional_args(self):
        """backtestサブコマンドがオプション引数をパースできること"""
        parser = create_parser()
        args = parser.parse_args(
            [
                "backtest",
                "--start-date", "2024-01-01",
                "--end-date", "2024-03-31",
                "--config", "config/custom.yaml",
                "--model", "data/models/best.txt",
            ]
        )
        assert args.config == "config/custom.yaml"
        assert args.model == "data/models/best.txt"

    def test_predict_parses_required_args(self):
        """predictサブコマンドが必須引数をパースできること"""
        parser = create_parser()
        args = parser.parse_args(
            ["predict", "--date", "2024-06-01", "--model", "data/models/model.txt"]
        )
        assert args.command == "predict"
        assert args.date == date(2024, 6, 1)
        assert args.model == "data/models/model.txt"
        assert args.config == "config/default.yaml"

    def test_record_parses_required_args(self):
        """recordサブコマンドが必須引数をパースできること"""
        parser = create_parser()
        args = parser.parse_args(["record", "--date", "2024-06-01"])
        assert args.command == "record"
        assert args.date == date(2024, 6, 1)

    def test_retrain_parses_required_args(self):
        """retrainサブコマンドが必須引数をパースできること"""
        parser = create_parser()
        args = parser.parse_args(
            ["retrain", "--start-date", "2023-01-01", "--end-date", "2024-01-01"]
        )
        assert args.command == "retrain"
        assert args.start_date == date(2023, 1, 1)
        assert args.end_date == date(2024, 1, 1)
        assert args.config == "config/default.yaml"
        assert args.output == "data/models/model.txt"

    def test_retrain_parses_output_arg(self):
        """retrainサブコマンドが出力先引数をパースできること"""
        parser = create_parser()
        args = parser.parse_args(
            [
                "retrain",
                "--start-date", "2023-01-01",
                "--end-date", "2024-01-01",
                "--output", "data/models/new_model.txt",
            ]
        )
        assert args.output == "data/models/new_model.txt"

    def test_invalid_date_format_raises_error(self):
        """不正な日付形式でエラーが発生すること"""
        parser = create_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["backtest", "--start-date", "2024/01/01", "--end-date", "2024-03-31"])

    def test_no_command_sets_none(self):
        """サブコマンドなしの場合commandがNoneになること"""
        parser = create_parser()
        args = parser.parse_args([])
        assert args.command is None


class TestMain:
    """main関数のテスト"""

    def test_no_command_shows_help_and_exits(self, capsys):
        """コマンドなしでヘルプ表示して正常終了すること"""
        with patch("sys.argv", ["horse-race-predictor"]):
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 0

        captured = capsys.readouterr()
        assert "backtest" in captured.out
        assert "predict" in captured.out
        assert "record" in captured.out
        assert "retrain" in captured.out

    def test_backtest_command_dispatches(self, capsys):
        """backtestコマンドがハンドラに正しくディスパッチされること"""
        with patch(
            "sys.argv",
            ["horse-race-predictor", "backtest", "--start-date", "2024-01-01", "--end-date", "2024-03-31"],
        ):
            main()

        captured = capsys.readouterr()
        assert "[backtest]" in captured.out
        assert "2024-01-01" in captured.out
        assert "2024-03-31" in captured.out

    def test_record_command_dispatches(self, capsys):
        """recordコマンドがハンドラに正しくディスパッチされること"""
        with patch("sys.argv", ["horse-race-predictor", "record", "--date", "2024-06-01"]):
            main()

        captured = capsys.readouterr()
        assert "[record]" in captured.out
        assert "2024-06-01" in captured.out

    def test_retrain_command_dispatches(self, capsys):
        """retrainコマンドがハンドラに正しくディスパッチされること"""
        with patch(
            "sys.argv",
            ["horse-race-predictor", "retrain", "--start-date", "2023-01-01", "--end-date", "2024-01-01"],
        ):
            main()

        captured = capsys.readouterr()
        assert "[retrain]" in captured.out
        assert "2023-01-01" in captured.out


class TestHandleBacktest:
    """handle_backtest関数のテスト"""

    def test_loads_config_and_prints_info(self, capsys):
        """設定を読み込み情報を表示すること"""
        parser = create_parser()
        args = parser.parse_args(
            ["backtest", "--start-date", "2024-01-01", "--end-date", "2024-03-31"]
        )
        handle_backtest(args)

        captured = capsys.readouterr()
        assert "config/default.yaml" in captured.out
        assert "2024-01-01" in captured.out
        assert "2024-03-31" in captured.out
        assert "バックテストを開始します" in captured.out

    def test_invalid_config_exits(self):
        """不正な設定ファイルパスでエラー終了すること"""
        parser = create_parser()
        args = parser.parse_args(
            [
                "backtest",
                "--start-date", "2024-01-01",
                "--end-date", "2024-03-31",
                "--config", "nonexistent.yaml",
            ]
        )
        with pytest.raises(SystemExit) as exc_info:
            handle_backtest(args)
        assert exc_info.value.code == 1


class TestHandleRecord:
    """handle_record関数のテスト"""

    def test_prints_date(self, capsys):
        """指定日を表示すること"""
        parser = create_parser()
        args = parser.parse_args(["record", "--date", "2024-06-01"])
        handle_record(args)

        captured = capsys.readouterr()
        assert "2024-06-01" in captured.out
        assert "実績記録を開始します" in captured.out


class TestHandleRetrain:
    """handle_retrain関数のテスト"""

    def test_loads_config_and_prints_info(self, capsys):
        """設定を読み込み情報を表示すること"""
        parser = create_parser()
        args = parser.parse_args(
            ["retrain", "--start-date", "2023-01-01", "--end-date", "2024-01-01"]
        )
        handle_retrain(args)

        captured = capsys.readouterr()
        assert "2023-01-01" in captured.out
        assert "2024-01-01" in captured.out
        assert "モデル再学習を開始します" in captured.out

    def test_invalid_config_exits(self):
        """不正な設定ファイルパスでエラー終了すること"""
        parser = create_parser()
        args = parser.parse_args(
            [
                "retrain",
                "--start-date", "2023-01-01",
                "--end-date", "2024-01-01",
                "--config", "nonexistent.yaml",
            ]
        )
        with pytest.raises(SystemExit) as exc_info:
            handle_retrain(args)
        assert exc_info.value.code == 1
