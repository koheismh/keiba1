"""2015-2024の取得完了を待って、2025年分を自動実行するスクリプト。"""

import subprocess
import sys
import time

import psutil


def is_scrape_all_running() -> bool:
    """scrape_all_years.py が実行中かチェックする。"""
    for proc in psutil.process_iter(["cmdline"]):
        try:
            cmdline = proc.info.get("cmdline") or []
            if any("scrape_all_years.py" in arg for arg in cmdline):
                return True
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return False


def main():
    print("2015-2024年の取得完了を待機中...")
    print("(scrape_all_years.py のプロセス終了を監視)")
    print()

    # まず実行中であることを確認
    if not is_scrape_all_running():
        print("scrape_all_years.py が見つかりません。直接2025年の取得を開始します。")
    else:
        # 完了を待つ（30秒ごとにチェック）
        while is_scrape_all_running():
            time.sleep(30)
        print("scrape_all_years.py が完了しました。")

    print()
    print("=== 2025年（1〜6月）の取得を開始 ===")

    cmd = [
        sys.executable,
        "scripts/scrape_historical.py",
        "--year", "2025",
        "--months", "1", "2", "3", "4", "5", "6",
        "--limit", "0",
    ]

    result = subprocess.run(cmd)
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
