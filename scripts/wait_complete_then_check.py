"""補完バッチ完了を待ってから最終チェック＆レポートを実行する。

scrape_until_complete.py / wait_and_complete.py の完了を待ち、
その後 final_check_and_report.py を実行する。
"""

import subprocess
import sys
import time

import psutil


def is_scrape_running() -> bool:
    """スクレイピング関連プロセスが実行中かチェック。"""
    keywords = ["scrape_all_years.py", "scrape_historical.py",
                "scrape_until_complete.py", "wait_and_complete.py"]
    for proc in psutil.process_iter(["cmdline"]):
        try:
            cmdline = proc.info.get("cmdline") or []
            cmdline_str = " ".join(cmdline)
            if any(kw in cmdline_str for kw in keywords):
                # 自分自身は除外
                if "wait_complete_then_check.py" not in cmdline_str:
                    return True
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return False


def main():
    print(f"[{time.strftime('%H:%M:%S')}] スクレイピング完了を待機中...")
    print()

    while is_scrape_running():
        time.sleep(60)

    print(f"[{time.strftime('%H:%M:%S')}] スクレイピングが完了しました。")
    print()
    print("=== 最終チェック＆レポート生成を開始 ===")
    print()

    result = subprocess.run([sys.executable, "scripts/final_check_and_report.py"])
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
