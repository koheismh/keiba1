"""現在のバッチ完了を待ってから、不足分の補完を繰り返す。

scrape_all_years.py のプロセス終了を待ち、
その後 scrape_until_complete.py を実行する。
"""

import subprocess
import sys
import time

import psutil


def is_scrape_running() -> bool:
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
    print(f"[{time.strftime('%H:%M:%S')}] 現在のバッチ完了を待機中...")
    print()

    if is_scrape_running():
        while is_scrape_running():
            time.sleep(60)
        print(f"[{time.strftime('%H:%M:%S')}] バッチが完了しました。")
    else:
        print("scrape_all_years.py が見つかりません。直接補完を開始します。")

    print()
    print("=== 不足分の補完を開始 ===")
    print()

    result = subprocess.run([sys.executable, "scripts/scrape_until_complete.py"])
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
