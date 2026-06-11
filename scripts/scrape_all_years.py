"""全年のデータを並列でスクレイピングするスクリプト。

10年分（2015-2024）のデータを並列プロセスで取得する。
各プロセスは1年分を担当し、リクエスト間に1.0〜1.5秒のスリープを入れる。

使用方法:
    python3 scripts/scrape_all_years.py
    python3 scripts/scrape_all_years.py --workers 5
    python3 scripts/scrape_all_years.py --years 2020 2021 2022
"""

import argparse
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path


def scrape_year(year: int) -> tuple[int, bool, str]:
    """1年分のスクレイピングを実行する。"""
    cmd = [
        sys.executable,
        "scripts/scrape_historical.py",
        "--year", str(year),
        "--months", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12",
        "--limit", "0",
    ]
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=18000,  # 5時間タイムアウト
        )
        # 最終行を取得
        lines = result.stdout.strip().split("\n") if result.stdout else []
        last_lines = "\n".join(lines[-5:]) if lines else ""
        stderr_tail = result.stderr[-500:] if result.stderr else ""
        output = last_lines + "\n" + stderr_tail
        return (year, result.returncode == 0, output)
    except subprocess.TimeoutExpired:
        return (year, False, "タイムアウト（5時間）")
    except Exception as e:
        return (year, False, str(e))


def main():
    parser = argparse.ArgumentParser(description="全年のデータを並列スクレイピング")
    parser.add_argument(
        "--years",
        type=int,
        nargs="+",
        default=list(range(2015, 2025)),
        help="対象年（デフォルト: 2015-2024）",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=3,
        help="並列ワーカー数（デフォルト: 3）",
    )
    args = parser.parse_args()

    print(f"=== 並列スクレイピング開始 ===")
    print(f"対象年: {args.years}")
    print(f"並列数: {args.workers}")
    print(f"スリープ: 1.0〜1.5秒/リクエスト")
    print()

    start_time = time.time()
    results = {}

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(scrape_year, year): year for year in args.years}

        for future in as_completed(futures):
            year, success, output = future.result()
            status = "✅ 成功" if success else "❌ 失敗"
            elapsed = time.time() - start_time
            print(f"[{elapsed/60:.0f}分経過] {year}年: {status}")
            if not success:
                print(f"  {output[:200]}")
            results[year] = success

    elapsed = time.time() - start_time
    print()
    print(f"=== 完了 ({elapsed/60:.1f}分) ===")
    succeeded = sum(1 for v in results.values() if v)
    print(f"成功: {succeeded}/{len(args.years)}年")


if __name__ == "__main__":
    main()
