"""全年度のデータが揃うまで繰り返しスクレイピングを実行するスクリプト。

scrape_all_years.py を繰り返し実行し、不足が解消されたら
check_data_completeness.py でデータチェックを行う。
"""

import json
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path


YEARS = list(range(2015, 2027))
# 2026年は6月まで、それ以外は12月まで
EXPECTED_MONTHS = {
    str(y): set(f"{m:02d}" for m in range(1, 13)) for y in range(2015, 2026)
}
EXPECTED_MONTHS["2026"] = set(f"{m:02d}" for m in range(1, 7))


def get_coverage() -> dict[str, set[str]]:
    """各年のカバー済み月を返す。"""
    data_dir = Path("data/raw")
    by_year: dict[str, set[str]] = defaultdict(set)
    for f in data_dir.glob("*.json"):
        year = f.stem[:4]
        month = f.stem[5:7]
        by_year[year].add(month)
    return by_year


def get_incomplete_years() -> list[str]:
    """欠落のある年を返す。"""
    coverage = get_coverage()
    incomplete = []
    for year_str, expected in sorted(EXPECTED_MONTHS.items()):
        covered = coverage.get(year_str, set())
        if not expected.issubset(covered):
            incomplete.append(year_str)
    return incomplete


def run_batch(years: list[str]) -> None:
    """指定年のスクレイピングバッチを実行する。"""
    year_args = " ".join(years)
    cmd = [
        sys.executable,
        "scripts/scrape_all_years.py",
        "--workers", "1",
        "--years",
    ] + years

    print(f"  実行: scrape_all_years.py --workers 1 --years {year_args}")
    print(f"  開始時刻: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    result = subprocess.run(cmd, timeout=None)
    print()
    print(f"  終了時刻: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  終了コード: {result.returncode}")


def run_check() -> None:
    """データ網羅性チェックを実行する。"""
    print()
    print("=" * 60)
    print("データ網羅性チェック")
    print("=" * 60)
    subprocess.run([sys.executable, "scripts/check_data_completeness.py"])


def main():
    max_iterations = 5  # 無限ループ防止
    iteration = 0

    while True:
        iteration += 1
        print()
        print("=" * 60)
        print(f"  反復 {iteration}/{max_iterations}")
        print("=" * 60)

        incomplete = get_incomplete_years()
        if not incomplete:
            print("  全年度のデータが揃っています！")
            break

        if iteration > max_iterations:
            print(f"  最大反復回数 ({max_iterations}) に達しました。")
            print(f"  まだ不足のある年: {incomplete}")
            break

        print(f"  不足のある年: {incomplete}")
        print()

        run_batch(incomplete)

    # 最終チェック
    run_check()


if __name__ == "__main__":
    main()
