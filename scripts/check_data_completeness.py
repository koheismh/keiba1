"""データの網羅性をチェックするスクリプト。

各年のレース数・ファイル数を確認し、欠落がないか判定する。
"""

import json
from collections import defaultdict
from pathlib import Path


def main():
    data_dir = Path("data/raw")
    if not data_dir.exists():
        print("data/raw ディレクトリが見つかりません")
        return

    by_year = defaultdict(lambda: {"files": 0, "races": 0, "months": set()})
    broken_files = []

    for f in sorted(data_dir.glob("*.json")):
        year = f.stem[:4]
        month = f.stem[5:7]
        by_year[year]["files"] += 1
        by_year[year]["months"].add(month)
        try:
            with open(f, encoding="utf-8") as fh:
                data = json.load(fh)
            by_year[year]["races"] += len(data)
        except Exception as e:
            broken_files.append((f.name, str(e)[:80]))

    print("=" * 60)
    print(f"{'年':<6} {'ファイル数':<10} {'レース数':<10} {'月カバー':<15} {'判定'}")
    print("=" * 60)

    total_races = 0
    incomplete_years = []

    for year in sorted(by_year.keys()):
        info = by_year[year]
        total_races += info["races"]
        months_covered = sorted(info["months"])
        month_range = f"{months_covered[0]}〜{months_covered[-1]}月" if months_covered else "なし"

        # 2026年は6月まで、それ以外は12月まで期待
        if year == "2026":
            expected_months = set(f"{m:02d}" for m in range(1, 7))
        else:
            expected_months = set(f"{m:02d}" for m in range(1, 13))

        missing_months = expected_months - info["months"]

        if missing_months:
            status = f"❌ 欠落: {','.join(sorted(missing_months))}月"
            incomplete_years.append((year, missing_months))
        else:
            status = "✅ 完了"

        print(f"{year:<6} {info['files']:<10} {info['races']:<10} {month_range:<15} {status}")

    print("=" * 60)
    print(f"合計: {total_races}レース")
    print()

    if broken_files:
        print(f"⚠️  破損ファイル ({len(broken_files)}件):")
        for name, err in broken_files:
            print(f"  {name}: {err}")
        print()

    if incomplete_years:
        print(f"⚠️  欠落のある年 ({len(incomplete_years)}件):")
        for year, months in incomplete_years:
            print(f"  {year}年: {','.join(sorted(months))}月が未取得")
        print()
        print("再取得コマンド:")
        for year, months in incomplete_years:
            month_args = " ".join(str(int(m)) for m in sorted(months))
            print(f"  python scripts/scrape_historical.py --year {year} --months {month_args} --limit 0")
    else:
        print("✅ 全年度のデータが揃っています！")


if __name__ == "__main__":
    main()
