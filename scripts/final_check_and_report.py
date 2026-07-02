"""全データ取得完了後の最終チェック＆レポート生成スクリプト。

1. データの網羅性チェック（月カバレッジ）
2. 不足があれば補完を繰り返す（完全に揃うまで）
3. データ品質チェック（不審なデータの検出）
4. report.md にレポートを出力
"""

import json
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from datetime import datetime


DATA_DIR = Path("data/raw")

# 期待する月カバレッジ
EXPECTED_MONTHS = {
    str(y): set(f"{m:02d}" for m in range(1, 13)) for y in range(2015, 2026)
}
EXPECTED_MONTHS["2026"] = set(f"{m:02d}" for m in range(1, 7))


def get_coverage() -> dict[str, set[str]]:
    """各年のカバー済み月を返す。"""
    by_year: dict[str, set[str]] = defaultdict(set)
    for f in DATA_DIR.glob("*.json"):
        year = f.stem[:4]
        month = f.stem[5:7]
        by_year[year].add(month)
    return by_year


def get_incomplete_years() -> list[tuple[str, list[str]]]:
    """欠落のある年と欠落月を返す。"""
    coverage = get_coverage()
    incomplete = []
    for year_str, expected in sorted(EXPECTED_MONTHS.items()):
        covered = coverage.get(year_str, set())
        missing = sorted(expected - covered)
        if missing:
            incomplete.append((year_str, missing))
    return incomplete


def run_补完(year: str, months: list[str]) -> bool:
    """指定年の欠落月を取得する。"""
    month_args = [str(int(m)) for m in months]
    cmd = [
        sys.executable,
        "scripts/scrape_historical.py",
        "--year", year,
        "--months",
    ] + month_args + ["--limit", "0"]

    print(f"    補完実行: {year}年 {','.join(months)}月")
    try:
        result = subprocess.run(cmd, timeout=18000, capture_output=True, text=True)
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        print(f"    タイムアウト: {year}年")
        return False
    except Exception as e:
        print(f"    エラー: {year}年 - {e}")
        return False


def ensure_complete():
    """全データが揃うまで補完を繰り返す。"""
    max_iterations = 20
    iteration = 0

    while True:
        iteration += 1
        incomplete = get_incomplete_years()

        if not incomplete:
            print("  ✅ 全年度・全月のデータが揃いました！")
            return True

        if iteration > max_iterations:
            print(f"  ⚠️  最大反復回数 ({max_iterations}) に達しました。")
            print(f"  まだ不足: {[(y, m) for y, m in incomplete]}")
            return False

        print(f"\n  --- 補完反復 {iteration}/{max_iterations} ---")
        print(f"  不足のある年: {len(incomplete)}件")

        for year, months in incomplete:
            run_补完(year, months)

        # 改善されたか確認
        new_incomplete = get_incomplete_years()
        if len(new_incomplete) == len(incomplete):
            # 前回と同じ不足 → ブロックされている可能性
            same_count = sum(1 for (y1, m1), (y2, m2) in zip(incomplete, new_incomplete) if y1 == y2 and m1 == m2)
            if same_count == len(incomplete):
                print("  ⚠️  進展なし。ブロックされている可能性があります。")
                print("  10分待機してから再試行...")
                time.sleep(600)


def load_all_races() -> list[dict]:
    """全レースデータを読み込む。"""
    all_races = []
    broken_files = []

    for f in sorted(DATA_DIR.glob("*.json")):
        try:
            with open(f, encoding="utf-8") as fh:
                data = json.load(fh)
            for race in data:
                race["_source_file"] = f.name
            all_races.extend(data)
        except Exception as e:
            broken_files.append((f.name, str(e)[:100]))

    return all_races, broken_files


def check_data_quality(races: list[dict]) -> dict:
    """データ品質をチェックし、不審なデータを検出する。"""
    issues = {
        "missing_fields": [],
        "invalid_distance": [],
        "invalid_odds": [],
        "no_entries": [],
        "no_results": [],
        "no_payouts": [],
        "duplicate_race_ids": [],
        "invalid_dates": [],
        "suspicious_payouts": [],
    }

    race_ids_seen = {}
    required_fields = ["race_id", "race_name", "race_date", "venue", "course_type",
                       "distance", "entries", "results", "payouts"]

    for race in races:
        race_id = race.get("race_id", "unknown")
        source = race.get("_source_file", "unknown")

        # 重複チェック
        if race_id in race_ids_seen:
            issues["duplicate_race_ids"].append({
                "race_id": race_id,
                "file1": race_ids_seen[race_id],
                "file2": source,
            })
        else:
            race_ids_seen[race_id] = source

        # 必須フィールドチェック
        missing = [f for f in required_fields if f not in race or race[f] is None]
        if missing:
            issues["missing_fields"].append({
                "race_id": race_id,
                "source": source,
                "missing": missing,
            })

        # 距離チェック
        distance = race.get("distance", 0)
        if distance is not None and (distance < 800 or distance > 4000):
            if distance != 0:  # 0は未取得
                issues["invalid_distance"].append({
                    "race_id": race_id,
                    "source": source,
                    "distance": distance,
                })

        # エントリーチェック
        entries = race.get("entries", [])
        if not entries:
            issues["no_entries"].append({"race_id": race_id, "source": source})

        # 結果チェック
        results = race.get("results", [])
        if not results:
            issues["no_results"].append({"race_id": race_id, "source": source})

        # オッズチェック
        for entry in entries:
            odds = entry.get("win_odds")
            if odds is not None and (odds <= 0 or odds > 10000):
                issues["invalid_odds"].append({
                    "race_id": race_id,
                    "source": source,
                    "horse": entry.get("horse_name", "?"),
                    "odds": odds,
                })

        # 払い戻しチェック
        payouts = race.get("payouts", [])
        if not payouts:
            issues["no_payouts"].append({"race_id": race_id, "source": source})

        for payout in payouts:
            amount = payout.get("payout", 0)
            if amount > 10000000:  # 1000万円超
                issues["suspicious_payouts"].append({
                    "race_id": race_id,
                    "source": source,
                    "bet_type": payout.get("bet_type"),
                    "payout": amount,
                })

        # 日付チェック
        race_date = race.get("race_date", "")
        if race_date:
            try:
                dt = datetime.strptime(race_date, "%Y-%m-%d")
                if dt.year < 2015 or dt > datetime(2026, 7, 1):
                    issues["invalid_dates"].append({
                        "race_id": race_id,
                        "source": source,
                        "date": race_date,
                    })
            except ValueError:
                issues["invalid_dates"].append({
                    "race_id": race_id,
                    "source": source,
                    "date": race_date,
                    "error": "パース失敗",
                })

    return issues


def generate_report(races: list[dict], broken_files: list, issues: dict, coverage: dict):
    """report.md を生成する。"""
    lines = []
    lines.append("# データ取得・品質チェックレポート")
    lines.append("")
    lines.append(f"生成日時: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")

    # サマリー
    lines.append("## サマリー")
    lines.append("")
    lines.append(f"- 総レース数: **{len(races):,}**")
    lines.append(f"- JSONファイル数: **{len(list(DATA_DIR.glob('*.json')))}**")

    by_year = defaultdict(int)
    for race in races:
        year = race.get("race_date", "0000")[:4]
        by_year[year] += 1

    lines.append(f"- 年の範囲: {min(by_year.keys())}〜{max(by_year.keys())}")
    lines.append("")

    # 年別レース数
    lines.append("## 年別レース数")
    lines.append("")
    lines.append("| 年 | レース数 | カバー月 | ステータス |")
    lines.append("|----|---------|---------|-----------|")

    for year_str in sorted(EXPECTED_MONTHS.keys()):
        count = by_year.get(year_str, 0)
        covered = sorted(coverage.get(year_str, set()))
        expected = EXPECTED_MONTHS[year_str]
        missing = expected - set(covered)
        month_range = f"{covered[0]}〜{covered[-1]}月" if covered else "なし"
        status = "✅ 完了" if not missing else f"❌ 欠落: {','.join(sorted(missing))}月"
        lines.append(f"| {year_str} | {count:,} | {month_range} | {status} |")

    lines.append("")

    # 破損ファイル
    if broken_files:
        lines.append("## ⚠️ 破損ファイル")
        lines.append("")
        for name, err in broken_files:
            lines.append(f"- `{name}`: {err}")
        lines.append("")

    # データ品質問題
    lines.append("## データ品質チェック結果")
    lines.append("")

    issue_labels = {
        "duplicate_race_ids": "重複レースID",
        "missing_fields": "必須フィールド欠損",
        "invalid_distance": "異常な距離値",
        "invalid_odds": "異常なオッズ値",
        "no_entries": "出走馬データなし",
        "no_results": "結果データなし",
        "no_payouts": "払い戻しデータなし",
        "invalid_dates": "異常な日付",
        "suspicious_payouts": "高額払い戻し（要確認）",
    }

    has_issues = False
    for key, label in issue_labels.items():
        count = len(issues.get(key, []))
        if count > 0:
            has_issues = True
            lines.append(f"### {label}（{count}件）")
            lines.append("")
            # 最大20件表示
            for item in issues[key][:20]:
                lines.append(f"- {item}")
            if count > 20:
                lines.append(f"- ... 他 {count - 20}件")
            lines.append("")

    if not has_issues:
        lines.append("✅ 重大な品質問題は検出されませんでした。")
        lines.append("")

    # 統計
    lines.append("## 統計情報")
    lines.append("")

    # コースタイプ分布
    course_types = defaultdict(int)
    venues = defaultdict(int)
    for race in races:
        course_types[race.get("course_type", "不明")] += 1
        venues[race.get("venue", "不明")] += 1

    lines.append("### コースタイプ別")
    lines.append("")
    for ct, count in sorted(course_types.items(), key=lambda x: -x[1]):
        lines.append(f"- {ct}: {count:,}レース ({count/len(races)*100:.1f}%)")
    lines.append("")

    lines.append("### 会場別（上位10）")
    lines.append("")
    for venue, count in sorted(venues.items(), key=lambda x: -x[1])[:10]:
        lines.append(f"- {venue}: {count:,}レース")
    lines.append("")

    # ファイル書き出し
    report_path = Path("report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"  レポート生成完了: {report_path}")
    return report_path


def main():
    print("=" * 60)
    print("  最終チェック＆レポート生成")
    print("=" * 60)
    print()

    # Step 1: 網羅性チェック＆補完
    print("Step 1: データ網羅性チェック＆補完")
    ensure_complete()

    # Step 2: 全データ読み込み
    print()
    print("Step 2: 全データ読み込み")
    races, broken_files = load_all_races()
    print(f"  読み込み完了: {len(races):,}レース")
    if broken_files:
        print(f"  ⚠️ 破損ファイル: {len(broken_files)}件")

    # Step 3: 品質チェック
    print()
    print("Step 3: データ品質チェック")
    issues = check_data_quality(races)
    total_issues = sum(len(v) for v in issues.values())
    print(f"  検出された問題: {total_issues}件")

    # Step 4: レポート生成
    print()
    print("Step 4: レポート生成")
    coverage = get_coverage()
    generate_report(races, broken_files, issues, coverage)

    print()
    print("=" * 60)
    print("  完了！ report.md を確認してください。")
    print("=" * 60)


if __name__ == "__main__":
    main()
