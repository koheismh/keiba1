"""JSONデータをParquet形式に変換するスクリプト。

data/raw/*.json を読み込み、以下のテーブルに正規化して
data/processed/ に Parquet ファイルとして保存する。

出力ファイル:
- races.parquet: レース基本情報
- entries.parquet: 出走馬情報（レースIDで紐付け）
- results.parquet: 着順結果（レースIDで紐付け）
- payouts.parquet: 払い戻し情報（レースIDで紐付け）
"""

import json
from collections import defaultdict
from pathlib import Path

import pandas as pd


DATA_DIR = Path("data/raw")
OUTPUT_DIR = Path("data/processed")


def load_all_races() -> list[dict]:
    """全JSONファイルからレースデータを読み込む。"""
    all_races = []
    for f in sorted(DATA_DIR.glob("*.json")):
        try:
            with open(f, encoding="utf-8") as fh:
                data = json.load(fh)
            all_races.extend(data)
        except (json.JSONDecodeError, OSError):
            pass
    return all_races


def normalize_races(races: list[dict]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """レースデータを正規化して4つのDataFrameに分割する。"""
    race_rows = []
    entry_rows = []
    result_rows = []
    payout_rows = []

    seen_race_ids = set()

    for race in races:
        race_id = race.get("race_id")
        if not race_id or race_id in seen_race_ids:
            continue
        seen_race_ids.add(race_id)

        # レース基本情報
        race_rows.append({
            "race_id": race_id,
            "race_name": race.get("race_name"),
            "race_date": race.get("race_date"),
            "post_time": race.get("post_time"),
            "venue": race.get("venue"),
            "course_type": race.get("course_type"),
            "distance": race.get("distance"),
            "track_condition": race.get("track_condition"),
            "weather": race.get("weather"),
        })

        # 出走馬
        for entry in race.get("entries", []):
            entry_rows.append({
                "race_id": race_id,
                "horse_name": entry.get("horse_name"),
                "jockey_name": entry.get("jockey_name"),
                "gate_number": entry.get("gate_number"),
                "horse_number": entry.get("horse_number"),
                "weight": entry.get("weight"),
                "weight_change": entry.get("weight_change"),
                "win_odds": entry.get("win_odds"),
            })

        # 結果
        for result in race.get("results", []):
            result_rows.append({
                "race_id": race_id,
                "horse_number": result.get("horse_number"),
                "finish_position": result.get("finish_position"),
            })

        # 払い戻し
        for payout in race.get("payouts", []):
            payout_rows.append({
                "race_id": race_id,
                "bet_type": payout.get("bet_type"),
                "combination": payout.get("combination"),
                "payout": payout.get("payout"),
            })

    df_races = pd.DataFrame(race_rows)
    df_entries = pd.DataFrame(entry_rows)
    df_results = pd.DataFrame(result_rows)
    df_payouts = pd.DataFrame(payout_rows)

    # 型の最適化
    if not df_races.empty:
        df_races["race_date"] = pd.to_datetime(df_races["race_date"], errors="coerce")
        df_races["distance"] = pd.to_numeric(df_races["distance"], errors="coerce").astype("Int32")
        df_races["course_type"] = df_races["course_type"].astype("category")
        df_races["track_condition"] = df_races["track_condition"].astype("category")
        df_races["venue"] = df_races["venue"].astype("category")

    if not df_entries.empty:
        df_entries["gate_number"] = pd.to_numeric(df_entries["gate_number"], errors="coerce").astype("Int16")
        df_entries["horse_number"] = pd.to_numeric(df_entries["horse_number"], errors="coerce").astype("Int16")
        df_entries["weight"] = pd.to_numeric(df_entries["weight"], errors="coerce").astype("Int16")
        df_entries["weight_change"] = pd.to_numeric(df_entries["weight_change"], errors="coerce").astype("Int16")
        df_entries["win_odds"] = pd.to_numeric(df_entries["win_odds"], errors="coerce").astype("Float32")

    if not df_results.empty:
        df_results["horse_number"] = pd.to_numeric(df_results["horse_number"], errors="coerce").astype("Int16")
        df_results["finish_position"] = pd.to_numeric(df_results["finish_position"], errors="coerce").astype("Int16")

    if not df_payouts.empty:
        df_payouts["payout"] = pd.to_numeric(df_payouts["payout"], errors="coerce").astype("Int64")
        df_payouts["bet_type"] = df_payouts["bet_type"].astype("category")

    return df_races, df_entries, df_results, df_payouts


def main():
    print("=" * 60)
    print("  JSON → Parquet 変換")
    print("=" * 60)
    print()

    # 読み込み
    print("Step 1: JSONデータ読み込み...")
    races = load_all_races()
    print(f"  読み込み完了: {len(races):,}レース")
    print()

    # 正規化
    print("Step 2: データ正規化...")
    df_races, df_entries, df_results, df_payouts = normalize_races(races)
    print(f"  races:   {len(df_races):,}行")
    print(f"  entries: {len(df_entries):,}行")
    print(f"  results: {len(df_results):,}行")
    print(f"  payouts: {len(df_payouts):,}行")
    print()

    # 保存
    print("Step 3: Parquet保存...")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    races_path = OUTPUT_DIR / "races.parquet"
    entries_path = OUTPUT_DIR / "entries.parquet"
    results_path = OUTPUT_DIR / "results.parquet"
    payouts_path = OUTPUT_DIR / "payouts.parquet"

    df_races.to_parquet(races_path, index=False, engine="pyarrow")
    df_entries.to_parquet(entries_path, index=False, engine="pyarrow")
    df_results.to_parquet(results_path, index=False, engine="pyarrow")
    df_payouts.to_parquet(payouts_path, index=False, engine="pyarrow")

    print(f"  {races_path} ({races_path.stat().st_size / 1024 / 1024:.1f} MB)")
    print(f"  {entries_path} ({entries_path.stat().st_size / 1024 / 1024:.1f} MB)")
    print(f"  {results_path} ({results_path.stat().st_size / 1024 / 1024:.1f} MB)")
    print(f"  {payouts_path} ({payouts_path.stat().st_size / 1024 / 1024:.1f} MB)")

    total_size = sum(f.stat().st_size for f in OUTPUT_DIR.glob("*.parquet"))
    print()
    print(f"  合計サイズ: {total_size / 1024 / 1024:.1f} MB")
    print()
    print("=" * 60)
    print("  完了！")
    print("=" * 60)


if __name__ == "__main__":
    main()
