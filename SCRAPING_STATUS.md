# スクレイピング状況と再開手順

## 現在の状況（2026-06-12時点）

### 取得済みデータ

| 年 | レース数 | 月の範囲 | 備考 |
|----|---------|---------|------|
| 2015 | 2,580 | 1月〜4月 | ばんえい競馬含む |
| 2016 | 2,540 | 1月〜4月 | ばんえい競馬含む |
| 2017 | 2,560 | 1月〜4月 | ばんえい競馬含む |
| **合計** | **7,680** | | |

### 中断の経緯

1. 最初にリクエスト間隔2〜4秒、limit=300で各年1月分のみ取得（各年300レース）
2. 全月を取得するため、3並列×リクエスト間隔1.0〜1.5秒に変更
3. 約2時間で各年2,500レース取得した時点でnetkeiba.comの`db`サブドメインからIPブロック（HTTP 400）
4. `www.netkeiba.com` にはアクセス可能だが `db.netkeiba.com` と `race.netkeiba.com` がブロックされている

### 未取得データ

- 2015年: 5月〜12月（残り約2,000レース）
- 2016年: 5月〜12月（残り約2,000レース）
- 2017年: 5月〜12月（残り約2,000レース）
- 2018年〜2024年: 全月（各年約3,400レース）

---

## 再開手順

### 1. ブロック解除の確認

```bash
python3 -c "
import requests
session = requests.Session()
session.headers.update({'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'})
r = session.get('https://db.netkeiba.com/race/202406010101/', timeout=30)
print(f'Status: {r.status_code}')
if r.status_code == 200:
    print('ブロック解除！')
else:
    print('まだブロック中')
"
```

### 2. スクレイピング再開（1年ずつ順次実行）

スクリプトは既存データをスキップする機能が実装済みなので、そのまま再実行すれば残りだけ取得される。

```bash
# リクエスト間隔は 2〜3秒に設定推奨（scripts/scrape_historical.py の MIN_SLEEP / MAX_SLEEP を変更）
# 現在の設定: MIN_SLEEP=1.0, MAX_SLEEP=1.5 → ブロック対策として 2.0〜3.0 に変更推奨

# 2015年の残りを取得
python3 scripts/scrape_historical.py --year 2015 --months 1 2 3 4 5 6 7 8 9 10 11 12 --limit 0

# 2016年
python3 scripts/scrape_historical.py --year 2016 --months 1 2 3 4 5 6 7 8 9 10 11 12 --limit 0

# 2017年
python3 scripts/scrape_historical.py --year 2017 --months 1 2 3 4 5 6 7 8 9 10 11 12 --limit 0

# 2018年〜2024年
for year in 2018 2019 2020 2021 2022 2023 2024; do
  python3 scripts/scrape_historical.py --year $year --months 1 2 3 4 5 6 7 8 9 10 11 12 --limit 0
done
```

### 3. 並列実行する場合（ブロック対策版）

```bash
# 2並列でスリープ2〜3秒なら問題ないはず
python3 scripts/scrape_all_years.py --workers 2
```

### 4. 取得後のデータ確認

```bash
python3 -c "
import json
from pathlib import Path
from collections import defaultdict

by_year = defaultdict(int)
for f in sorted(Path('data/raw').glob('*.json')):
    with open(f) as fh:
        data = json.load(fh)
    year = f.stem[:4]
    by_year[year] += len(data)

total = sum(by_year.values())
print(f'合計: {total}レース')
for year in sorted(by_year.keys()):
    print(f'  {year}: {by_year[year]}レース')
"
```

### 5. プッシュ

```bash
git add data/raw/ README.md SCRAPING_STATUS.md
git commit -m "data: YYYY年レースデータ追加"
git push origin main
```

---

## スリープ設定の推奨値

| 並列数 | MIN_SLEEP | MAX_SLEEP | 想定所要時間/年 |
|--------|-----------|-----------|---------------|
| 1 | 2.0 | 3.0 | 約2.5時間 |
| 2 | 3.0 | 4.0 | 約2時間 |
| 3 | 5.0 | 6.0 | 約2時間 |

`scripts/scrape_historical.py` の先頭付近にある定数を変更：

```python
MIN_SLEEP = 2.0  # ← ここを変更
MAX_SLEEP = 3.0  # ← ここを変更
```

---

## バックテスト（現在のデータで実行可能）

現在の7,680レースでもバックテストは実行可能：

```bash
python3 scripts/run_backtest.py --config config/backtest.yaml
```

---

## 技術メモ

- スクレイピング対象: `https://db.netkeiba.com/race/{race_id}/`
- レースID形式: YYYYVVKKDDRRNN（年4桁、場所2桁、開催2桁、日目2桁、レース番号2桁）
- HTMLエンコーディング: EUC-JP
- ばんえい競馬（帯広）は距離情報なしのためデータクリーニングで自動除外される
- 距離パース問題: 「芝右 外1200m」のようなスペース入りパターンは修正済み
- 払い戻しパース: `<br/>`区切りの複数組み合わせ対応済み
- 既存データスキップ: race_idベースで重複取得を防止する機能あり
