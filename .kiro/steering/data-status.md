---
inclusion: auto
---

# データ取得状況

## Parquet データ

- 最終レース日付: 2026-06-29
- 次回取得開始日: 2026-06-30 以降
- 保存先: data/processed/
- 詳細: data/processed/META.md を参照

## 更新手順

1. 新規データをスクレイピング（2026-06-30以降のみ）
2. `python scripts/convert_to_parquet.py` で全データからParquet再生成
3. data/processed/META.md の最終レース日付を更新
