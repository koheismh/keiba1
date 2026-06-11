---
inclusion: manual
---

# スクレイピング再開ガイド

データ取得の開始・再開を行う際は、必ず以下のファイルを先に確認してください：

#[[file:SCRAPING_STATUS.md]]

## 手順

1. まず `SCRAPING_STATUS.md` を読み、現在の取得状況・中断理由・再開手順を確認する
2. ブロック確認コマンドを実行してアクセス可能か確認する
3. スリープ設定（`scripts/scrape_historical.py` の `MIN_SLEEP` / `MAX_SLEEP`）を適切な値に設定する
4. 既存データスキップ機能があるため、同じコマンドを再実行すれば中断箇所から再開される
5. 取得完了後は `README.md` と `SCRAPING_STATUS.md` を更新してプッシュする
