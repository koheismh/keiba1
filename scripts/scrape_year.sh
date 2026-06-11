#!/bin/bash
# 1年分のデータを取得するスクリプト
# Usage: bash scripts/scrape_year.sh <year> <limit>
YEAR=$1
LIMIT=${2:-300}
echo "=== ${YEAR}年のデータを取得 (上限: ${LIMIT}レース) ==="
python3 scripts/scrape_historical.py --year $YEAR --months 1 2 3 4 5 6 7 8 9 10 11 12 --limit $LIMIT
echo "=== ${YEAR}年 完了 ==="
