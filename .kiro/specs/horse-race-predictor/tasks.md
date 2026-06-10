# Implementation Plan: 競馬レース予測システム

## Overview

本実装計画は、Python + LightGBMベースの競馬レース予測アドバイスシステムを段階的に構築する。Phase 1（バックテスト）とPhase 2（本番運用）を含む全モジュールをインクリメンタルに開発し、各ステップで動作確認を行う。

## Tasks

- [x] 1. プロジェクト構造とコアインターフェースのセットアップ
  - [x] 1.1 プロジェクト初期化とディレクトリ構造作成
    - `pyproject.toml` を作成し、依存関係（lightgbm, pandas, numpy, scikit-learn, hypothesis, pytest, pyyaml, requests, beautifulsoup4）を定義する
    - 設計書に記載のディレクトリ構成（`src/`, `tests/`, `data/`, `config/`）を作成する
    - `src/__init__.py` と各サブパッケージの `__init__.py` を作成する
    - _Requirements: 全体_

  - [x] 1.2 データモデル定義
    - `src/data/models.py` に設計書記載のすべてのデータクラス（`BetType`, `TrackCondition`, `HorseEntry`, `RaceData`, `RaceResult`, `PayoutInfo`, `FeatureVector`, `BetRecommendation`, `AllocatedBet`, `RaceEvaluation`, `BacktestResult`, `Config`, `CleaningReport`）を実装する
    - _Requirements: 1.4, 4.3, 5.2, 6.2_

  - [x] 1.3 例外クラス定義
    - `src/exceptions.py` に設計書記載の例外クラス階層（`HorseRacePredictorError`, `DataFetchError`, `DataValidationError`, `ConfigError`, `ModelError`）を実装する
    - _Requirements: 1.3, 7.3, 10.6_

  - [x] 1.4 設定管理モジュール実装
    - `src/config.py` に `ConfigManager` クラスを実装する
    - `config/default.yaml` にデフォルト設定ファイルを作成する
    - YAML読み込み、バリデーション（有効範囲チェック）、`Config` データクラスへの変換を実装する
    - 有効範囲外の設定値に対してエラーメッセージと有効範囲を提示する
    - _Requirements: 10.1, 10.2, 10.3, 10.4, 10.5, 10.6_

  - [x] 1.5 設定バリデーションのプロパティテスト
    - **Property 15: 設定値バリデーション**
    - `tests/property/test_config_properties.py` にHypothesisで有効範囲外の設定値を自動生成し、バリデーション失敗を検証する
    - **Validates: Requirements 10.6**

- [x] 2. チェックポイント - プロジェクト基盤確認
  - Ensure all tests pass, ask the user if questions arise.

- [x] 3. 過去データ読み込みモジュール実装
  - [x] 3.1 Historical_Data_Loader 実装
    - `src/data/historical_loader.py` に `HistoricalDataLoader` クラスを実装する
    - `load_races(start_date, end_date)`: 指定期間の過去レースデータをファイルまたはDBから読み込む
    - `split_data(races, train_ratio)`: 学習用（80%）と検証用（20%）にデータ分割する
    - `validate_and_clean(races)`: フォーマット不正・欠損データを除外し、除外件数と理由をログ出力する `CleaningReport` を返す
    - _Requirements: 1.1, 1.2, 1.3, 1.4_

  - [x] 3.5 データ永続化モジュール実装
    - `src/data/data_store.py` に `DataStore` クラスを実装する
    - `save_races(races, race_date)`: 日付単位でレースデータをJSONファイルに保存する
    - `save_single_race(race)`: 個別レースを追加・更新保存する
    - `has_race_date(race_date)` / `has_race(race_id, race_date)`: 保存済みデータの存在確認（キャッシュ判定）
    - `load_race_date(race_date)`: 保存済みデータの読み込み
    - `get_stored_dates()`: 保存済み日付一覧の取得
    - `RaceDataFetcher` にキャッシュ連携機能を追加（data_store引数、取得前のキャッシュ確認、取得後の自動保存）
    - `ResultRecorder` にJSON永続化機能を追加（記録時の即時保存、起動時の自動読み込み）
    - _Requirements: 1b.1, 1b.2, 1b.3, 1b.4, 1b.5, 1b.6_

  - [x] 3.2 データ読み込みのプロパティテスト
    - **Property 1: 日付範囲フィルタリング**
    - `tests/property/test_data_loader_properties.py` に日付範囲指定で返されるデータが必ず範囲内であることを検証する
    - **Validates: Requirements 1.1**

  - [x] 3.3 データ分割のプロパティテスト
    - **Property 2: データ分割不変量**
    - 分割後の学習用データ + 検証用データ = 元データ件数であること、比率が指定に近似することを検証する
    - **Validates: Requirements 1.2**

  - [x] 3.4 データクリーニングのプロパティテスト
    - **Property 3: 不正データ除外の完全性**
    - クリーニング後データに不正レコードが含まれず、除外件数がレポートと一致することを検証する
    - **Validates: Requirements 1.3**

- [x] 4. 特徴量エンジニアリングモジュール実装
  - [x] 4.1 Feature_Engineer 実装
    - `src/features/engineer.py` に `FeatureEngineer` クラスを実装する
    - 以下の特徴量を抽出する: 過去成績、騎手成績、コース適性、距離適性、馬場状態適性、枠順、馬体重変動、クラス実績
    - `extract_features(race, horse)`: 個別馬の特徴量ベクトルを返す
    - `get_feature_names()`: 使用する特徴量名のリストを返す
    - _Requirements: 2.2_

  - [x] 4.2 Feature_Engineer のユニットテスト
    - `tests/unit/test_feature_engineer.py` に各特徴量の抽出が正しく動作することをテストする
    - データ不足時（新馬等）のフォールバック処理を検証する
    - _Requirements: 2.2, 2.4_

- [x] 5. 予測モデル実装
  - [x] 5.1 Prediction_Model 実装
    - `src/prediction/model.py` に `PredictionModel` クラスを実装する
    - LightGBMを使用し `train(features, labels)` でモデル学習を行う
    - `predict_probabilities(race_features)` で各馬の着順確率を推定する（確率は[0,1]、合計≈1.0）
    - `save(path)` / `load(path)` でモデルの保存・読み込みを行う
    - _Requirements: 2.1, 2.3_

  - [x] 5.2 交差検証実装
    - `src/prediction/trainer.py` に `ModelTrainer` クラスを実装する
    - `cross_validate(features, labels, n_splits)` でK分割交差検証を実行し、過学習を防止する
    - `CrossValidationResult` を返す
    - _Requirements: 2.5_

  - [x] 5.3 確率分布のプロパティテスト
    - **Property 4: 確率分布の妥当性**
    - `tests/property/test_prediction_properties.py` に予測確率が[0,1]範囲内かつ合計≈1.0であることを検証する
    - **Validates: Requirements 2.1**

  - [x] 5.4 交差検証のプロパティテスト
    - **Property 12: 交差検証の網羅性**
    - 各データポイントがちょうど1回検証データとして使用されることを検証する
    - **Validates: Requirements 2.5**

- [x] 6. レース評価・見送り判定モジュール実装
  - [x] 6.1 Race_Evaluator 実装
    - `src/evaluation/race_evaluator.py` に `RaceEvaluator` クラスを実装する
    - `evaluate(race, predictions)`: 0〜100の信頼度スコアを算出する
    - スコアが閾値以下の場合、見送り判定（`should_bet=False`）を返す
    - 見送り理由（荒れやすさ、実力差の明確さ、データの充実度）を含む`RaceEvaluation`を生成する
    - 全レースが閾値以下の場合は全レース見送りを判定する
    - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.5_

  - [x] 6.2 信頼度スコアのプロパティテスト
    - **Property 6: 信頼度スコアの範囲と閾値判定**
    - `tests/property/test_evaluator_properties.py` にスコアが[0,100]であること、閾値判定ロジックの正しさを検証する
    - **Validates: Requirements 3.1, 3.2, 3.3, 3.4**

- [x] 7. 買い目選択モジュール実装
  - [x] 7.1 Bet_Selector 実装
    - `src/betting/bet_selector.py` に `BetSelector` クラスを実装する
    - `select_bets(race, probabilities, max_bets)`: 期待値 > 閾値の買い目を最大max_bets件選択する
    - `calculate_expected_value(probability, odds)`: 期待値 = p × odds を計算する
    - 複数券種（単勝、複勝、馬連、馬単、ワイド、三連複、三連単）から最も期待値の高い組み合わせを選択する
    - 期待値1.0超の買い目がない場合、当該レースを見送り対象とする
    - 買い目は期待値の降順でソートして返す
    - _Requirements: 4.1, 4.2, 4.3, 4.4, 4.5, 4.6_

  - [x] 7.2 期待値計算のプロパティテスト
    - **Property 5: 期待値計算の正確性**
    - `tests/property/test_bet_selector_properties.py` に期待値が p × odds と一致することを検証する
    - **Validates: Requirements 2.3**

  - [x] 7.3 買い目選択制約のプロパティテスト
    - **Property 7: 買い目選択の制約**
    - 件数が最大買い目数以下、全買い目の期待値が最低基準超、期待値降順ソートを検証する
    - **Validates: Requirements 4.1, 4.2, 4.3, 4.4, 4.6**

- [x] 8. 資金配分モジュール実装
  - [x] 8.1 Fund_Allocator 実装
    - `src/betting/fund_allocator.py` に `FundAllocator` クラスを実装する
    - `allocate(bets, budget)`: 予算内で最適な資金配分を算出する
    - `apply_kelly_criterion(probability, odds, budget)`: ケリー基準に基づく推奨投資金額を100円単位で算出する
    - `cap_allocation(allocations, budget, max_ratio)`: 単一買い目の配分額が予算の30%を超える場合に制限し再配分する
    - 全買い目合計が予算を超えないことを保証する
    - _Requirements: 5.1, 5.2, 5.3, 5.4, 5.5_

  - [x] 8.2 資金配分不変量のプロパティテスト
    - **Property 8: 資金配分不変量**
    - `tests/property/test_fund_allocator_properties.py` に合計≤予算、各金額が100円単位、単一買い目≤30%を検証する
    - **Validates: Requirements 5.1, 5.2, 5.3, 5.5**

  - [x] 8.3 ケリー基準のプロパティテスト
    - **Property 9: ケリー基準の正確性**
    - ケリー基準の配分率が (p × odds - 1) / (odds - 1) と一致すること（負の場合は0）を検証する
    - **Validates: Requirements 5.4**

- [x] 9. チェックポイント - Phase 1 コアロジック確認
  - Ensure all tests pass, ask the user if questions arise.

- [x] 10. バックテストモジュール実装
  - [x] 10.1 Backtester 実装
    - `src/backtest/backtester.py` に `Backtester` クラスを実装する
    - `run(races, model, config)`: 指定期間の全レースに対してモデル適用・仮想馬券購入シミュレーションを実行する
    - `generate_report(result)`: 的中率、回収率、最大ドローダウン、シャープレシオ、日次/週次/月次回収率推移、券種別統計を含むレポートを生成する
    - 回収率100%未満時のパラメータ調整推奨事項を提示する
    - 検証用vs学習用の回収率比較による過学習判定を行う
    - _Requirements: 6.1, 6.2, 6.3, 6.4, 6.5, 6.6_

  - [x] 10.2 バックテスト会計のプロパティテスト
    - **Property 10: バックテスト会計不変量**
    - `tests/property/test_backtester_properties.py` に投資対象レース数 + 見送りレース数 = 全レース数を検証する
    - **Validates: Requirements 6.1, 6.4**

  - [x] 10.3 期間集計のプロパティテスト
    - **Property 11: 期間集計の正確性**
    - 週次リターンが日次リターンの合計と一致し、月次リターンが週次の合計と一致することを検証する
    - **Validates: Requirements 6.3, 9.2**

  - [x] 10.4 的中判定ロジック実装
    - `src/backtest/backtester.py` 内に券種ごとの的中判定ロジックを実装する
    - 単勝: 1着一致、複勝: 3着以内一致、馬連: 1-2着の組み合わせ一致、馬単: 1-2着の順序一致、ワイド: 3着以内2頭の組み合わせ、三連複: 1-3着の組み合わせ一致、三連単: 1-3着の順序一致
    - _Requirements: 9.1_

  - [x] 10.5 的中判定のプロパティテスト
    - **Property 14: 的中判定の正確性**
    - `tests/property/test_hit_judgment_properties.py` に各券種の的中判定が正しいことを検証する
    - **Validates: Requirements 9.1**

- [x] 11. Phase 2: 当日データ取得モジュール実装
  - [x] 11.1 Race_Data_Fetcher 実装
    - `src/data/race_fetcher.py` に `RaceDataFetcher` クラスを実装する
    - `fetch_race_day(race_date)`: 指定日の全レース情報（出走馬、騎手、枠順、オッズ、馬体重、天候、馬場状態）を取得する
    - `fetch_realtime_odds(race_id)`: 直前オッズ・馬体重変動情報を取得する
    - データ取得失敗時はリトライ（最大3回）後、エラーメッセージと失敗レース情報を返す
    - 30秒タイムアウト設定を行う
    - _Requirements: 7.1, 7.2, 7.3, 7.4_

  - [x] 11.2 Race_Data_Fetcher のユニットテスト
    - `tests/unit/test_race_fetcher.py` にモックHTTPレスポンスでデータ取得・エラー処理・リトライ動作を検証する
    - _Requirements: 7.1, 7.2, 7.3, 7.4_

- [x] 12. 出力フォーマッタと実績記録モジュール実装
  - [x] 12.1 Output_Formatter 実装
    - `src/output/formatter.py` に `OutputFormatter` クラスを実装する
    - 予測結果をレース番号順に整理して出力する
    - 各レースについてレース名、発走時刻、買い目リスト（券種、馬番組み合わせ、推奨投資金額、期待値）、見送り判定結果を含める
    - 1日全体のサマリー（投資対象レース数、見送りレース数、合計投資金額、期待回収率）を出力する
    - _Requirements: 8.1, 8.2, 8.3, 8.4, 8.5_

  - [x] 12.2 出力順序のプロパティテスト
    - **Property 13: 予測結果出力の順序性と完全性**
    - `tests/property/test_output_properties.py` に出力がレース番号昇順であること、各レースに必要情報が含まれることを検証する
    - **Validates: Requirements 8.1, 8.2, 8.3, 8.5**

  - [x] 12.3 Result_Recorder 実装
    - `src/output/result_recorder.py` に `ResultRecorder` クラスを実装する
    - 予測結果と実際結果の比較・的中不的中の記録
    - 日次・週次・月次の実績回収率集計とレポート生成
    - 直近30日間の実績回収率が80%未満時にモデル再学習推奨通知を出力する
    - _Requirements: 9.1, 9.2, 9.3, 9.4_

- [x] 13. CLIエントリーポイントとパイプライン統合
  - [x] 13.1 CLI 実装
    - `src/cli.py` にCLIエントリーポイントを実装する
    - サブコマンド: `backtest`（バックテスト実行）、`predict`（当日予測）、`record`（実績記録）、`retrain`（モデル再学習）
    - 各サブコマンドが対応するモジュールを呼び出し、パイプライン全体を統合する
    - _Requirements: 6.1, 7.1, 8.1, 9.3_

  - [x] 13.2 予測パイプライン統合
    - データ取得 → 特徴量抽出 → 予測 → レース評価 → 買い目選択 → 資金配分 → 出力のフルパイプラインを接続する
    - 各モジュール間のデータ受け渡しを実装する
    - エラー発生時のグレースフルデグラデーション（一部レース失敗時も他レースは処理継続）を実装する
    - _Requirements: 7.1, 8.1, 8.2, 8.5_

  - [x] 13.3 統合テスト
    - `tests/integration/test_pipeline.py` にモックデータを使った予測パイプライン全体の統合テストを作成する
    - バックテストの端から端までの実行テストを作成する
    - _Requirements: 6.1, 8.1_

- [x] 14. 最終チェックポイント - 全テスト通過確認
  - Ensure all tests pass, ask the user if questions arise.

## Notes

- `*` マーク付きタスクはオプションであり、MVP構築時にスキップ可能
- 各タスクは特定の要件に対応しており、トレーサビリティを確保している
- チェックポイントで段階的にバリデーションを行う
- プロパティテストはCorrectnessProperties（全15件）に基づく普遍的な正当性検証
- ユニットテストはプロパティテストでカバーしきれない具体的シナリオを検証
- 外部API（netkeiba）への依存はモックで代替してテスト
- Phase 1（バックテスト）が安定した後にPhase 2（本番運用）の開発に進む

## Task Dependency Graph

```json
{
  "waves": [
    { "id": 0, "tasks": ["1.1"] },
    { "id": 1, "tasks": ["1.2", "1.3"] },
    { "id": 2, "tasks": ["1.4"] },
    { "id": 3, "tasks": ["1.5", "3.1"] },
    { "id": 4, "tasks": ["3.2", "3.3", "3.4", "4.1"] },
    { "id": 5, "tasks": ["4.2", "5.1"] },
    { "id": 6, "tasks": ["5.2", "5.3", "6.1"] },
    { "id": 7, "tasks": ["5.4", "6.2", "7.1"] },
    { "id": 8, "tasks": ["7.2", "7.3", "8.1"] },
    { "id": 9, "tasks": ["8.2", "8.3", "10.1", "10.4"] },
    { "id": 10, "tasks": ["10.2", "10.3", "10.5", "11.1"] },
    { "id": 11, "tasks": ["11.2", "12.1"] },
    { "id": 12, "tasks": ["12.2", "12.3"] },
    { "id": 13, "tasks": ["13.1"] },
    { "id": 14, "tasks": ["13.2"] },
    { "id": 15, "tasks": ["13.3"] }
  ]
}
```
