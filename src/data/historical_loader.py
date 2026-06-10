"""過去レースデータの読み込みと前処理を行うモジュール。

Historical_Data_Loaderは過去のレースデータを読み込み、
学習・検証用に構造化するための機能を提供する。
"""

import json
import logging
from datetime import date
from pathlib import Path
from typing import Any

from src.data.models import (
    CleaningReport,
    HorseEntry,
    PayoutInfo,
    RaceData,
    RaceResult,
    TrackCondition,
    BetType,
)

logger = logging.getLogger(__name__)


class HistoricalDataLoader:
    """過去レースデータの読み込みと前処理を行う。

    data/raw/ ディレクトリからJSONファイルを読み込み、
    RaceDataオブジェクトに変換する。
    """

    def __init__(self, data_dir: Path | None = None) -> None:
        """初期化。

        Args:
            data_dir: データディレクトリのパス。Noneの場合はdata/raw/を使用。
        """
        if data_dir is None:
            self.data_dir = Path("data/raw")
        else:
            self.data_dir = data_dir

    def load_races(self, start_date: date, end_date: date) -> list[RaceData]:
        """指定期間のレースデータを読み込む。

        data/raw/ ディレクトリ内のJSONファイルからレースデータを読み込み、
        指定された日付範囲でフィルタリングして返す。

        Args:
            start_date: 開始日（この日を含む）
            end_date: 終了日（この日を含む）

        Returns:
            指定期間内のRaceDataオブジェクトのリスト（日付昇順）
        """
        if not self.data_dir.exists():
            logger.warning(f"データディレクトリが存在しません: {self.data_dir}")
            return []

        races: list[RaceData] = []

        for json_file in sorted(self.data_dir.glob("*.json")):
            try:
                with open(json_file, "r", encoding="utf-8") as f:
                    data = json.load(f)

                # JSONファイルがリストの場合は複数レース、辞書の場合は単一レース
                if isinstance(data, list):
                    for race_dict in data:
                        race = self._parse_race(race_dict)
                        if race is not None:
                            races.append(race)
                elif isinstance(data, dict):
                    race = self._parse_race(data)
                    if race is not None:
                        races.append(race)

            except (json.JSONDecodeError, OSError) as e:
                logger.warning(f"ファイル読み込みエラー: {json_file}: {e}")
                continue

        # 日付範囲でフィルタリング
        filtered = [
            race for race in races
            if start_date <= race.race_date <= end_date
        ]

        # 日付昇順でソート
        filtered.sort(key=lambda r: (r.race_date, r.race_id))

        return filtered

    def split_data(
        self, races: list[RaceData], train_ratio: float = 0.8
    ) -> tuple[list[RaceData], list[RaceData]]:
        """学習用と検証用にデータを時系列順に分割する。

        レースを日付順にソートし、前半を学習用、後半を検証用とする。

        Args:
            races: 分割対象のレースデータリスト
            train_ratio: 学習データの比率（デフォルト: 0.8）

        Returns:
            (学習用レースリスト, 検証用レースリスト)のタプル
        """
        if not races:
            return ([], [])

        # 時系列順にソート
        sorted_races = sorted(races, key=lambda r: (r.race_date, r.race_id))

        # 分割位置を計算
        split_index = int(len(sorted_races) * train_ratio)

        train_races = sorted_races[:split_index]
        validation_races = sorted_races[split_index:]

        logger.info(
            f"データ分割完了: 学習用={len(train_races)}件, "
            f"検証用={len(validation_races)}件"
        )

        return (train_races, validation_races)

    def validate_and_clean(self, races: list[RaceData]) -> CleaningReport:
        """データのバリデーションとクリーニングを行う。

        各レースレコードを検証し、不正なデータを除外する。
        除外されたレコードについては理由をログに出力する。

        バリデーションルール:
        - race_id, race_name, venue は空でないこと
        - entries（出走馬）が2頭以上であること
        - 各出走馬のhorse_numberが1以上であること
        - 各出走馬のgate_numberが1〜8の範囲であること
        - 各出走馬のhorse_name, jockey_nameが空でないこと
        - distanceが正の値であること

        Args:
            races: バリデーション対象のレースデータリスト

        Returns:
            CleaningReport（総レコード数、除外件数、除外理由、クリーン済みデータ）
        """
        total_records = len(races)
        clean_races: list[RaceData] = []
        exclusion_reasons: list[tuple[str, str]] = []

        for race in races:
            reason = self._validate_race(race)
            if reason is None:
                clean_races.append(race)
            else:
                exclusion_reasons.append((race.race_id, reason))
                logger.warning(
                    f"レース除外: race_id={race.race_id}, 理由={reason}"
                )

        excluded_count = len(exclusion_reasons)

        if excluded_count > 0:
            logger.info(
                f"データクリーニング完了: 総数={total_records}, "
                f"除外={excluded_count}, 有効={len(clean_races)}"
            )

        return CleaningReport(
            total_records=total_records,
            excluded_count=excluded_count,
            exclusion_reasons=exclusion_reasons,
            clean_races=clean_races,
        )

    def _validate_race(self, race: RaceData) -> str | None:
        """単一レースのバリデーションを行う。

        Args:
            race: バリデーション対象のレースデータ

        Returns:
            不正な場合は除外理由の文字列、正常な場合はNone
        """
        # 必須フィールドの空チェック
        if not race.race_id:
            return "race_idが空です"
        if not race.race_name:
            return "race_nameが空です"
        if not race.venue:
            return "venueが空です"

        # 距離チェック
        if race.distance <= 0:
            return f"distanceが不正です: {race.distance}"

        # 出走馬数チェック
        if len(race.entries) < 2:
            return f"出走馬が2頭未満です: {len(race.entries)}頭"

        # 各出走馬のバリデーション
        for entry in race.entries:
            entry_reason = self._validate_entry(entry)
            if entry_reason is not None:
                return entry_reason

        return None

    def _validate_entry(self, entry: HorseEntry) -> str | None:
        """出走馬エントリのバリデーションを行う。

        Args:
            entry: バリデーション対象の出走馬情報

        Returns:
            不正な場合は除外理由の文字列、正常な場合はNone
        """
        if entry.horse_number < 1:
            return f"horse_numberが不正です: {entry.horse_number}"
        if entry.gate_number < 1 or entry.gate_number > 8:
            return f"gate_numberが不正です: {entry.gate_number}"
        if not entry.horse_name:
            return "horse_nameが空です"
        if not entry.jockey_name:
            return "jockey_nameが空です"

        return None

    def _parse_race(self, data: dict[str, Any]) -> RaceData | None:
        """辞書データからRaceDataオブジェクトを生成する。

        Args:
            data: レースデータの辞書

        Returns:
            パース成功時はRaceDataオブジェクト、失敗時はNone
        """
        try:
            # 出走馬のパース
            entries = []
            for entry_data in data.get("entries", []):
                entry = HorseEntry(
                    horse_name=entry_data["horse_name"],
                    jockey_name=entry_data["jockey_name"],
                    gate_number=entry_data["gate_number"],
                    horse_number=entry_data["horse_number"],
                    weight=entry_data.get("weight"),
                    weight_change=entry_data.get("weight_change"),
                    win_odds=entry_data.get("win_odds"),
                )
                entries.append(entry)

            # レース結果のパース（存在する場合）
            results = None
            if "results" in data and data["results"] is not None:
                results = [
                    RaceResult(
                        horse_number=r["horse_number"],
                        finish_position=r["finish_position"],
                    )
                    for r in data["results"]
                ]

            # 払い戻し情報のパース（存在する場合）
            payouts = None
            if "payouts" in data and data["payouts"] is not None:
                payouts = [
                    PayoutInfo(
                        bet_type=BetType(p["bet_type"]),
                        combination=tuple(p["combination"]),
                        payout=p["payout"],
                    )
                    for p in data["payouts"]
                ]

            # 日付のパース
            race_date = date.fromisoformat(data["race_date"])

            # 発走時刻のパース（存在する場合）
            post_time = None
            if "post_time" in data and data["post_time"] is not None:
                from datetime import time as time_type
                parts = data["post_time"].split(":")
                post_time = time_type(int(parts[0]), int(parts[1]))

            # 馬場状態のパース
            track_condition = TrackCondition(data["track_condition"])

            return RaceData(
                race_id=data["race_id"],
                race_name=data["race_name"],
                race_date=race_date,
                post_time=post_time,
                venue=data["venue"],
                course_type=data.get("course_type", "芝"),
                distance=data["distance"],
                track_condition=track_condition,
                weather=data.get("weather"),
                entries=entries,
                results=results,
                payouts=payouts,
            )

        except (KeyError, ValueError, TypeError) as e:
            logger.warning(f"レースデータのパースに失敗: {e}")
            return None
