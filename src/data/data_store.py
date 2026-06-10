"""データ永続化モジュール。

スクレイピングで取得したレースデータや実績記録をJSONファイルとして
ディスクに保存し、再利用可能にする。一度取得したデータはキャッシュとして
保持され、同じレースの再取得を防ぐ。
"""

import json
import logging
from datetime import date, time
from pathlib import Path
from typing import Any

from src.data.models import (
    BetType,
    HorseEntry,
    PayoutInfo,
    RaceData,
    RaceResult,
    TrackCondition,
)

logger = logging.getLogger(__name__)


class DataStore:
    """レースデータの永続化を管理するクラス。

    スクレイピング結果を data/raw/ にJSON形式で保存し、
    HistoricalDataLoader が読み込める形式で格納する。
    既に保存済みのデータがある場合は再取得をスキップする。
    """

    def __init__(self, data_dir: Path | None = None) -> None:
        """初期化。

        Args:
            data_dir: データ保存先ディレクトリ。Noneの場合は data/raw/ を使用。
        """
        if data_dir is None:
            self.data_dir = Path("data/raw")
        else:
            self.data_dir = data_dir
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def save_races(self, races: list[RaceData], race_date: date) -> Path:
        """レースデータをJSONファイルとして保存する。

        1日分のレースデータを1ファイルとして保存する。
        ファイル名は日付ベース（例: 2024-01-01.json）。

        Args:
            races: 保存対象のレースデータリスト。
            race_date: レース開催日。

        Returns:
            保存先ファイルのパス。
        """
        file_path = self._get_file_path(race_date)
        data = [self._race_to_dict(race) for race in races]

        file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        logger.info(f"レースデータを保存: {file_path} ({len(races)}件)")
        return file_path

    def save_single_race(self, race: RaceData) -> Path:
        """単一レースデータを既存ファイルに追記する。

        対象日のファイルが存在する場合は既存データに追加し、
        存在しない場合は新規ファイルを作成する。
        同じrace_idが既に存在する場合は上書きする。

        Args:
            race: 保存対象のレースデータ。

        Returns:
            保存先ファイルのパス。
        """
        file_path = self._get_file_path(race.race_date)
        existing_races = self._load_raw_list(file_path)

        # 同じrace_idのデータがあれば置換、なければ追加
        race_dict = self._race_to_dict(race)
        updated = False
        for i, existing in enumerate(existing_races):
            if existing.get("race_id") == race.race_id:
                existing_races[i] = race_dict
                updated = True
                break
        if not updated:
            existing_races.append(race_dict)

        file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(existing_races, f, ensure_ascii=False, indent=2)

        action = "更新" if updated else "追加"
        logger.info(f"レースデータを{action}: {race.race_id} -> {file_path}")
        return file_path

    def has_race_date(self, race_date: date) -> bool:
        """指定日のデータが既に保存されているか確認する。

        Args:
            race_date: 確認対象の日付。

        Returns:
            True: データが存在する。False: データが存在しない。
        """
        file_path = self._get_file_path(race_date)
        return file_path.exists() and file_path.stat().st_size > 0

    def has_race(self, race_id: str, race_date: date) -> bool:
        """指定レースIDのデータが既に保存されているか確認する。

        Args:
            race_id: 確認対象のレースID。
            race_date: レース開催日。

        Returns:
            True: データが存在する。False: データが存在しない。
        """
        file_path = self._get_file_path(race_date)
        if not file_path.exists():
            return False

        existing_races = self._load_raw_list(file_path)
        return any(r.get("race_id") == race_id for r in existing_races)

    def load_race_date(self, race_date: date) -> list[RaceData]:
        """指定日の保存済みレースデータを読み込む。

        Args:
            race_date: 読み込み対象の日付。

        Returns:
            保存済みのRaceDataリスト。ファイルが存在しない場合は空リスト。
        """
        file_path = self._get_file_path(race_date)
        if not file_path.exists():
            return []

        raw_list = self._load_raw_list(file_path)
        races: list[RaceData] = []
        for race_dict in raw_list:
            race = self._dict_to_race(race_dict)
            if race is not None:
                races.append(race)
        return races

    def get_stored_dates(self) -> list[date]:
        """保存済みの全日付を返す。

        Returns:
            保存済みデータが存在する日付のリスト（昇順）。
        """
        dates: list[date] = []
        for json_file in sorted(self.data_dir.glob("*.json")):
            try:
                date_str = json_file.stem  # "2024-01-01"
                d = date.fromisoformat(date_str)
                dates.append(d)
            except ValueError:
                continue
        return dates

    def _get_file_path(self, race_date: date) -> Path:
        """日付からファイルパスを生成する。

        Args:
            race_date: 対象日付。

        Returns:
            対応するJSONファイルのパス。
        """
        return self.data_dir / f"{race_date.isoformat()}.json"

    def _load_raw_list(self, file_path: Path) -> list[dict[str, Any]]:
        """JSONファイルから生データのリストを読み込む。

        Args:
            file_path: 読み込み対象のファイルパス。

        Returns:
            辞書のリスト。読み込み失敗時は空リスト。
        """
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                return data
            elif isinstance(data, dict):
                return [data]
            return []
        except (json.JSONDecodeError, OSError) as e:
            logger.warning(f"ファイル読み込みエラー: {file_path}: {e}")
            return []

    def _race_to_dict(self, race: RaceData) -> dict[str, Any]:
        """RaceDataオブジェクトを辞書に変換する。

        Args:
            race: 変換対象のRaceData。

        Returns:
            JSON互換の辞書。
        """
        entries = [
            {
                "horse_name": e.horse_name,
                "jockey_name": e.jockey_name,
                "gate_number": e.gate_number,
                "horse_number": e.horse_number,
                "weight": e.weight,
                "weight_change": e.weight_change,
                "win_odds": e.win_odds,
            }
            for e in race.entries
        ]

        results = None
        if race.results is not None:
            results = [
                {
                    "horse_number": r.horse_number,
                    "finish_position": r.finish_position,
                }
                for r in race.results
            ]

        payouts = None
        if race.payouts is not None:
            payouts = [
                {
                    "bet_type": p.bet_type.value,
                    "combination": list(p.combination),
                    "payout": p.payout,
                }
                for p in race.payouts
            ]

        post_time_str = None
        if race.post_time is not None:
            post_time_str = race.post_time.strftime("%H:%M")

        return {
            "race_id": race.race_id,
            "race_name": race.race_name,
            "race_date": race.race_date.isoformat(),
            "post_time": post_time_str,
            "venue": race.venue,
            "course_type": race.course_type,
            "distance": race.distance,
            "track_condition": race.track_condition.value,
            "weather": race.weather,
            "entries": entries,
            "results": results,
            "payouts": payouts,
        }

    def _dict_to_race(self, data: dict[str, Any]) -> RaceData | None:
        """辞書からRaceDataオブジェクトを生成する。

        Args:
            data: レースデータの辞書。

        Returns:
            パース成功時はRaceData、失敗時はNone。
        """
        try:
            entries = [
                HorseEntry(
                    horse_name=e["horse_name"],
                    jockey_name=e["jockey_name"],
                    gate_number=e["gate_number"],
                    horse_number=e["horse_number"],
                    weight=e.get("weight"),
                    weight_change=e.get("weight_change"),
                    win_odds=e.get("win_odds"),
                )
                for e in data.get("entries", [])
            ]

            results = None
            if data.get("results") is not None:
                results = [
                    RaceResult(
                        horse_number=r["horse_number"],
                        finish_position=r["finish_position"],
                    )
                    for r in data["results"]
                ]

            payouts = None
            if data.get("payouts") is not None:
                payouts = [
                    PayoutInfo(
                        bet_type=BetType(p["bet_type"]),
                        combination=tuple(p["combination"]),
                        payout=p["payout"],
                    )
                    for p in data["payouts"]
                ]

            post_time = None
            if data.get("post_time") is not None:
                parts = data["post_time"].split(":")
                post_time = time(int(parts[0]), int(parts[1]))

            track_condition = TrackCondition(data["track_condition"])

            return RaceData(
                race_id=data["race_id"],
                race_name=data["race_name"],
                race_date=date.fromisoformat(data["race_date"]),
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
