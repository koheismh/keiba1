"""実績記録モジュール

予測結果と実際結果の比較、的中・不的中の記録、
日次・週次・月次の実績回収率集計とレポート生成を行う。
記録はJSONファイルに永続化され、セッション間で保持される。
"""

import json
import logging
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any

from src.data.models import BetType

logger = logging.getLogger(__name__)


@dataclass
class BetRecord:
    """1つの買い目の記録"""

    date: date
    race_id: str
    bet_type: BetType
    combination: tuple[int, ...]
    amount: int
    is_hit: bool
    payout: int  # 0 if miss


class ResultRecorder:
    """実績記録クラス

    予測結果と実際の結果を比較し、的中・不的中を記録する。
    日次・週次・月次の実績回収率集計とレポート生成を行い、
    直近30日間の回収率が80%未満の場合はモデル再学習を推奨する。

    記録はJSONファイルに永続化され、プログラム再起動後も保持される。
    """

    def __init__(self, storage_path: Path | None = None) -> None:
        """初期化。

        Args:
            storage_path: 記録の保存先ファイルパス。
                Noneの場合は data/processed/results.json を使用。
        """
        if storage_path is None:
            self._storage_path = Path("data/processed/results.json")
        else:
            self._storage_path = storage_path
        self._records: list[BetRecord] = []
        self._load_from_disk()

    def record_result(self, bet_record: BetRecord) -> None:
        """1つの結果を記録する。

        記録はメモリに追加され、即座にディスクにも永続化される。

        Args:
            bet_record: 記録する買い目の結果
        """
        self._records.append(bet_record)
        self._save_to_disk()

    def get_records(self) -> list[BetRecord]:
        """全記録を返す。"""
        return list(self._records)

    def get_daily_return_rate(self, target_date: date) -> float:
        """指定日の回収率を返す。

        Args:
            target_date: 対象日

        Returns:
            回収率（払い戻し金額 / 投資金額）。投資がない場合は0.0。
        """
        daily_records = [r for r in self._records if r.date == target_date]
        return self._calculate_return_rate(daily_records)

    def get_weekly_return_rate(self, year: int, week: int) -> float:
        """指定週の回収率を返す。

        ISO週番号に基づいて集計する。

        Args:
            year: 年
            week: ISO週番号 (1-53)

        Returns:
            回収率（払い戻し金額 / 投資金額）。投資がない場合は0.0。
        """
        weekly_records = [
            r
            for r in self._records
            if r.date.isocalendar()[0] == year and r.date.isocalendar()[1] == week
        ]
        return self._calculate_return_rate(weekly_records)

    def get_monthly_return_rate(self, year: int, month: int) -> float:
        """指定月の回収率を返す。

        Args:
            year: 年
            month: 月 (1-12)

        Returns:
            回収率（払い戻し金額 / 投資金額）。投資がない場合は0.0。
        """
        monthly_records = [
            r for r in self._records if r.date.year == year and r.date.month == month
        ]
        return self._calculate_return_rate(monthly_records)

    def get_recent_return_rate(self, days: int = 30) -> float:
        """直近N日間の回収率を返す。

        Args:
            days: 対象日数（デフォルト30日）

        Returns:
            回収率（払い戻し金額 / 投資金額）。投資がない場合は0.0。
        """
        if not self._records:
            return 0.0

        latest_date = max(r.date for r in self._records)
        cutoff_date = latest_date - timedelta(days=days - 1)
        recent_records = [r for r in self._records if r.date >= cutoff_date]
        return self._calculate_return_rate(recent_records)

    def should_retrain(self) -> bool:
        """直近30日間の回収率が80%未満かどうかを判定する。

        Returns:
            True: 再学習推奨（回収率80%未満）
            False: 再学習不要（回収率80%以上、またはデータなし）
        """
        if not self._records:
            return False

        recent_rate = self.get_recent_return_rate(days=30)
        # データがない場合(0.0)も再学習不要とする
        if recent_rate == 0.0:
            return False
        return recent_rate < 0.8

    def get_retrain_notification(self) -> str | None:
        """モデル再学習推奨通知を返す。

        直近30日間の回収率が80%未満の場合に通知メッセージを返す。

        Returns:
            通知メッセージ。再学習不要の場合はNone。
        """
        if not self.should_retrain():
            return None

        recent_rate = self.get_recent_return_rate(days=30)
        return (
            f"[警告] 直近30日間の回収率が{recent_rate * 100:.1f}%です（基準: 80%）。"
            f"モデルの再学習を推奨します。"
        )

    def generate_report(self) -> str:
        """実績レポートを生成する。

        全記録に基づいて、全体サマリー、券種別成績、
        直近30日間の回収率、再学習推奨通知を含むレポートを返す。

        Returns:
            フォーマット済みレポート文字列
        """
        if not self._records:
            return "実績データがありません。"

        lines: list[str] = []
        lines.append("=" * 50)
        lines.append("実績レポート")
        lines.append("=" * 50)

        # 全体サマリー
        total_investment = sum(r.amount for r in self._records)
        total_payout = sum(r.payout for r in self._records)
        total_bets = len(self._records)
        total_hits = sum(1 for r in self._records if r.is_hit)
        overall_return_rate = (
            total_payout / total_investment if total_investment > 0 else 0.0
        )
        hit_rate = total_hits / total_bets if total_bets > 0 else 0.0

        lines.append("")
        lines.append("【全体サマリー】")
        lines.append(f"  総買い目数: {total_bets}")
        lines.append(f"  的中数: {total_hits}")
        lines.append(f"  的中率: {hit_rate * 100:.1f}%")
        lines.append(f"  合計投資金額: {total_investment:,}円")
        lines.append(f"  合計払い戻し: {total_payout:,}円")
        lines.append(f"  回収率: {overall_return_rate * 100:.1f}%")

        # 券種別成績
        lines.append("")
        lines.append("【券種別成績】")
        bet_type_stats = self._get_bet_type_stats()
        for bet_type, stats in bet_type_stats.items():
            lines.append(f"  {bet_type.value}:")
            lines.append(f"    買い目数: {stats['count']}")
            lines.append(f"    的中数: {stats['hits']}")
            lines.append(f"    的中率: {stats['hit_rate'] * 100:.1f}%")
            lines.append(f"    回収率: {stats['return_rate'] * 100:.1f}%")

        # 直近30日間
        lines.append("")
        lines.append("【直近30日間】")
        recent_rate = self.get_recent_return_rate(days=30)
        lines.append(f"  回収率: {recent_rate * 100:.1f}%")

        # 再学習推奨通知
        notification = self.get_retrain_notification()
        if notification:
            lines.append("")
            lines.append(notification)

        lines.append("")
        lines.append("=" * 50)

        return "\n".join(lines)

    def _calculate_return_rate(self, records: list[BetRecord]) -> float:
        """記録リストから回収率を計算する。

        Args:
            records: 対象の記録リスト

        Returns:
            回収率（払い戻し金額 / 投資金額）。投資がない場合は0.0。
        """
        if not records:
            return 0.0

        total_investment = sum(r.amount for r in records)
        if total_investment == 0:
            return 0.0

        total_payout = sum(r.payout for r in records)
        return total_payout / total_investment

    def _get_bet_type_stats(self) -> dict[BetType, dict[str, float]]:
        """券種別の統計情報を算出する。

        Returns:
            券種をキーとし、count, hits, hit_rate, return_rateを含む辞書。
        """
        stats: dict[BetType, dict[str, float]] = {}

        bet_types_in_records = sorted(
            set(r.bet_type for r in self._records), key=lambda bt: bt.value
        )

        for bet_type in bet_types_in_records:
            type_records = [r for r in self._records if r.bet_type == bet_type]
            count = len(type_records)
            hits = sum(1 for r in type_records if r.is_hit)
            total_investment = sum(r.amount for r in type_records)
            total_payout = sum(r.payout for r in type_records)

            stats[bet_type] = {
                "count": count,
                "hits": hits,
                "hit_rate": hits / count if count > 0 else 0.0,
                "return_rate": (
                    total_payout / total_investment if total_investment > 0 else 0.0
                ),
            }

        return stats

    def _save_to_disk(self) -> None:
        """全記録をJSONファイルに保存する。"""
        self._storage_path.parent.mkdir(parents=True, exist_ok=True)
        data = [self._record_to_dict(r) for r in self._records]
        try:
            with open(self._storage_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except OSError as e:
            logger.error(f"実績記録の保存に失敗: {e}")

    def _load_from_disk(self) -> None:
        """JSONファイルから記録を読み込む。"""
        if not self._storage_path.exists():
            return

        try:
            with open(self._storage_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                for item in data:
                    record = self._dict_to_record(item)
                    if record is not None:
                        self._records.append(record)
            logger.info(f"実績記録を読み込み: {len(self._records)}件")
        except (json.JSONDecodeError, OSError) as e:
            logger.warning(f"実績記録の読み込みに失敗: {e}")

    def _record_to_dict(self, record: BetRecord) -> dict[str, Any]:
        """BetRecordを辞書に変換する。"""
        return {
            "date": record.date.isoformat(),
            "race_id": record.race_id,
            "bet_type": record.bet_type.value,
            "combination": list(record.combination),
            "amount": record.amount,
            "is_hit": record.is_hit,
            "payout": record.payout,
        }

    def _dict_to_record(self, data: dict[str, Any]) -> BetRecord | None:
        """辞書からBetRecordを生成する。"""
        try:
            return BetRecord(
                date=date.fromisoformat(data["date"]),
                race_id=data["race_id"],
                bet_type=BetType(data["bet_type"]),
                combination=tuple(data["combination"]),
                amount=data["amount"],
                is_hit=data["is_hit"],
                payout=data["payout"],
            )
        except (KeyError, ValueError, TypeError) as e:
            logger.warning(f"記録のパースに失敗: {e}")
            return None
