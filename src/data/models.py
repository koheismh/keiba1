"""データモデル定義"""

from dataclasses import dataclass, field
from datetime import date, time
from enum import Enum
from typing import Optional

import numpy as np


class BetType(Enum):
    """券種"""

    WIN = "単勝"
    PLACE = "複勝"
    QUINELLA = "馬連"
    EXACTA = "馬単"
    WIDE = "ワイド"
    TRIO = "三連複"
    TRIFECTA = "三連単"


class TrackCondition(Enum):
    """馬場状態"""

    FIRM = "良"
    GOOD = "稍重"
    YIELDING = "重"
    SOFT = "不良"


@dataclass(frozen=True)
class HorseEntry:
    """出走馬情報"""

    horse_name: str  # 馬名
    jockey_name: str  # 騎手名
    gate_number: int  # 枠番 (1-8)
    horse_number: int  # 馬番
    weight: Optional[int]  # 馬体重（kg）
    weight_change: Optional[int]  # 馬体重変動（kg）
    win_odds: Optional[float]  # 単勝オッズ


@dataclass(frozen=True)
class RaceResult:
    """レース結果"""

    horse_number: int  # 馬番
    finish_position: int  # 着順


@dataclass(frozen=True)
class PayoutInfo:
    """払い戻し情報"""

    bet_type: BetType  # 券種
    combination: tuple[int, ...]  # 馬番の組み合わせ
    payout: int  # 払い戻し金額（円）


@dataclass(frozen=True)
class RaceData:
    """レースデータ"""

    race_id: str  # レースID
    race_name: str  # レース名
    race_date: date  # 開催日
    post_time: Optional[time]  # 発走時刻
    venue: str  # 開催場
    course_type: str  # コースタイプ（芝/ダート）
    distance: int  # 距離（メートル）
    track_condition: TrackCondition  # 馬場状態
    weather: Optional[str]  # 天候
    entries: list[HorseEntry]  # 出走馬リスト
    results: Optional[list[RaceResult]]  # レース結果（過去データのみ）
    payouts: Optional[list[PayoutInfo]]  # 払い戻し情報（過去データのみ）


@dataclass(frozen=True)
class FeatureVector:
    """特徴量ベクトル"""

    values: np.ndarray  # 特徴量の値
    feature_names: list[str]  # 特徴量名


@dataclass(frozen=True)
class BetRecommendation:
    """買い目推奨"""

    bet_type: BetType  # 券種
    combination: tuple[int, ...]  # 馬番の組み合わせ
    estimated_probability: float  # 推定的中確率
    estimated_odds: float  # 推定オッズ
    expected_value: float  # 期待値


@dataclass(frozen=True)
class AllocatedBet:
    """資金配分済み買い目"""

    recommendation: BetRecommendation  # 買い目推奨
    amount: int  # 投資金額（100円単位）


@dataclass(frozen=True)
class RaceEvaluation:
    """レース評価結果"""

    race_id: str  # レースID
    confidence_score: int  # 信頼度スコア (0-100)
    should_bet: bool  # 投資判定
    skip_reason: Optional[str]  # 見送り理由
    factors: dict[str, float]  # 評価要素（荒れやすさ、実力差、データ充実度）


@dataclass
class BacktestResult:
    """バックテスト結果"""

    total_races: int  # 対象レース数
    bet_races: int  # 投資レース数
    skipped_races: int  # 見送りレース数
    total_investment: int  # 合計投資金額
    total_return: int  # 合計払い戻し金額
    hit_rate: float  # 的中率
    return_rate: float  # 回収率
    max_drawdown: float  # 最大ドローダウン
    sharpe_ratio: float  # シャープレシオ
    daily_returns: list[float]  # 日次リターン
    weekly_returns: list[float]  # 週次リターン
    monthly_returns: list[float]  # 月次リターン
    bet_type_stats: dict[BetType, dict[str, float]]  # 券種別統計


@dataclass(frozen=True)
class Config:
    """システム設定"""

    confidence_threshold: int = 50  # 信頼度スコア閾値 (0-100)
    max_bets_per_race: int = 10  # 1レースあたり最大買い目数
    min_expected_value: float = 1.0  # 期待値最低基準
    daily_budget: int = 10000  # 1日の投資予算（円）
    target_bet_types: list[BetType] = field(
        default_factory=lambda: list(BetType)
    )  # 対象券種
    max_single_bet_ratio: float = 0.3  # 単一買い目の最大配分比率


@dataclass
class CleaningReport:
    """データクリーニングレポート"""

    total_records: int  # 総レコード数
    excluded_count: int  # 除外件数
    exclusion_reasons: list[tuple[str, str]]  # (レースID, 除外理由)
    clean_races: list[RaceData]  # クリーニング済みデータ
