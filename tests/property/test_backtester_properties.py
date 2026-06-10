"""バックテスト会計不変量のプロパティベーステスト。

Feature: horse-race-predictor, Property 10: バックテスト会計不変量

Validates: Requirements 6.1, 6.4
"""

from datetime import date
from unittest.mock import MagicMock

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

from src.backtest.backtester import Backtester
from src.betting.bet_selector import BetSelector
from src.betting.fund_allocator import FundAllocator
from src.data.models import (
    Config,
    HorseEntry,
    PayoutInfo,
    BetType,
    RaceData,
    RaceResult,
    TrackCondition,
)
from src.evaluation.race_evaluator import RaceEvaluator
from src.features.engineer import FeatureEngineer


# --- Strategies ---


def horse_entry_strategy(horse_number: int) -> st.SearchStrategy[HorseEntry]:
    """指定馬番のHorseEntryを生成するストラテジ。"""
    return st.builds(
        HorseEntry,
        horse_name=st.just(f"Horse{horse_number}"),
        jockey_name=st.just(f"Jockey{horse_number}"),
        gate_number=st.just((horse_number - 1) % 8 + 1),
        horse_number=st.just(horse_number),
        weight=st.one_of(st.none(), st.integers(min_value=400, max_value=600)),
        weight_change=st.one_of(st.none(), st.integers(min_value=-20, max_value=20)),
        win_odds=st.one_of(st.none(), st.floats(min_value=1.0, max_value=200.0)),
    )


@st.composite
def race_data_strategy(draw: st.DrawFn) -> RaceData:
    """ランダムなRaceDataを生成するストラテジ。

    出走馬数を2〜12頭の範囲で変化させ、
    結果データと払い戻し情報を含むレースを生成する。
    """
    n_entries = draw(st.integers(min_value=2, max_value=12))
    entries = [draw(horse_entry_strategy(i + 1)) for i in range(n_entries)]

    # レース結果を生成（着順は1〜n_entries）
    finish_order = list(range(1, n_entries + 1))
    results = [
        RaceResult(horse_number=i + 1, finish_position=finish_order[i])
        for i in range(n_entries)
    ]

    # 払い戻し情報（単勝の払い戻し）
    payout_amount = draw(st.integers(min_value=100, max_value=10000))
    payouts = [
        PayoutInfo(bet_type=BetType.WIN, combination=(1,), payout=payout_amount),
    ]

    race_date = draw(st.dates(min_value=date(2020, 1, 1), max_value=date(2024, 12, 31)))
    race_id = draw(st.text(min_size=1, max_size=8, alphabet="abcdefghijklmnop0123456789"))

    return RaceData(
        race_id=race_id,
        race_name=f"Race {race_id}",
        race_date=race_date,
        post_time=None,
        venue="東京",
        course_type=draw(st.sampled_from(["芝", "ダート"])),
        distance=draw(st.sampled_from([1200, 1600, 2000, 2400, 3000])),
        track_condition=draw(st.sampled_from(list(TrackCondition))),
        weather=None,
        entries=entries,
        results=results,
        payouts=payouts,
    )


@st.composite
def race_list_strategy(draw: st.DrawFn) -> list[RaceData]:
    """1〜20件のランダムなレースリストを生成するストラテジ。"""
    n_races = draw(st.integers(min_value=1, max_value=20))
    races = [draw(race_data_strategy()) for _ in range(n_races)]
    return races


# --- Property Tests ---


class TestBacktestAccountingInvariant:
    """Property 10: バックテスト会計不変量

    Feature: horse-race-predictor, Property 10: バックテスト会計不変量

    For any バックテスト結果に対して、投資対象レース数 + 見送りレース数 = 全レース数であること。

    Validates: Requirements 6.1, 6.4
    """

    @settings(max_examples=100)
    @given(races=race_list_strategy())
    def test_bet_races_plus_skipped_equals_total(
        self,
        races: list[RaceData],
    ) -> None:
        """投資対象レース数 + 見送りレース数 = 全レース数を検証する。

        ランダムなレースリストに対してバックテストを実行し、
        会計不変量（bet_races + skipped_races == total_races）が
        常に成立することを確認する。

        **Validates: Requirements 6.1, 6.4**
        """
        # Backtesterを実コンポーネントで構成（信頼度閾値はランダム性を持たせず固定）
        feature_engineer = FeatureEngineer(historical_races=[])
        bet_selector = BetSelector(min_expected_value=0.5)
        fund_allocator = FundAllocator(max_single_bet_ratio=0.3)
        race_evaluator = RaceEvaluator(confidence_threshold=30)

        backtester = Backtester(
            feature_engineer=feature_engineer,
            bet_selector=bet_selector,
            fund_allocator=fund_allocator,
            race_evaluator=race_evaluator,
        )

        # モックモデル: 出走馬数に応じた確率分布を返す
        model = MagicMock()

        def mock_predict(features: np.ndarray) -> np.ndarray:
            n_horses = features.shape[0]
            # 一様に近い確率を返す
            probs = np.ones(n_horses) / n_horses
            # 少し偏りを持たせる（最初の馬を少し有利に）
            probs[0] += 0.1
            probs = probs / probs.sum()
            return probs

        model.predict_probabilities.side_effect = mock_predict

        config = Config(
            daily_budget=10000,
            max_bets_per_race=10,
            confidence_threshold=30,
        )

        # バックテスト実行
        result = backtester.run(races, model, config)

        # 不変量の検証: bet_races + skipped_races == total_races
        assert result.bet_races + result.skipped_races == result.total_races, (
            f"会計不変量が成立しない: "
            f"bet_races({result.bet_races}) + "
            f"skipped_races({result.skipped_races}) = "
            f"{result.bet_races + result.skipped_races} != "
            f"total_races({result.total_races})"
        )

        # total_racesが入力のレース数と一致することも検証
        assert result.total_races == len(races), (
            f"total_racesが入力レース数と一致しない: "
            f"total_races={result.total_races}, "
            f"len(races)={len(races)}"
        )
