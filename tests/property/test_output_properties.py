"""出力フォーマッタのプロパティベーステスト。

Feature: horse-race-predictor, Property 13: 予測結果出力の順序性と完全性

Validates: Requirements 8.1, 8.2, 8.3, 8.5
"""

import re
from datetime import date, time

from hypothesis import given, settings
from hypothesis import strategies as st

from src.data.models import (
    AllocatedBet,
    BetRecommendation,
    BetType,
    HorseEntry,
    RaceData,
    RaceEvaluation,
    TrackCondition,
)
from src.output.formatter import OutputFormatter, RacePrediction


# --- Strategies ---


@st.composite
def horse_entry_strategy(draw: st.DrawFn, horse_number: int) -> HorseEntry:
    """有効なHorseEntryを生成するストラテジ。"""
    return HorseEntry(
        horse_name=draw(st.text(min_size=1, max_size=8, alphabet=st.characters(categories=("L",)))),
        jockey_name=draw(st.text(min_size=1, max_size=8, alphabet=st.characters(categories=("L",)))),
        gate_number=draw(st.integers(min_value=1, max_value=8)),
        horse_number=horse_number,
        weight=draw(st.integers(min_value=400, max_value=600)),
        weight_change=draw(st.integers(min_value=-20, max_value=20)),
        win_odds=draw(st.floats(min_value=1.1, max_value=200.0, allow_nan=False, allow_infinity=False)),
    )


@st.composite
def race_data_strategy(draw: st.DrawFn, race_id: str) -> RaceData:
    """指定されたrace_idでRaceDataを生成するストラテジ。"""
    n_entries = draw(st.integers(min_value=3, max_value=12))
    entries = [draw(horse_entry_strategy(i + 1)) for i in range(n_entries)]

    return RaceData(
        race_id=race_id,
        race_name=draw(st.text(min_size=1, max_size=15, alphabet=st.characters(categories=("L",)))),
        race_date=draw(st.dates(min_value=date(2020, 1, 1), max_value=date(2024, 12, 31))),
        post_time=draw(st.sampled_from([time(10, 0), time(11, 30), time(14, 0), time(15, 30)])),
        venue=draw(st.sampled_from(["東京", "中山", "阪神", "京都", "中京"])),
        course_type=draw(st.sampled_from(["芝", "ダート"])),
        distance=draw(st.sampled_from([1200, 1600, 1800, 2000, 2400])),
        track_condition=draw(st.sampled_from(list(TrackCondition))),
        weather=None,
        entries=entries,
        results=None,
        payouts=None,
    )


@st.composite
def bet_recommendation_strategy(draw: st.DrawFn) -> BetRecommendation:
    """有効なBetRecommendationを生成するストラテジ。"""
    bet_type = draw(st.sampled_from(list(BetType)))
    # 券種に応じたcombinationのサイズ
    combo_size = {
        BetType.WIN: 1,
        BetType.PLACE: 1,
        BetType.QUINELLA: 2,
        BetType.EXACTA: 2,
        BetType.WIDE: 2,
        BetType.TRIO: 3,
        BetType.TRIFECTA: 3,
    }[bet_type]
    combination = tuple(
        draw(
            st.lists(
                st.integers(min_value=1, max_value=18),
                min_size=combo_size,
                max_size=combo_size,
                unique=True,
            )
        )
    )
    probability = draw(st.floats(min_value=0.01, max_value=0.5, allow_nan=False, allow_infinity=False))
    odds = draw(st.floats(min_value=1.5, max_value=100.0, allow_nan=False, allow_infinity=False))

    return BetRecommendation(
        bet_type=bet_type,
        combination=combination,
        estimated_probability=probability,
        estimated_odds=odds,
        expected_value=probability * odds,
    )


@st.composite
def allocated_bet_strategy(draw: st.DrawFn) -> AllocatedBet:
    """AllocatedBetを生成するストラテジ。"""
    recommendation = draw(bet_recommendation_strategy())
    amount = draw(st.integers(min_value=1, max_value=50)) * 100  # 100円単位
    return AllocatedBet(recommendation=recommendation, amount=amount)


@st.composite
def race_prediction_bet_strategy(draw: st.DrawFn, race_id: str) -> RacePrediction:
    """投資対象のRacePredictionを生成するストラテジ（should_bet=True）。"""
    race = draw(race_data_strategy(race_id))
    n_bets = draw(st.integers(min_value=1, max_value=5))
    allocated_bets = [draw(allocated_bet_strategy()) for _ in range(n_bets)]

    evaluation = RaceEvaluation(
        race_id=race_id,
        confidence_score=draw(st.integers(min_value=50, max_value=100)),
        should_bet=True,
        skip_reason=None,
        factors={"荒れやすさ": 0.3, "実力差": 0.5, "データ充実度": 0.8},
    )

    return RacePrediction(
        race=race,
        evaluation=evaluation,
        allocated_bets=allocated_bets,
    )


@st.composite
def race_prediction_skip_strategy(draw: st.DrawFn, race_id: str) -> RacePrediction:
    """見送りのRacePredictionを生成するストラテジ（should_bet=False）。"""
    race = draw(race_data_strategy(race_id))
    skip_reason = draw(st.sampled_from([
        "信頼度スコアが閾値未満",
        "出走頭数が少なくデータ不足",
        "馬場状態が不良で予測精度低下",
        "オッズ変動が大きく不安定",
    ]))

    evaluation = RaceEvaluation(
        race_id=race_id,
        confidence_score=draw(st.integers(min_value=0, max_value=49)),
        should_bet=False,
        skip_reason=skip_reason,
        factors={"荒れやすさ": 0.7, "実力差": 0.2, "データ充実度": 0.3},
    )

    return RacePrediction(
        race=race,
        evaluation=evaluation,
        allocated_bets=[],
    )


@st.composite
def multiple_race_predictions_strategy(draw: st.DrawFn) -> list[RacePrediction]:
    """複数レースのRacePredictionリストを生成するストラテジ。

    各レースにはユニークなrace_idとユニークなrace_nameを割り当て、
    投資/見送りをランダムに混合する。
    race_idはソート順が確認しやすいように数字ベースで生成する。
    """
    n_races = draw(st.integers(min_value=2, max_value=8))

    # ユニークなrace_idを生成（順序をシャッフルするため後でソートしないものを渡す）
    race_ids = [f"race_{i:03d}" for i in range(1, n_races + 1)]
    # シャッフルした順序で予測結果を生成
    shuffled_ids = draw(st.permutations(race_ids))

    predictions = []
    for idx, race_id in enumerate(shuffled_ids):
        should_bet = draw(st.booleans())
        if should_bet:
            prediction = draw(race_prediction_bet_strategy(race_id))
        else:
            prediction = draw(race_prediction_skip_strategy(race_id))
        # ユニークなレース名を付与（出力内での位置特定に必要）
        unique_race = RaceData(
            race_id=prediction.race.race_id,
            race_name=f"テストレース{race_id}",
            race_date=prediction.race.race_date,
            post_time=prediction.race.post_time,
            venue=prediction.race.venue,
            course_type=prediction.race.course_type,
            distance=prediction.race.distance,
            track_condition=prediction.race.track_condition,
            weather=prediction.race.weather,
            entries=prediction.race.entries,
            results=prediction.race.results,
            payouts=prediction.race.payouts,
        )
        prediction = RacePrediction(
            race=unique_race,
            evaluation=prediction.evaluation,
            allocated_bets=prediction.allocated_bets,
        )
        predictions.append(prediction)

    return predictions


# --- Property 13 Tests ---


class TestOutputOrderingAndCompleteness:
    """Property 13: 予測結果出力の順序性と完全性

    Feature: horse-race-predictor, Property 13: 予測結果出力の順序性と完全性

    For any 複数レースの予測結果に対して、出力はレース番号の昇順にソートされていること。
    かつ、各レースの出力には投資対象の場合は買い目リストが、見送りの場合は見送り理由と
    信頼度スコアが含まれること。

    Validates: Requirements 8.1, 8.2, 8.3, 8.5
    """

    @settings(max_examples=100)
    @given(predictions=multiple_race_predictions_strategy())
    def test_output_races_in_ascending_race_id_order(
        self, predictions: list[RacePrediction]
    ) -> None:
        """出力がレース番号（race_id）の昇順であることを検証する。

        Validates: Requirements 8.1
        """
        formatter = OutputFormatter()
        output = formatter.format_predictions(predictions)

        # race_idをソート済み順序で取得
        sorted_race_ids = sorted(p.race.race_id for p in predictions)

        # 出力内でrace_nameの出現順序を確認する
        # format_raceではレース名が【...】で囲まれて出力される
        sorted_predictions_by_id = sorted(predictions, key=lambda p: p.race.race_id)
        race_names_in_order = [p.race.race_name for p in sorted_predictions_by_id]

        # 各レース名が出力内に存在し、出現順序が昇順であることを確認
        positions = []
        for race_name in race_names_in_order:
            pos = output.find(f"【{race_name}】")
            assert pos != -1, (
                f"レース名が出力に含まれない: {race_name}\n"
                f"出力（先頭200文字）: {output[:200]}"
            )
            positions.append(pos)

        # 出現位置が単調増加であること（昇順にソートされている）
        for i in range(len(positions) - 1):
            assert positions[i] < positions[i + 1], (
                f"レースが昇順で出力されていない: "
                f"race_ids={sorted_race_ids}, "
                f"出現位置={positions}"
            )

    @settings(max_examples=100)
    @given(predictions=multiple_race_predictions_strategy())
    def test_bet_races_contain_bet_details(
        self, predictions: list[RacePrediction]
    ) -> None:
        """投資対象レース（should_bet=True）の出力に買い目情報が含まれることを検証する。

        Validates: Requirements 8.2
        """
        formatter = OutputFormatter()
        output = formatter.format_predictions(predictions)

        bet_predictions = [p for p in predictions if p.evaluation.should_bet and p.allocated_bets]

        for prediction in bet_predictions:
            # 該当レースのセクションを抽出
            race_section = self._extract_race_section(output, prediction.race.race_name)
            assert race_section is not None, (
                f"投資対象レースのセクションが見つからない: {prediction.race.race_name}"
            )

            # 買い目数の表示があること
            assert "買い目数" in race_section, (
                f"投資対象レースに買い目数が含まれない: {prediction.race.race_name}\n"
                f"セクション: {race_section[:200]}"
            )

            # 各買い目の券種名が含まれること
            for bet in prediction.allocated_bets:
                bet_type_name = bet.recommendation.bet_type.value
                assert bet_type_name in race_section, (
                    f"投資対象レースに券種名が含まれない: "
                    f"race={prediction.race.race_name}, "
                    f"bet_type={bet_type_name}\n"
                    f"セクション: {race_section[:300]}"
                )

            # 合計投資金額が含まれること
            assert "合計投資金額" in race_section, (
                f"投資対象レースに合計投資金額が含まれない: {prediction.race.race_name}"
            )

    @settings(max_examples=100)
    @given(predictions=multiple_race_predictions_strategy())
    def test_skipped_races_contain_skip_info(
        self, predictions: list[RacePrediction]
    ) -> None:
        """見送りレース（should_bet=False）の出力に「見送り」と見送り理由が含まれることを検証する。

        Validates: Requirements 8.3, 8.5
        """
        formatter = OutputFormatter()
        output = formatter.format_predictions(predictions)

        skip_predictions = [
            p for p in predictions if not p.evaluation.should_bet or not p.allocated_bets
        ]

        for prediction in skip_predictions:
            # 該当レースのセクションを抽出
            race_section = self._extract_race_section(output, prediction.race.race_name)
            assert race_section is not None, (
                f"見送りレースのセクションが見つからない: {prediction.race.race_name}"
            )

            # 「見送り」が含まれること
            assert "見送り" in race_section, (
                f"見送りレースに「見送り」が含まれない: {prediction.race.race_name}\n"
                f"セクション: {race_section[:200]}"
            )

            # 信頼度スコアが含まれること
            assert "信頼度スコア" in race_section, (
                f"見送りレースに信頼度スコアが含まれない: {prediction.race.race_name}\n"
                f"セクション: {race_section[:200]}"
            )

            # 見送り理由が含まれること（skip_reasonが設定されている場合）
            if prediction.evaluation.skip_reason:
                assert prediction.evaluation.skip_reason in race_section, (
                    f"見送りレースに見送り理由が含まれない: "
                    f"race={prediction.race.race_name}, "
                    f"skip_reason={prediction.evaluation.skip_reason}\n"
                    f"セクション: {race_section[:300]}"
                )

    def _extract_race_section(self, output: str, race_name: str) -> str | None:
        """出力文字列からレース名に対応するセクションを抽出する。

        レースセクションは「【レース名】」から次の「---」区切り線または末尾までとする。
        """
        marker = f"【{race_name}】"
        start = output.find(marker)
        if start == -1:
            return None

        # 次のレースセクションの開始（次の「【」）を探す
        next_race_start = output.find("【", start + len(marker))
        if next_race_start == -1:
            return output[start:]
        else:
            return output[start:next_race_start]
