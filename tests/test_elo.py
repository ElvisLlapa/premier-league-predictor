import pandas as pd
import pytest

from src.elo import (
    INITIAL_RATING,
    add_elo_features,
    calculate_expected_scores,
    update_elo_ratings,
)


def test_expected_scores_add_to_one():
    home_expected, away_expected = calculate_expected_scores(
        1500,
        1500,
    )

    assert home_expected + away_expected == pytest.approx(1.0)


def test_home_advantage_increases_home_expectation():
    home_expected, away_expected = calculate_expected_scores(
        1500,
        1500,
    )

    assert home_expected > away_expected


def test_surprising_result_causes_larger_change():
    home_win_ratings = update_elo_ratings(1500, 1500, "H")
    away_win_ratings = update_elo_ratings(1500, 1500, "A")

    expected_home_win_change = home_win_ratings[0] - 1500
    surprising_away_win_change = away_win_ratings[1] - 1500

    assert surprising_away_win_change > expected_home_win_change


def test_first_match_uses_initial_ratings():
    matches = pd.DataFrame(
        [
            {
                "date": "2026-01-01",
                "home_team": "Arsenal",
                "away_team": "Chelsea",
                "result": "H",
            }
        ]
    )

    result = add_elo_features(matches)

    assert result.loc[0, "home_elo"] == INITIAL_RATING
    assert result.loc[0, "away_elo"] == INITIAL_RATING


def test_current_result_only_affects_following_match():
    matches = pd.DataFrame(
        [
            {
                "date": "2026-01-01",
                "home_team": "Arsenal",
                "away_team": "Chelsea",
                "result": "H",
            },
            {
                "date": "2026-01-08",
                "home_team": "Arsenal",
                "away_team": "Chelsea",
                "result": "D",
            },
        ]
    )

    result = add_elo_features(matches)

    assert result.loc[0, "home_elo"] == INITIAL_RATING
    assert result.loc[0, "away_elo"] == INITIAL_RATING
    assert result.loc[1, "home_elo"] > INITIAL_RATING
    assert result.loc[1, "away_elo"] < INITIAL_RATING