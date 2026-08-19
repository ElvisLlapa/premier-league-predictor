import pandas as pd
import pytest

from src.feature_engineer import (
    add_five_match_form,
    add_previous_match_features,
    add_rest_days,
    add_season_points,
    add_venue_form_features,
    build_match_feature_table,
    calculate_match_outcomes,
    create_team_match_records,
)


@pytest.fixture
def sample_matches() -> pd.DataFrame:
    """Create a small chronological dataset for feature tests."""
    return pd.DataFrame(
        {
            "season": ["2020-21", "2020-21", "2020-21", "2021-22"],
            "division": ["E0", "E0", "E0", "E0"],
            "date": pd.to_datetime(
                [
                    "2020-09-12",
                    "2020-09-19",
                    "2020-09-26",
                    "2021-08-14",
                ]
            ),
            "time": ["15:00", "15:00", "15:00", "15:00"],
            "home_team": [
                "Arsenal",
                "Liverpool",
                "Arsenal",
                "Arsenal",
            ],
            "away_team": [
                "Chelsea",
                "Arsenal",
                "Everton",
                "Fulham",
            ],
            "home_goals": [2, 1, 0, 3],
            "away_goals": [0, 1, 1, 0],
            "result": ["H", "D", "A", "H"],
            "home_shots": [12, 10, 8, 14],
            "away_shots": [7, 9, 10, 6],
            "home_shots_on_target": [6, 4, 3, 7],
            "away_shots_on_target": [2, 4, 5, 2],
            "home_corners": [5, 6, 4, 7],
            "away_corners": [3, 4, 5, 2],
            "home_fouls": [10, 11, 9, 8],
            "away_fouls": [12, 10, 11, 13],
            "home_yellow_cards": [1, 2, 1, 0],
            "away_yellow_cards": [2, 1, 2, 2],
            "home_red_cards": [0, 0, 0, 0],
            "away_red_cards": [0, 0, 0, 0],
        }
    )


@pytest.fixture
def team_records(sample_matches: pd.DataFrame) -> pd.DataFrame:
    """Run all team-level feature steps on the sample matches."""
    records = create_team_match_records(sample_matches)
    records = calculate_match_outcomes(records)
    records = add_previous_match_features(records)
    records = add_five_match_form(records)
    records = add_venue_form_features(records)
    records = add_season_points(records)
    return add_rest_days(records)


def test_creates_two_team_records_per_match(
    sample_matches: pd.DataFrame,
) -> None:
    records = create_team_match_records(sample_matches)

    assert len(records) == len(sample_matches) * 2
    assert records.groupby("match_id").size().eq(2).all()


def test_calculates_team_outcomes(
    team_records: pd.DataFrame,
) -> None:
    arsenal_first_match = team_records[
        (team_records["match_id"] == 0)
        & (team_records["team"] == "Arsenal")
    ].iloc[0]

    assert arsenal_first_match["team_result"] == "W"
    assert arsenal_first_match["points"] == 3
    assert arsenal_first_match["goal_difference"] == 2


def test_previous_points_exclude_current_match(
    team_records: pd.DataFrame,
) -> None:
    arsenal = team_records[
        team_records["team"] == "Arsenal"
    ].reset_index(drop=True)

    assert pd.isna(arsenal.loc[0, "previous_points"])
    assert arsenal.loc[1, "previous_points"] == 3
    assert arsenal.loc[2, "previous_points"] == 1


def test_five_match_form_excludes_current_match(
    team_records: pd.DataFrame,
) -> None:
    arsenal = team_records[
        team_records["team"] == "Arsenal"
    ].reset_index(drop=True)

    assert arsenal.loc[2, "last_5_points"] == 4
    assert arsenal.loc[2, "last_5_wins"] == 1
    assert arsenal.loc[2, "last_5_draws"] == 1


def test_venue_form_uses_same_venue_only(
    team_records: pd.DataFrame,
) -> None:
    arsenal_third_match = team_records[
        (team_records["match_id"] == 2)
        & (team_records["team"] == "Arsenal")
    ].iloc[0]

    assert arsenal_third_match["venue"] == "home"
    assert arsenal_third_match["venue_last_5_points"] == 3
    assert arsenal_third_match["venue_matches_available_last_5"] == 1


def test_season_points_reset_for_new_season(
    team_records: pd.DataFrame,
) -> None:
    arsenal_new_season = team_records[
        (team_records["match_id"] == 3)
        & (team_records["team"] == "Arsenal")
    ].iloc[0]

    assert arsenal_new_season["season_points_before_match"] == 0
    assert arsenal_new_season["season_matches_before_match"] == 0


def test_calculates_rest_days(
    team_records: pd.DataFrame,
) -> None:
    arsenal_second_match = team_records[
        (team_records["match_id"] == 1)
        & (team_records["team"] == "Arsenal")
    ].iloc[0]

    assert arsenal_second_match["rest_days"] == 7


def test_builds_one_feature_row_per_match(
    sample_matches: pd.DataFrame,
    team_records: pd.DataFrame,
) -> None:
    features = build_match_feature_table(
        sample_matches,
        team_records,
    )

    assert len(features) == len(sample_matches)
    assert features["match_id"].is_unique
    assert features.loc[0, "home_last_5_points"] == 0
    assert features.loc[1, "away_last_5_points"] == 3
    assert features.loc[0, "result"] == "H"