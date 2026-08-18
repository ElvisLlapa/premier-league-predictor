"""Tests for the cleaned and combined Premier League dataset."""

import pandas as pd
import pytest

from src.data_cleaner import PROCESSED_DATA_FILE, SEASON_FILES

EXPECTED_COLUMNS = [
    "season",
    "division",
    "date",
    "time",
    "home_team",
    "away_team",
    "home_goals",
    "away_goals",
    "result",
    "home_shots",
    "away_shots",
    "home_shots_on_target",
    "away_shots_on_target",
    "home_corners",
    "away_corners",
    "home_fouls",
    "away_fouls",
    "home_yellow_cards",
    "away_yellow_cards",
    "home_red_cards",
    "away_red_cards",
]


@pytest.fixture(scope="module")
def processed_data() -> pd.DataFrame:
    """Load the processed dataset once for all tests."""
    assert PROCESSED_DATA_FILE.exists()

    return pd.read_csv(
        PROCESSED_DATA_FILE,
        parse_dates=["date"],
    )


@pytest.mark.parametrize("season", SEASON_FILES)
def test_each_season_has_380_matches(
    processed_data: pd.DataFrame,
    season: str,
) -> None:
    """Verify that every season contains 380 matches."""
    season_data = processed_data[processed_data["season"] == season]

    assert len(season_data) == 380


def test_processed_dataset_structure(
    processed_data: pd.DataFrame,
) -> None:
    """Verify the combined dataset's size and columns."""
    assert len(processed_data) == 2280
    assert processed_data["season"].nunique() == 6
    assert list(processed_data.columns) == EXPECTED_COLUMNS


def test_processed_dataset_is_clean(
    processed_data: pd.DataFrame,
) -> None:
    """Verify that the processed data has no missing or duplicate rows."""
    assert not processed_data.isna().any().any()
    assert not processed_data.duplicated().any()


def test_processed_dataset_is_chronological(
    processed_data: pd.DataFrame,
) -> None:
    """Verify that match dates are in chronological order."""
    assert processed_data["date"].is_monotonic_increasing


def test_results_match_scores(
    processed_data: pd.DataFrame,
) -> None:
    """Verify that every result agrees with its final score."""
    expected_results = pd.Series("D", index=processed_data.index)

    expected_results.loc[
        processed_data["home_goals"] > processed_data["away_goals"]
    ] = "H"

    expected_results.loc[
        processed_data["home_goals"] < processed_data["away_goals"]
    ] = "A"

    assert processed_data["result"].equals(expected_results)
