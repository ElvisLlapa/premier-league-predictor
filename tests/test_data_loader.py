"""Tests for the raw Premier League datasets."""

from pathlib import Path

import pandas as pd
import pytest

from src.data_loader import RAW_DATA_DIR, SEASON_URLS

REQUIRED_COLUMNS = {
    "Date",
    "HomeTeam",
    "AwayTeam",
    "FTHG",
    "FTAG",
    "FTR",
}


@pytest.mark.parametrize("season", SEASON_URLS)
def test_raw_season_file(season: str) -> None:
    """Check that one raw season is complete and valid."""
    file_path = RAW_DATA_DIR / f"PL_{season}.csv"

    assert file_path.exists(), f"Missing file: {file_path}"

    matches = pd.read_csv(file_path)

    assert len(matches) == 380
    assert REQUIRED_COLUMNS.issubset(matches.columns)
    assert matches["HomeTeam"].nunique() == 20
    assert matches["AwayTeam"].nunique() == 20
    assert set(matches["FTR"].dropna().unique()) == {"H", "D", "A"}
    assert matches[list(REQUIRED_COLUMNS)].isna().sum().sum() == 0
    assert matches.duplicated().sum() == 0

    dates = pd.to_datetime(
        matches["Date"],
        dayfirst=True,
        errors="coerce",
    )
    assert dates.notna().all()


def test_six_seasons_are_configured() -> None:
    """Check that exactly six completed seasons are configured."""
    assert len(SEASON_URLS) == 6


def test_total_match_count() -> None:
    """Check the number of files and total matches."""
    files = sorted(Path(RAW_DATA_DIR).glob("PL_*.csv"))
    total_matches = sum(len(pd.read_csv(file_path)) for file_path in files)

    assert len(files) == 6
    assert total_matches == 2280
