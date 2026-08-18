"""Clean and combine the raw Premier League match datasets."""

from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"

PROCESSED_DATA_FILE = PROCESSED_DATA_DIR / "premier_league_matches.csv"

SEASON_FILES = {
    "2020-21": "PL_2020-21.csv",
    "2021-22": "PL_2021-22.csv",
    "2022-23": "PL_2022-23.csv",
    "2023-24": "PL_2023-24.csv",
    "2024-25": "PL_2024-25.csv",
    "2025-26": "PL_2025-26.csv",
}
COLUMN_RENAMES = {
    "Div": "division",
    "Date": "date",
    "Time": "time",
    "HomeTeam": "home_team",
    "AwayTeam": "away_team",
    "FTHG": "home_goals",
    "FTAG": "away_goals",
    "FTR": "result",
    "HS": "home_shots",
    "AS": "away_shots",
    "HST": "home_shots_on_target",
    "AST": "away_shots_on_target",
    "HC": "home_corners",
    "AC": "away_corners",
    "HF": "home_fouls",
    "AF": "away_fouls",
    "HY": "home_yellow_cards",
    "AY": "away_yellow_cards",
    "HR": "home_red_cards",
    "AR": "away_red_cards",
}

REQUIRED_RAW_COLUMNS = {
    "Date",
    "HomeTeam",
    "AwayTeam",
    "FTHG",
    "FTAG",
    "FTR",
}

NUMERIC_COLUMNS = [
    "home_goals",
    "away_goals",
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

TEAM_NAME_RENAMES = {
    "Bournemouth": "AFC Bournemouth",
    "Brighton": "Brighton & Hove Albion",
    "Leeds": "Leeds United",
    "Leicester": "Leicester City",
    "Luton": "Luton Town",
    "Man City": "Manchester City",
    "Man United": "Manchester United",
    "Newcastle": "Newcastle United",
    "Norwich": "Norwich City",
    "Nott'm Forest": "Nottingham Forest",
    "Tottenham": "Tottenham Hotspur",
    "West Ham": "West Ham United",
    "Wolves": "Wolverhampton Wanderers",
}


def load_raw_season(season: str, filename: str) -> pd.DataFrame:
    """Load one raw season without modifying its source CSV."""
    file_path = RAW_DATA_DIR / filename

    if not file_path.exists():
        raise FileNotFoundError(
            f"Raw data file not found: {file_path}\n"
            "Run `python -m src.data_loader` to download the raw data."
        )

    dataframe = pd.read_csv(file_path)

    print(f"Loaded {season}: {len(dataframe)} matches")

    return dataframe


def standardize_columns(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Select Version 1 columns and give them consistent names."""
    missing_columns = REQUIRED_RAW_COLUMNS - set(dataframe.columns)

    if missing_columns:
        missing_list = ", ".join(sorted(missing_columns))
        raise ValueError(f"Missing required raw columns: {missing_list}")

    selected_columns = [
        column for column in COLUMN_RENAMES if column in dataframe.columns
    ]

    standardized = dataframe.loc[:, selected_columns].rename(columns=COLUMN_RENAMES)

    return standardized


def convert_data_types(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Convert the date and match-statistic columns to suitable data types."""
    converted = dataframe.copy()

    converted["date"] = pd.to_datetime(
        converted["date"],
        format="%d/%m/%Y",
        errors="raise",
    )

    for column in NUMERIC_COLUMNS:
        if column in converted.columns:
            converted[column] = pd.to_numeric(
                converted[column],
                errors="coerce",
            )

    return converted


def standardize_team_names(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Remove extra whitespace and apply consistent team names."""
    standardized = dataframe.copy()

    for column in ["home_team", "away_team"]:
        standardized[column] = (
            standardized[column].str.strip().replace(TEAM_NAME_RENAMES)
        )

    return standardized


def clean_records(dataframe: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Check for missing values and remove duplicate rows."""
    cleaned = dataframe.copy()

    cleaned[["home_team", "away_team"]] = cleaned[["home_team", "away_team"]].replace(
        "", pd.NA
    )

    missing_counts = cleaned.isna().sum()
    missing_counts = missing_counts[missing_counts > 0]

    if not missing_counts.empty:
        missing_details = ", ".join(
            f"{column}={count}" for column, count in missing_counts.items()
        )
        raise ValueError(f"Missing values found: {missing_details}")

    duplicate_count = int(cleaned.duplicated().sum())
    cleaned = cleaned.drop_duplicates().reset_index(drop=True)

    return cleaned, duplicate_count


def validate_results(dataframe: pd.DataFrame) -> None:
    """Check that each result agrees with the final score."""
    expected_results = pd.Series("D", index=dataframe.index)

    expected_results.loc[dataframe["home_goals"] > dataframe["away_goals"]] = "H"

    expected_results.loc[dataframe["home_goals"] < dataframe["away_goals"]] = "A"

    incorrect_results = dataframe["result"] != expected_results
    incorrect_count = int(incorrect_results.sum())

    if incorrect_count > 0:
        raise ValueError(
            f"Found {incorrect_count} results that do not match the scores."
        )


def combine_seasons() -> pd.DataFrame:
    """Clean all seasons and combine them in date order."""
    season_dataframes = []

    for season, filename in SEASON_FILES.items():
        raw_dataframe = load_raw_season(season, filename)
        standardized_dataframe = standardize_columns(raw_dataframe)
        converted_dataframe = convert_data_types(standardized_dataframe)
        team_dataframe = standardize_team_names(converted_dataframe)
        cleaned_dataframe, duplicate_count = clean_records(team_dataframe)
        validate_results(cleaned_dataframe)

        cleaned_dataframe.insert(0, "season", season)
        season_dataframes.append(cleaned_dataframe)

        print(
            f"Prepared {season}: "
            f"{len(cleaned_dataframe)} matches, "
            f"{duplicate_count} duplicates removed"
        )

    combined_dataframe = pd.concat(
        season_dataframes,
        ignore_index=True,
    )

    combined_dataframe = combined_dataframe.sort_values(
        by=["date", "time", "home_team", "away_team"]
    ).reset_index(drop=True)

    return combined_dataframe


def save_processed_data(dataframe: pd.DataFrame) -> Path:
    """Save the cleaned dataset without changing the raw files."""
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    dataframe.to_csv(
        PROCESSED_DATA_FILE,
        index=False,
        date_format="%Y-%m-%d",
    )

    return PROCESSED_DATA_FILE


def main() -> None:
    """Combine and save all cleaned seasons."""
    combined_dataframe = combine_seasons()
    output_path = save_processed_data(combined_dataframe)

    print(
        f"Combined dataset: "
        f"{len(combined_dataframe)} matches, "
        f"{combined_dataframe['season'].nunique()} seasons"
    )
    print(
        f"Date range: "
        f"{combined_dataframe['date'].min().date()} to "
        f"{combined_dataframe['date'].max().date()}"
    )
    print(f"Saved processed dataset: {output_path}")


if __name__ == "__main__":
    main()
