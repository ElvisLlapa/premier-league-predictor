from pathlib import Path

import pandas as pd

PROCESSED_DATA_PATH = Path("data/processed/premier_league_matches.csv")

FEATURE_DATA_PATH = Path("data/processed/premier_league_features.csv")

REQUIRED_COLUMNS = [
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

TEAM_RECORD_COLUMNS = [
    "match_id",
    "season",
    "date",
    "time",
    "team",
    "opponent",
    "venue",
    "goals_for",
    "goals_against",
    "shots_for",
    "shots_against",
    "shots_on_target_for",
    "shots_on_target_against",
    "corners_for",
    "corners_against",
    "fouls_for",
    "fouls_against",
    "yellow_cards_for",
    "yellow_cards_against",
    "red_cards_for",
    "red_cards_against",
]

HISTORICAL_COLUMNS = [
    "points",
    "win",
    "draw",
    "loss",
    "goals_for",
    "goals_against",
    "goal_difference",
    "shots_for",
    "shots_against",
    "shots_on_target_for",
    "shots_on_target_against",
    "corners_for",
    "corners_against",
    "fouls_for",
    "fouls_against",
    "yellow_cards_for",
    "yellow_cards_against",
    "red_cards_for",
    "red_cards_against",
]

FORM_COLUMNS = {
    "points": "points",
    "win": "wins",
    "draw": "draws",
    "loss": "losses",
    "goals_for": "goals_for",
    "goals_against": "goals_against",
    "goal_difference": "goal_difference",
}

FINAL_FEATURE_COLUMNS = [
    "last_5_points",
    "last_5_wins",
    "last_5_draws",
    "last_5_losses",
    "last_5_goals_for",
    "last_5_goals_against",
    "last_5_goal_difference",
    "matches_available_last_5",
    "venue_last_5_points",
    "venue_last_5_wins",
    "venue_last_5_draws",
    "venue_last_5_losses",
    "venue_last_5_goals_for",
    "venue_last_5_goals_against",
    "venue_last_5_goal_difference",
    "venue_matches_available_last_5",
    "season_points_before_match",
    "season_matches_before_match",
    "rest_days",
]

def load_processed_matches(
    file_path: Path = PROCESSED_DATA_PATH,
) -> pd.DataFrame:
    """Load and validate the cleaned match dataset."""
    if not file_path.exists():
        raise FileNotFoundError(
            f"Processed dataset was not found: {file_path}. "
            "Run 'python -m src.data_cleaner' first."
        )

    matches = pd.read_csv(file_path, parse_dates=["date"])

    missing_columns = [
        column for column in REQUIRED_COLUMNS if column not in matches.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Processed dataset is missing required columns: {missing_columns}"
        )

    if matches.isna().any().any():
        raise ValueError("Processed dataset contains missing values.")

    if matches.duplicated().any():
        raise ValueError("Processed dataset contains duplicate rows.")

    if not matches["date"].is_monotonic_increasing:
        raise ValueError("Processed matches are not in chronological order.")

    return matches


def create_team_match_records(matches: pd.DataFrame) -> pd.DataFrame:
    """Create one home-team record and one away-team record per match."""
    matches = matches.copy()
    matches["match_id"] = matches.index

    home_records = matches[
        [
            "match_id",
            "season",
            "date",
            "time",
            "home_team",
            "away_team",
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
    ].copy()

    home_records = home_records.rename(
        columns={
            "home_team": "team",
            "away_team": "opponent",
            "home_goals": "goals_for",
            "away_goals": "goals_against",
            "home_shots": "shots_for",
            "away_shots": "shots_against",
            "home_shots_on_target": "shots_on_target_for",
            "away_shots_on_target": "shots_on_target_against",
            "home_corners": "corners_for",
            "away_corners": "corners_against",
            "home_fouls": "fouls_for",
            "away_fouls": "fouls_against",
            "home_yellow_cards": "yellow_cards_for",
            "away_yellow_cards": "yellow_cards_against",
            "home_red_cards": "red_cards_for",
            "away_red_cards": "red_cards_against",
        }
    )
    home_records["venue"] = "home"

    away_records = matches[
        [
            "match_id",
            "season",
            "date",
            "time",
            "away_team",
            "home_team",
            "away_goals",
            "home_goals",
            "away_shots",
            "home_shots",
            "away_shots_on_target",
            "home_shots_on_target",
            "away_corners",
            "home_corners",
            "away_fouls",
            "home_fouls",
            "away_yellow_cards",
            "home_yellow_cards",
            "away_red_cards",
            "home_red_cards",
        ]
    ].copy()

    away_records = away_records.rename(
        columns={
            "away_team": "team",
            "home_team": "opponent",
            "away_goals": "goals_for",
            "home_goals": "goals_against",
            "away_shots": "shots_for",
            "home_shots": "shots_against",
            "away_shots_on_target": "shots_on_target_for",
            "home_shots_on_target": "shots_on_target_against",
            "away_corners": "corners_for",
            "home_corners": "corners_against",
            "away_fouls": "fouls_for",
            "home_fouls": "fouls_against",
            "away_yellow_cards": "yellow_cards_for",
            "home_yellow_cards": "yellow_cards_against",
            "away_red_cards": "red_cards_for",
            "home_red_cards": "red_cards_against",
        }
    )
    away_records["venue"] = "away"

    team_records = pd.concat(
        [home_records, away_records],
        ignore_index=True,
    )

    team_records = team_records[TEAM_RECORD_COLUMNS]

    team_records = team_records.sort_values(
        ["date", "time", "match_id", "venue"],
        kind="stable",
    ).reset_index(drop=True)

    return team_records

def calculate_match_outcomes(team_records: pd.DataFrame) -> pd.DataFrame:
    """Add each team's result, points, and goal difference."""
    team_records = team_records.copy()

    team_records["win"] = (
        team_records["goals_for"] > team_records["goals_against"]
    ).astype(int)

    team_records["draw"] = (
        team_records["goals_for"] == team_records["goals_against"]
    ).astype(int)

    team_records["loss"] = (
        team_records["goals_for"] < team_records["goals_against"]
    ).astype(int)

    team_records["points"] = (
        team_records["win"] * 3 + team_records["draw"]
    )

    team_records["goal_difference"] = (
        team_records["goals_for"] - team_records["goals_against"]
    )

    team_records["team_result"] = "L"
    team_records.loc[team_records["draw"] == 1, "team_result"] = "D"
    team_records.loc[team_records["win"] == 1, "team_result"] = "W"

    return team_records

def add_previous_match_features(
    team_records: pd.DataFrame,
) -> pd.DataFrame:
    """Add statistics from each team's previous match."""
    team_records = team_records.copy()

    for column in HISTORICAL_COLUMNS:
        team_records[f"previous_{column}"] = (
            team_records.groupby("team", sort=False)[column].shift(1)
        )

    return team_records

def add_five_match_form(
    team_records: pd.DataFrame,
) -> pd.DataFrame:
    """Calculate each team's totals from its previous five matches."""
    team_records = team_records.copy()

    for source_name, feature_name in FORM_COLUMNS.items():
        previous_column = f"previous_{source_name}"
        form_column = f"last_5_{feature_name}"

        team_records[form_column] = (
            team_records.groupby("team", sort=False)[previous_column]
            .transform(
                lambda values: values.rolling(
                    window=5,
                    min_periods=1,
                ).sum()
            )
        )

    team_records["matches_available_last_5"] = (
        team_records["previous_points"]
        .notna()
        .groupby(team_records["team"], sort=False)
        .transform(
            lambda values: values.rolling(
                window=5,
                min_periods=1,
            ).sum()
        )
        .astype(int)
    )

    return team_records
def add_venue_form_features(
    team_records: pd.DataFrame,
) -> pd.DataFrame:
    """Calculate form from a team's previous five matches at the same venue."""
    team_records = team_records.copy()

    grouped_records = team_records.groupby(
        ["team", "venue"],
        sort=False,
    )

    for source_name, feature_name in FORM_COLUMNS.items():
        team_records[f"venue_last_5_{feature_name}"] = (
            grouped_records[source_name].transform(
                lambda values: values.shift(1).rolling(
                    window=5,
                    min_periods=1,
                ).sum()
            )
        )

    team_records["venue_matches_available_last_5"] = (
        grouped_records["points"].transform(
            lambda values: values.shift(1).notna().rolling(
                window=5,
                min_periods=1,
            ).sum()
        )
        .fillna(0)
        .astype(int)
    )

    return team_records

def add_season_points(
    team_records: pd.DataFrame,
) -> pd.DataFrame:
    """Calculate points earned earlier in the current season."""
    team_records = team_records.copy()

    season_groups = team_records.groupby(
        ["season", "team"],
        sort=False,
    )

    team_records["season_points_before_match"] = (
        season_groups["points"].transform(
            lambda values: values.shift(1).fillna(0).cumsum()
        )
    )

    team_records["season_matches_before_match"] = (
        season_groups.cumcount()
    )

    return team_records

def add_rest_days(
    team_records: pd.DataFrame,
) -> pd.DataFrame:
    """Calculate days since each team's previous match."""
    team_records = team_records.copy()

    team_records["previous_match_date"] = (
        team_records.groupby("team", sort=False)["date"].shift(1)
    )

    team_records["rest_days"] = (
        team_records["date"] - team_records["previous_match_date"]
    ).dt.days

    return team_records

def build_match_feature_table(
    matches: pd.DataFrame,
    team_records: pd.DataFrame,
) -> pd.DataFrame:
    """Merge home and away historical features onto each match."""
    match_table = matches.reset_index(names="match_id")[
        [
            "match_id",
            "season",
            "division",
            "date",
            "time",
            "home_team",
            "away_team",
            "home_goals",
            "away_goals",
            "result",
        ]
    ]

    home_features = team_records[
        team_records["venue"] == "home"
    ][["match_id", *FINAL_FEATURE_COLUMNS]].copy()

    home_features = home_features.rename(
        columns={
            column: f"home_{column}"
            for column in FINAL_FEATURE_COLUMNS
        }
    )

    away_features = team_records[
        team_records["venue"] == "away"
    ][["match_id", *FINAL_FEATURE_COLUMNS]].copy()

    away_features = away_features.rename(
        columns={
            column: f"away_{column}"
            for column in FINAL_FEATURE_COLUMNS
        }
    )

    feature_table = match_table.merge(
        home_features,
        on="match_id",
        how="left",
        validate="one_to_one",
    )

    feature_table = feature_table.merge(
        away_features,
        on="match_id",
        how="left",
        validate="one_to_one",
    )

    zero_fill_columns = [
        column
        for column in feature_table.columns
        if "last_5" in column
    ]

    feature_table[zero_fill_columns] = feature_table[
        zero_fill_columns
    ].fillna(0)

    return feature_table


def save_feature_table(
    feature_table: pd.DataFrame,
    file_path: Path = FEATURE_DATA_PATH,
) -> None:
    """Save the final match-level feature table."""
    file_path.parent.mkdir(parents=True, exist_ok=True)
    feature_table.to_csv(file_path, index=False)


def main() -> None:
    """Build and save the leakage-safe feature table."""
    matches = load_processed_matches()
    team_records = create_team_match_records(matches)
    team_records = calculate_match_outcomes(team_records)
    team_records = add_previous_match_features(team_records)
    team_records = add_five_match_form(team_records)
    team_records = add_venue_form_features(team_records)
    team_records = add_season_points(team_records)
    team_records = add_rest_days(team_records)

    feature_table = build_match_feature_table(
        matches,
        team_records,
    )
    save_feature_table(feature_table)

    print("Feature table created successfully.")
    print(f"Matches: {len(feature_table):,}")
    print(f"Columns: {len(feature_table.columns)}")
    print(f"Saved to: {FEATURE_DATA_PATH}")

    print("\nFirst match:")
    print(
        feature_table[
            [
                "date",
                "home_team",
                "away_team",
                "home_last_5_points",
                "away_last_5_points",
                "home_season_points_before_match",
                "away_season_points_before_match",
                "home_rest_days",
                "away_rest_days",
                "result",
            ]
        ].head(1).to_string(index=False)
    )


if __name__ == "__main__":
    main()