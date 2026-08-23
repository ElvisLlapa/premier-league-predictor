"""Create chronological Elo ratings for Premier League teams."""

from pathlib import Path

import pandas as pd

INITIAL_RATING = 1500.0
K_FACTOR = 20.0
HOME_ADVANTAGE = 100.0

def calculate_expected_scores(
    home_elo: float,
    away_elo: float,
    home_advantage: float = HOME_ADVANTAGE,
) -> tuple[float, float]:
    """Calculate both teams' expected scores before a match."""
    adjusted_home_elo = home_elo + home_advantage

    home_expected = 1 / (
        1 + 10 ** ((away_elo - adjusted_home_elo) / 400)
    )
    away_expected = 1 - home_expected

    return home_expected, away_expected

def sort_matches_chronologically(matches: pd.DataFrame) -> pd.DataFrame:
    """Return matches ordered from oldest to newest."""
    sorted_matches = matches.copy()
    sorted_matches["date"] = pd.to_datetime(sorted_matches["date"])

    sorted_matches = sorted_matches.sort_values(
        by="date",
        kind="stable",
    ).reset_index(drop=True)

    return sorted_matches

def get_actual_scores(result: str) -> tuple[float, float]:
    """Convert a match result into actual Elo scores."""
    scores = {
        "H": (1.0, 0.0),
        "D": (0.5, 0.5),
        "A": (0.0, 1.0),
    }

    if result not in scores:
        raise ValueError(f"Invalid match result: {result}")

    return scores[result]


def update_elo_ratings(
    home_elo: float,
    away_elo: float,
    result: str,
    k_factor: float = K_FACTOR,
) -> tuple[float, float]:
    """Update both teams' Elo ratings after a match."""
    home_expected, away_expected = calculate_expected_scores(
        home_elo,
        away_elo,
    )
    home_actual, away_actual = get_actual_scores(result)

    new_home_elo = home_elo + k_factor * (
        home_actual - home_expected
    )
    new_away_elo = away_elo + k_factor * (
        away_actual - away_expected
    )

    return new_home_elo, new_away_elo

def add_elo_features(matches: pd.DataFrame) -> pd.DataFrame:
    """Add leakage-safe pre-match Elo features."""
    matches_with_elo = sort_matches_chronologically(matches)
    team_ratings: dict[str, float] = {}

    home_elos = []
    away_elos = []
    elo_differences = []
    home_expected_scores = []
    away_expected_scores = []

    for match in matches_with_elo.itertuples(index=False):
        home_elo = team_ratings.get(match.home_team, INITIAL_RATING)
        away_elo = team_ratings.get(match.away_team, INITIAL_RATING)

        home_expected, away_expected = calculate_expected_scores(
            home_elo,
            away_elo,
        )

        home_elos.append(home_elo)
        away_elos.append(away_elo)
        elo_differences.append(home_elo - away_elo)
        home_expected_scores.append(home_expected)
        away_expected_scores.append(away_expected)

        new_home_elo, new_away_elo = update_elo_ratings(
            home_elo,
            away_elo,
            match.result,
        )

        team_ratings[match.home_team] = new_home_elo
        team_ratings[match.away_team] = new_away_elo

    matches_with_elo["home_elo"] = home_elos
    matches_with_elo["away_elo"] = away_elos
    matches_with_elo["elo_difference"] = elo_differences
    matches_with_elo["home_elo_expected"] = home_expected_scores
    matches_with_elo["away_elo_expected"] = away_expected_scores

    return matches_with_elo

def create_elo_history(matches_with_elo: pd.DataFrame) -> pd.DataFrame:
    """Create one pre-match Elo record for each team in every match."""
    home_history = matches_with_elo[
        [
            "date",
            "season",
            "home_team",
            "away_team",
            "home_elo",
            "home_elo_expected",
        ]
    ].rename(
        columns={
            "home_team": "team",
            "away_team": "opponent",
            "home_elo": "pre_match_elo",
            "home_elo_expected": "expected_score",
        }
    )
    home_history["venue"] = "home"

    away_history = matches_with_elo[
        [
            "date",
            "season",
            "away_team",
            "home_team",
            "away_elo",
            "away_elo_expected",
        ]
    ].rename(
        columns={
            "away_team": "team",
            "home_team": "opponent",
            "away_elo": "pre_match_elo",
            "away_elo_expected": "expected_score",
        }
    )
    away_history["venue"] = "away"

    history = pd.concat(
        [home_history, away_history],
        ignore_index=True,
    )

    return history.sort_values(
        by=["date", "team"],
        kind="stable",
    ).reset_index(drop=True)


def save_elo_datasets() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create and save the match Elo features and team Elo history."""
    input_path = Path(
        "data/processed/premier_league_features.csv"
    )
    features_output_path = Path(
        "data/processed/premier_league_features_with_elo.csv"
    )
    history_output_path = Path(
        "data/processed/premier_league_elo_history.csv"
    )

    matches = pd.read_csv(input_path)
    matches_with_elo = add_elo_features(matches)
    elo_history = create_elo_history(matches_with_elo)

    matches_with_elo.to_csv(features_output_path, index=False)
    elo_history.to_csv(history_output_path, index=False)

    return matches_with_elo, elo_history


if __name__ == "__main__":
    elo_matches, history = save_elo_datasets()

    print(
        "Saved match features:",
        f"{len(elo_matches)} rows x {len(elo_matches.columns)} columns",
    )
    print("Saved Elo history:", len(history), "team records")