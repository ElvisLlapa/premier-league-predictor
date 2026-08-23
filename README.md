# Premier League Predictor

A portfolio project that uses historical Premier League match data and
machine learning to predict match outcomes and simulate the final league
table.

## Planned features

- Leakage-safe rolling team statistics
- Chronological Elo ratings
- Random Forest match predictions
- Time-based model evaluation
- Season simulations
- Streamlit dashboard

## Project status

- Phase 1: Project setup — complete
- Phase 2: Data download and inspection — complete
- Phase 3: Data cleaning and combination — complete
- Phase 4: Leakage-safe feature engineering — complete
- Phase 5: Chronological Elo ratings — complete

## Dataset

The project uses six completed Premier League seasons from
[Football-Data.co.uk](https://www.football-data.co.uk/englandm.php), covering
2020/21 through 2025/26.

Raw files are preserved without manual modification. Cleaning is performed
reproducibly in Python.

See [`data/README.md`](data/README.md) for the dataset source, column policy,
and leakage policy.

## Local setup

Create and activate a virtual environment:

```bash
python3.13 -m venv .venv
source .venv/bin/activate
```

Install the required packages:

```bash
python -m pip install -r requirements.txt
```

## Download and inspect the raw data

Download the configured seasons:

```bash
python -m src.data_loader
```

Inspect the raw datasets:

```bash
python notebooks/inspect_data.py
```

## Create the processed dataset

Run the cleaning program from the project root:

```bash
python -m src.data_cleaner
```

The program:

- Loads all six raw seasons
- Selects and renames the Version 1 columns
- Converts dates and numeric fields
- Standardizes team names
- Checks for missing and duplicate records
- Verifies results against final scores
- Adds season identifiers
- Combines matches chronologically
- Saves 2,280 matches to `data/processed/premier_league_matches.csv`

## Create the feature dataset

Run the feature-engineering program from the project root:

```bash
python -m src.feature_engineer
```

The program:

- Converts every match into home-team and away-team records
- Calculates team results, points, and goal difference
- Shifts match information to exclude the current match
- Calculates form from the previous five matches
- Calculates separate home and away form
- Calculates season points before each match
- Calculates days since each team's previous match
- Merges the home and away features onto one match row
- Saves 2,280 matches to `data/processed/premier_league_features.csv`

The feature dataset contains 48 columns. Historical features use only matches
played before the match represented by the current row.

The `home_goals`, `away_goals`, and `result` columns are prediction targets.
They must not be included as model input features.

## Create the Elo datasets

Run the Elo program from the project root:

```bash
python -m src.elo
```

The program:

- Gives every new team an initial Elo rating of 1500
- Uses a K-factor of 20 to update ratings
- Adds a 100-point home advantage to expected-score calculations
- Processes all matches chronologically
- Stores each team's rating before the current match
- Makes surprising results cause larger rating changes
- Adds home Elo, away Elo, Elo difference, and expected-score features
- Saves 2,280 matches to
  `data/processed/premier_league_features_with_elo.csv`
- Saves 4,560 team records to
  `data/processed/premier_league_elo_history.csv`

The current match result updates team ratings only for future matches. This
prevents match-result leakage into the Elo features.

## Testing and code quality

Run the automated tests:

```bash
python -m pytest -v
```

Run the Ruff checks:

```bash
python -m ruff check .
```