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
- Phase 6: First Random Forest model — complete
- Phase 7: Time-based validation and evaluation — complete

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

## Train the Random Forest model

Run the training program from the project root:

```bash
python -m src.train
```

The training program:

- Selects 43 leakage-safe pre-match features
- Excludes final goals, match results, dates, and team names
- Trains on the 2020/21 through 2024/25 seasons
- Reserves the complete 2025/26 season as unseen test data
- Builds a most-common-result baseline
- Learns missing-value medians from training data only
- Trains a three-class Random Forest classifier
- Produces home-win, draw, and away-win probabilities
- Maps probabilities using the model's actual class order
- Retrains the production model on all six completed seasons
- Saves the production model for 2026/27 predictions

The chronological holdout results are:

- Baseline accuracy: 42.6%
- Random Forest accuracy: 47.4%
- Improvement over baseline: 4.7 percentage points
- Training matches: 1,900
- Unseen test matches: 380

After evaluation, the production model is retrained on all 2,280 matches
through May 24, 2026. It is intended for predicting the 2026/27 season.

Generated artifacts:

- `models/random_forest_model.joblib`
- `data/processed/random_forest_test_predictions.csv`

The Joblib bundle contains the trained production model, fitted median
imputer, 43 feature names, class labels, holdout metrics, training seasons,
and prediction-season metadata. Generated model and data files are ignored
by Git.

## Evaluate the model over time

Run the time-based evaluation program from the project root:

```bash
python -m src.evaluate
```

The evaluation uses three expanding-window chronological folds:

| Fold | Training seasons | Unseen test season |
| ---: | --- | --- |
| 1 | 2020/21 through 2022/23 | 2023/24 |
| 2 | 2020/21 through 2023/24 | 2024/25 |
| 3 | 2020/21 through 2024/25 | 2025/26 |

A completely new median imputer, most-common-result baseline, and Random
Forest are fitted inside every fold. No model or preprocessing information
from a later season is reused in an earlier fold.

Average results across the three unseen seasons:

- Baseline accuracy: 43.2%
- Random Forest accuracy: 52.5%
- Improvement over baseline: 9.3 percentage points
- Multiclass log loss: 0.996
- Home-win F1 score: 0.639
- Draw F1 score: 0.014
- Away-win F1 score: 0.536

The Random Forest beat the baseline in all three folds. Home wins were the
strongest result class, while draw detection was the largest weakness. Only
2 of 279 draws were predicted correctly across the three test seasons.

The evaluation also calculates:

- Precision, recall, and F1 score for home wins, draws, and away wins
- Confusion matrices for every test season
- One-vs-rest probability calibration
- Brier scores for all three result classes
- Feature importance across all validation folds

Elo difference, Elo expected scores, and team Elo ratings were the most
important model inputs across the three folds.

The complete evaluation report is available at
[`reports/phase_7/evaluation_report.md`](reports/phase_7/evaluation_report.md).

Saved Phase 7 artifacts include:

- Seven CSV evaluation tables in `reports/phase_7/tables/`
- Accuracy comparison chart
- Three-fold confusion-matrix chart
- Probability-calibration chart
- Feature-importance chart

The Phase 7 evaluation models are separate from the final production model.
The production model remains trained on all six completed seasons for
predicting 2026/27.

## Testing and code quality

Run the automated tests:

```bash
python -m pytest -v
```

Run the Ruff checks:

```bash
python -m ruff check .
```