# Data

## Source

Historical Premier League results were downloaded from
[Football-Data.co.uk](https://www.football-data.co.uk/englandm.php).

Column definitions and source acknowledgements are available in the
[Football-Data notes](https://www.football-data.co.uk/notes.txt).

## Seasons

- 2020/21
- 2021/22
- 2022/23
- 2023/24
- 2024/25
- 2025/26

Each completed season contains 380 matches. The six seasons contain 2,280
matches in total.

## Raw files

Files under `data/raw/` are preserved in their original downloaded form.

They must not be manually edited. Cleaning, standardization, feature creation,
and combination produce separate files under `data/processed/`.

The raw CSV files are not committed to Git because the source does not provide
a clear redistribution license. They can be recreated with:

    python -m src.data_loader

## Processed dataset

Phase 3 creates:

    data/processed/premier_league_matches.csv

The processed dataset contains:

- Six seasons
- 380 matches per season
- 2,280 matches in total
- 21 columns
- Standardized column names
- Dates formatted as `YYYY-MM-DD`
- Standardized team names
- No missing values
- No duplicate rows
- Result labels verified against final scores
- Matches sorted chronologically

Create or replace the processed dataset with:

    python -m src.data_cleaner

The cleaning program reads the raw CSV files but never changes them.

## Feature dataset

Phase 4 creates:

    data/processed/premier_league_features.csv

The feature dataset contains:

- 2,280 match rows
- 48 columns
- Previous-five-match points, wins, draws, and losses
- Previous-five-match goals and goal difference
- Separate home and away form
- Season points before each match
- Matches played before each match
- Days since each team's previous match
- Home and away historical features on the same row

Create or replace the feature dataset with:

    python -m src.feature_engineer

All rolling and cumulative features are shifted so that the current match is
excluded. Form values are set to zero when no earlier matches are available.
Rest days are missing for a team's first appearance because no earlier match
date exists.

The `home_goals`, `away_goals`, and `result` columns are prediction targets
and must not be used as model inputs.

Like the cleaned dataset, the generated feature CSV is not committed to Git.
It can be recreated from the processed match dataset.

## Primary raw columns

- `Date`: match date
- `HomeTeam`: home club
- `AwayTeam`: away club
- `FTHG`: full-time home goals
- `FTAG`: full-time away goals
- `FTR`: full-time result (`H`, `D`, or `A`)
- `HS` and `AS`: shots
- `HST` and `AST`: shots on target
- `HC` and `AC`: corners
- `HF` and `AF`: fouls
- `HY` and `AY`: yellow cards
- `HR` and `AR`: red cards

## Version 1 column policy

Identity and ordering columns:

- `Div`
- `Date`
- `Time`
- `HomeTeam`
- `AwayTeam`

Result and target columns:

- `FTHG`
- `FTAG`
- `FTR`

Historical performance candidates:

- `HS` and `AS`
- `HST` and `AST`
- `HC` and `AC`
- `HF` and `AF`
- `HY` and `AY`
- `HR` and `AR`

Half-time results, referee information, and bookmaker odds are excluded from
Version 1 model inputs.

## Leakage policy

Statistics created during a match cannot be used to predict that same match.

Match statistics may only be used after they have been shifted into
historical features calculated from earlier matches. Final scores and the
full-time result of the match being predicted cannot be model inputs.

## Reproducibility

Download the raw data:

    python -m src.data_loader

Inspect the raw data:

    python notebooks/inspect_data.py

Create the processed dataset:

    python -m src.data_cleaner

Create the feature dataset:

    python -m src.feature_engineer

Run the tests:

    python -m pytest -v