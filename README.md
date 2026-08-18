# Premier League Predictor

A portfolio project that will use historical Premier League match data and
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
- Phase 4: Feature engineering — next

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

## Testing and code quality

Run the automated tests:

```bash
python -m pytest -v
```

Run the Ruff checks:

```bash
python -m ruff check .
```