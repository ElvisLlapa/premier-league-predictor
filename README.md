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

- Phase 1 complete: project environment and structure
- Phase 2 complete: six historical seasons downloaded, inspected, tested,
  and documented
- Next: Phase 3 — clean and combine the raw match data

## Dataset

The project uses six completed Premier League seasons from
[Football-Data.co.uk](https://www.football-data.co.uk/englandm.php), covering
2020/21 through 2025/26.

Raw files are preserved without manual modification. All cleaning and feature
engineering will be performed reproducibly in Python.

See [`data/README.md`](data/README.md) for the dataset source, column policy,
and leakage policy.

## Local setup

Create and activate a virtual environment:

```zsh
python3.13 -m venv .venv
source .venv/bin/activate
```

Install the required packages:

```zsh
python -m pip install -r requirements.txt
```

## Download and validate the data

Download the configured seasons:

```zsh
python -m src.data_loader
```

Inspect the raw datasets:

```zsh
python notebooks/inspect_data.py
```

Run the automated tests:

```zsh
python -m pytest
```