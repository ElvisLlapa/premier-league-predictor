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

Each completed Premier League season contains 380 matches. The six seasons
produce 2,280 raw match records in total.

## Raw files

Files under `data/raw/` are preserved in their original downloaded form.

They should not be manually edited. Cleaning, standardization, feature
creation, and combination will produce separate files under
`data/processed/`.

The raw CSV files are not committed to Git. They can be recreated using the
project downloader.

## Primary columns

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
the Version 1 model inputs.

## Leakage policy

Statistics created during a match cannot be used to predict that same match.

Match statistics may only be used after they have been shifted into
historical features calculated from earlier matches. Final scores and the
full-time result of the match being predicted cannot be model inputs.

## Reproducibility

Run the downloader from the project root:

```bash
python -m src.data_loader
```

Inspect the downloaded files:

```bash
python notebooks/inspect_data.py
```

Validate the datasets:

```bash
python -m pytest -v
```