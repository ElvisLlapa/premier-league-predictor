"""Inspect the raw Premier League CSV files."""

from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"

REQUIRED_COLUMNS = {
    "Date",
    "HomeTeam",
    "AwayTeam",
    "FTHG",
    "FTAG",
    "FTR",
}


def inspect_file(file_path: Path) -> dict[str, object]:
    """Return a validation summary for one raw season file."""
    matches = pd.read_csv(file_path)

    missing_required = REQUIRED_COLUMNS - set(matches.columns)

    dates = pd.to_datetime(
        matches["Date"],
        dayfirst=True,
        errors="coerce",
    )

    teams = sorted(
        set(matches["HomeTeam"].dropna()) | set(matches["AwayTeam"].dropna())
    )

    result_counts = matches["FTR"].value_counts(dropna=False).to_dict()

    return {
        "file": file_path.name,
        "rows": len(matches),
        "columns": len(matches.columns),
        "teams": len(teams),
        "first_date": dates.min(),
        "last_date": dates.max(),
        "invalid_dates": int(dates.isna().sum()),
        "missing_required_columns": sorted(missing_required),
        "duplicate_rows": int(matches.duplicated().sum()),
        "results": result_counts,
    }


def main() -> None:
    """Print a summary of every downloaded season."""
    files = sorted(RAW_DATA_DIR.glob("PL_*.csv"))

    if not files:
        raise FileNotFoundError("No CSV files found. Run: python -m src.data_loader")

    summaries = [inspect_file(file_path) for file_path in files]
    summary_table = pd.DataFrame(summaries)

    print("\nDATASET SUMMARY")
    print(summary_table.to_string(index=False))

    total_matches = int(summary_table["rows"].sum())
    print(f"\nFiles: {len(files)}")
    print(f"Total matches: {total_matches}")

    first_file = files[0]
    sample = pd.read_csv(first_file)

    print(f"\nCOLUMNS IN {first_file.name}")
    for column in sample.columns:
        print(column)

    print("\nFIRST FIVE MATCHES")
    display_columns = [
        "Date",
        "HomeTeam",
        "AwayTeam",
        "FTHG",
        "FTAG",
        "FTR",
    ]
    print(sample[display_columns].head().to_string(index=False))


if __name__ == "__main__":
    main()
