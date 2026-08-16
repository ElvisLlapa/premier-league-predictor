"""Download and inspect raw Premier League match data."""

import time
from pathlib import Path

import requests

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"

SEASON_URLS = {
    "2020-21": "https://www.football-data.co.uk/mmz4281/2021/E0.csv",
    "2021-22": "https://www.football-data.co.uk/mmz4281/2122/E0.csv",
    "2022-23": "https://www.football-data.co.uk/mmz4281/2223/E0.csv",
    "2023-24": "https://www.football-data.co.uk/mmz4281/2324/E0.csv",
    "2024-25": "https://www.football-data.co.uk/mmz4281/2425/E0.csv",
    "2025-26": "https://www.football-data.co.uk/mmz4281/2526/E0.csv",
}


def download_season(
    season: str,
    url: str,
    overwrite: bool = False,
) -> Path:
    """Download one season without modifying the original CSV contents."""
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

    output_path = RAW_DATA_DIR / f"PL_{season}.csv"

    if output_path.exists() and not overwrite:
        print(f"Already exists: {output_path.name}")
        return output_path

    response = requests.get(
        url,
        headers={"User-Agent": "PremierLeaguePredictor/1.0"},
        timeout=30,
    )
    response.raise_for_status()

    output_path.write_bytes(response.content)
    print(f"Downloaded: {output_path.name}")

    return output_path


def download_all_seasons(overwrite: bool = False) -> list[Path]:
    """Download every season listed in SEASON_URLS."""
    downloaded_files = []

    for season, url in SEASON_URLS.items():
        path = download_season(season, url, overwrite=overwrite)
        downloaded_files.append(path)

        # Avoid sending many requests to the source at once.
        time.sleep(1)

    return downloaded_files


if __name__ == "__main__":
    files = download_all_seasons()
    print(f"\nAvailable raw files: {len(files)}")
