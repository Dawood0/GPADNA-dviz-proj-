"""Precompute reusable beeswarm point positions."""

from pathlib import Path

from preprocessing import DATA_FILE, detect_columns, load_data
from vizs_src.beeswarm import prepare_beeswarm_data, preload_beeswarm_cache


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CACHE_DIR = PROJECT_ROOT / "preloaded" / "beeswarm"


def main() -> None:
    df = load_data(DATA_FILE)
    detected = detect_columns(df)
    data = prepare_beeswarm_data(
        df,
        features=detected["features"],
        target_col=detected["target_col"],
        student_id_col=detected["student_id_col"],
    )
    saved = preload_beeswarm_cache(data, CACHE_DIR)
    print(f"Saved {saved} beeswarm feature caches to {CACHE_DIR}")


if __name__ == "__main__":
    main()
