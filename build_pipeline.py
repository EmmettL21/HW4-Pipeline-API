"""Fits the NL MVP-share pipeline and dumps a bundle to pipeline.joblib.

Run with: uv run build_pipeline.py
"""

from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd
import sklearn
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from pipeline_def import STAT_COLS, YEAR_COL, LeagueSeasonAdjuster

DATA_PATH = Path(__file__).parent / "data" / "nl_mvp_voting_batters_2005_2025.csv"
ARTIFACT_PATH = Path(__file__).parent / "pipeline.joblib"


def load_data() -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(DATA_PATH)
    df["vote_share"] = df["share"].str.rstrip("%").astype(float) / 100.0

    X = df[[YEAR_COL] + STAT_COLS].copy()
    y = df["vote_share"]
    return X, y


def build_pipeline() -> Pipeline:
    return Pipeline(
        steps=[
            ("season_adjust", LeagueSeasonAdjuster(stat_cols=STAT_COLS, year_col=YEAR_COL)),
            ("regressor", LinearRegression()),
        ]
    )


def main() -> None:
    X, y = load_data()

    # Held-out check purely for our own sanity - not shipped in the artifact.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    holdout_pipeline = build_pipeline().fit(X_train, y_train)
    r2 = holdout_pipeline.score(X_test, y_test)
    print(f"Holdout R^2 on vote_share (20% test split): {r2:.3f}")

    # Real artifact: fit on the full dataset so it has every season's learned state.
    pipeline = build_pipeline().fit(X, y)

    years = sorted(X[YEAR_COL].unique().tolist())
    bundle = {
        "pipeline": pipeline,
        "stat_cols": STAT_COLS,
        "year_col": YEAR_COL,
        "target": "vote_share",
        "training_years": years,
        "n_train_rows": int(len(X)),
        "metadata": {
            "steps": [name for name, _ in pipeline.steps],
            "built_at": datetime.now(timezone.utc).isoformat(),
            "sklearn_version": sklearn.__version__,
            "description": (
                "Linear regression predicting a batter's NL MVP vote share "
                "(0-1) from season-adjusted batting stats. Custom "
                "LeagueSeasonAdjuster transformer z-scores each stat "
                "against that season's field of vote-getters before the "
                "regressor sees it."
            ),
            "holdout_r2": round(float(r2), 4),
        },
    }

    joblib.dump(bundle, ARTIFACT_PATH)
    print(f"Wrote {ARTIFACT_PATH} ({ARTIFACT_PATH.stat().st_size} bytes)")
    print(f"Trained on {len(X)} rows across seasons {years[0]}-{years[-1]} ({len(years)} seasons)")


if __name__ == "__main__":
    main()
