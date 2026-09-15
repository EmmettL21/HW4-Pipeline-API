"""FastAPI service around the fitted NL MVP vote-share pipeline.

Local dev:
    uv run uvicorn serve:app --reload
    -> http://localhost:8000/docs
"""

from pathlib import Path
from typing import Optional

import joblib
from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from pipeline_def import LeagueSeasonAdjuster, STAT_COLS, YEAR_COL  # noqa: F401  (needed to unpickle the bundle)

ARTIFACT_PATH = Path(__file__).parent / "pipeline.joblib"

app = FastAPI(
    title="NL MVP Vote-Share Predictor",
    description="Predicts a batter's National League MVP vote share from season-adjusted batting stats.",
    version="1.0.0",
)

# The Vercel frontend calls this API from the browser (a different origin),
# so it needs CORS headers - curl/Postman don't enforce CORS, only browsers
# do, which is why this is easy to miss until you actually click the button.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Loaded once at import time, not per-request and not per-container-boot from
# scratch - this IS the fitted pipeline.joblib, so if the file is missing or
# incompatible we fail soft (ARTIFACT stays None) rather than crashing import.
ARTIFACT = None
ARTIFACT_LOAD_ERROR: Optional[str] = None

try:
    ARTIFACT = joblib.load(ARTIFACT_PATH)
except Exception as exc:  # noqa: BLE001 - any load failure should degrade to 503, not crash the app
    ARTIFACT_LOAD_ERROR = f"{type(exc).__name__}: {exc}"


def get_artifact() -> dict:
    if ARTIFACT is None:
        raise HTTPException(
            status_code=503,
            detail=f"Model artifact unavailable: {ARTIFACT_LOAD_ERROR or 'not loaded'}",
        )
    return ARTIFACT


class PlayerStatsInput(BaseModel):
    year: int = Field(..., ge=1900, le=2035, description="Season year")
    WAR: float = Field(..., ge=-5.0, le=15.0, description="Wins Above Replacement")
    R: int = Field(..., ge=0, le=200, description="Runs scored")
    H: int = Field(..., ge=0, le=270, description="Hits")
    HR: int = Field(..., ge=0, le=80, description="Home runs")
    RBI: int = Field(..., ge=0, le=200, description="Runs batted in")
    SB: int = Field(..., ge=0, le=140, description="Stolen bases")
    BB: int = Field(..., ge=0, le=250, description="Walks")
    batting_avg: float = Field(..., ge=0.0, le=0.500, description="Batting average")
    onbase_perc: float = Field(..., ge=0.0, le=0.700, description="On-base percentage")
    slugging_perc: float = Field(..., ge=0.0, le=1.000, description="Slugging percentage")
    onbase_plus_slugging: float = Field(..., ge=0.0, le=1.700, description="OPS")
    player_name: Optional[str] = Field(None, max_length=100, description="Optional label, not used by the model")
    team: Optional[str] = Field(None, max_length=5, description="Optional label, not used by the model")


class PredictionResponse(BaseModel):
    player_name: Optional[str]
    team: Optional[str]
    year: int
    predicted_vote_share: float
    predicted_vote_share_clipped: float


@app.get("/health")
def health():
    if ARTIFACT is None:
        raise HTTPException(
            status_code=503,
            detail=f"Model artifact unavailable: {ARTIFACT_LOAD_ERROR or 'not loaded'}",
        )
    return {"status": "ok"}


@app.get("/info")
def info(bundle: dict = Depends(get_artifact)):
    return {
        "target": bundle["target"],
        "stat_cols": bundle["stat_cols"],
        "year_col": bundle["year_col"],
        "training_years": bundle["training_years"],
        "n_train_rows": bundle["n_train_rows"],
        "metadata": bundle["metadata"],
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(payload: PlayerStatsInput, bundle: dict = Depends(get_artifact)):
    import pandas as pd

    row = pd.DataFrame(
        [
            {
                YEAR_COL: payload.year,
                **{col: getattr(payload, col) for col in bundle["stat_cols"]},
            }
        ]
    )

    pipeline = bundle["pipeline"]
    predicted = float(pipeline.predict(row)[0])
    clipped = min(max(predicted, 0.0), 1.0)

    return PredictionResponse(
        player_name=payload.player_name,
        team=payload.team,
        year=payload.year,
        predicted_vote_share=predicted,
        predicted_vote_share_clipped=clipped,
    )
