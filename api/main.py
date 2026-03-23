"""FastAPI application entry point.

Manages the application lifespan (model loading), registers middleware,
mounts static files, and includes the /analyze router.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from api.routes.analyze import router
from pipeline.pose_detector import PoseDetector
from pipeline.weight_estimator import WeightEstimator
from training.config import Config

_log = logging.getLogger(__name__)

_cfg = Config()

# Resolved at import time so the path is correct regardless of the working
# directory from which uvicorn is invoked.
_FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"
_CHECKPOINT_PATH = (
    Path(__file__).resolve().parent.parent / _cfg.checkpoint_dir / "bioscan_model.pth"
)
_SCALER_PATH = Path(__file__).resolve().parent.parent / _cfg.scaler_path


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load shared resources once at startup; release them on shutdown.

    PoseDetector and WeightEstimator are both expensive to initialise and
    designed to be reused across requests.  Storing them on app.state keeps
    them alive for the full server lifetime without using module-level globals.
    """
    _log.info("BioScan AI starting — loading PoseDetector and WeightEstimator")
    app.state.pose_detector = PoseDetector()
    app.state.weight_estimator = WeightEstimator(
        checkpoint_path=_CHECKPOINT_PATH,
        scaler_path=_SCALER_PATH,
        std_threshold_kg=_cfg.std_threshold_kg,
    )
    _log.info("Models loaded successfully")
    yield
    _log.info("BioScan AI shutting down — releasing PoseDetector")
    app.state.pose_detector.close()


app = FastAPI(
    title="BioScan AI",
    version="0.1.0",
    description="Visual body weight estimator — POST an image with height and age to /analyze.",
    lifespan=lifespan,
)

# IMPORTANT: allow_origins=["*"] is intentional for local development only.
# Before deploying to production, replace ["*"] with the exact list of allowed
# origins (e.g. ["https://your-domain.com"]) to prevent cross-origin abuse.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory=_FRONTEND_DIR), name="static")
app.include_router(router)


@app.get("/health", tags=["ops"])
def health() -> dict[str, str]:
    """Liveness probe for load balancers and container orchestrators."""
    return {"status": "ok"}
