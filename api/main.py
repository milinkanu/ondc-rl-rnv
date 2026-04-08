"""
FastAPI layer for ONDCAgentEnv — session management, training, and health endpoints.
"""
from __future__ import annotations

import uuid
from dataclasses import asdict
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
import os

from ondc_env import ONDCAgentEnv, EnvConfig, EpisodeState

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(title="ONDCAgentEnv API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# In-memory stores
# ---------------------------------------------------------------------------

# session_id -> {"env": ONDCAgentEnv, "state": EpisodeState, "failed": bool}
_sessions: dict[str, dict] = {}

# run_id -> {"status": str, "metrics": dict}
_runs: dict[str, dict] = {}

# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class SessionStartRequest(BaseModel):
    max_steps: int = Field(default=50, ge=1)
    n_sellers: int = Field(default=5, ge=1, le=5)
    initial_budget: float = Field(default=1000.0, gt=0)
    target_item: str = Field(default="item")
    urgency: float | None = Field(default=None, ge=0.0, le=1.0)
    seed: int | None = None


class SessionStartResponse(BaseModel):
    session_id: str
    obs: list[float]
    info: dict[str, Any]


class StepRequest(BaseModel):
    action: int = Field(..., ge=0, le=14)


class StepResponse(BaseModel):
    obs: list[float]
    reward: float
    done: bool
    info: dict[str, Any]


class DeleteResponse(BaseModel):
    ok: bool


class HealthResponse(BaseModel):
    ok: bool


class TrainRequest(BaseModel):
    max_steps: int = Field(default=50, ge=1)
    n_sellers: int = Field(default=5, ge=1, le=5)
    initial_budget: float = Field(default=1000.0, gt=0)
    total_timesteps: int = Field(default=10000, ge=1)
    seed: int | None = None


class TrainResponse(BaseModel):
    run_id: str


class TrainStatusResponse(BaseModel):
    status: str
    metrics: dict[str, Any]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _serialize_info(info: dict) -> dict:
    """Convert info dict values to JSON-serialisable types."""
    result = {}
    for k, v in info.items():
        if hasattr(v, "__int__") and not isinstance(v, bool):
            result[k] = int(v)
        elif isinstance(v, list):
            result[k] = [str(e) for e in v]
        else:
            result[k] = v
    return result


def _serialize_state(state: EpisodeState) -> dict:
    """Serialize EpisodeState to a JSON-safe dict."""
    d = asdict(state)
    # Convert IntEnum fields
    d["current_phase"] = int(state.current_phase)
    d["order_status"] = int(state.order_status) if state.order_status is not None else None
    return d


def _get_session(session_id: str) -> dict:
    if session_id not in _sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    return _sessions[session_id]


# ---------------------------------------------------------------------------
# Session endpoints
# ---------------------------------------------------------------------------


@app.post("/session/start", response_model=SessionStartResponse)
def session_start(req: SessionStartRequest) -> SessionStartResponse:
    config = EnvConfig(
        max_steps=req.max_steps,
        n_sellers=req.n_sellers,
        initial_budget=req.initial_budget,
        seed=req.seed,
    )
    env = ONDCAgentEnv(config)

    options: dict[str, Any] = {"target_item": req.target_item}
    if req.urgency is not None:
        options["urgency"] = req.urgency

    obs, info = env.reset(seed=req.seed, options=options)

    session_id = str(uuid.uuid4())
    _sessions[session_id] = {"env": env, "failed": False}

    return SessionStartResponse(
        session_id=session_id,
        obs=obs.tolist(),
        info=_serialize_info(info),
    )


@app.post("/session/{session_id}/step", response_model=StepResponse)
def session_step(session_id: str, req: StepRequest) -> StepResponse:
    session = _get_session(session_id)
    env: ONDCAgentEnv = session["env"]

    try:
        obs, reward, terminated, truncated, info = env.step(req.action)
    except Exception as exc:
        session["failed"] = True
        raise HTTPException(
            status_code=500,
            detail={"detail": "Environment error", "error": str(exc)},
        ) from exc

    done = terminated or truncated
    return StepResponse(
        obs=obs.tolist(),
        reward=float(reward),
        done=done,
        info=_serialize_info(info),
    )


@app.get("/session/{session_id}/state")
def session_state(session_id: str) -> dict:
    session = _get_session(session_id)
    env: ONDCAgentEnv = session["env"]
    if env.state is None:
        raise HTTPException(status_code=404, detail="Session not found")
    return _serialize_state(env.state)


@app.delete("/session/{session_id}", response_model=DeleteResponse)
def session_delete(session_id: str) -> DeleteResponse:
    _get_session(session_id)  # raises 404 if missing
    del _sessions[session_id]
    return DeleteResponse(ok=True)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(ok=True)


# ---------------------------------------------------------------------------
# Training endpoints
# ---------------------------------------------------------------------------


@app.post("/train", response_model=TrainResponse)
def train_start(req: TrainRequest) -> TrainResponse:
    run_id = str(uuid.uuid4())
    _runs[run_id] = {
        "status": "running",
        "metrics": {
            "total_timesteps": req.total_timesteps,
            "steps_done": 0,
        },
    }
    return TrainResponse(run_id=run_id)


@app.get("/train/{run_id}/status", response_model=TrainStatusResponse)
def train_status(run_id: str) -> TrainStatusResponse:
    if run_id not in _runs:
        raise HTTPException(status_code=404, detail="Run not found")
    run = _runs[run_id]
    return TrainStatusResponse(status=run["status"], metrics=run["metrics"])

# ---------------------------------------------------------------------------
# OpenEnv Standard Root Endpoints
# ---------------------------------------------------------------------------
_global_session_id = "default_openenv"

class OpenEnvResetResponse(BaseModel):
    obs: list[float]
    info: dict[str, Any]

@app.post("/reset", response_model=OpenEnvResetResponse)
def openenv_reset(payload: dict[str, Any] | None = None) -> OpenEnvResetResponse:
    global _global_session_id
    config = EnvConfig(max_steps=50, initial_budget=1000.0)
    env = ONDCAgentEnv(config)
    obs, info = env.reset(seed=42)
    _sessions[_global_session_id] = {"env": env, "failed": False}
    return OpenEnvResetResponse(
        obs=obs.tolist(),
        info=_serialize_info(info),
    )

@app.post("/step", response_model=StepResponse)
def openenv_step(req: StepRequest) -> StepResponse:
    global _global_session_id
    if _global_session_id not in _sessions:
        raise HTTPException(status_code=400, detail="Must call /reset first")
    return session_step(_global_session_id, req)

# ---------------------------------------------------------------------------
# Frontend Mount
# ---------------------------------------------------------------------------

frontend_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "frontend")
os.makedirs(frontend_dir, exist_ok=True)
app.mount("/", StaticFiles(directory=frontend_dir, html=True), name="frontend")
