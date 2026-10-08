"""Application-service routes for the evaluation dashboard."""

import json
import os
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, status

from app.config import settings
from week4.evaluator import METRIC_DEFINITIONS, load_dataset


router = APIRouter(prefix="/evaluation", tags=["Evaluation"])


def evaluation_directory() -> Path:
    return Path(os.getenv("EVALUATION_PATH", "data/evaluations"))


@router.get("/overview")
async def overview() -> dict[str, Any]:
    directory = evaluation_directory()
    report = _read_json(directory / "latest.json")
    run_status = _read_json(directory / "status.json") or {
        "state": "not_run",
        "completed": 0,
        "total": len(load_dataset()) * len(settings.week4_models),
    }
    return {
        "week": 4,
        "same_application": True,
        "models": list(settings.week4_models),
        "dataset_size": len(load_dataset()),
        "metric_definitions": METRIC_DEFINITIONS,
        "status": run_status,
        "aggregates": report.get("aggregates", {}) if report else {},
        "categories": report.get("categories", []) if report else [],
        "category_aggregates": report.get("category_aggregates", {}) if report else {},
        "category_verdicts": report.get("category_verdicts", {}) if report else {},
        "conclusions": report.get("conclusions", []) if report else [],
        "evaluation_conditions": report.get("evaluation_conditions", {}) if report else {},
        "generated_at": report.get("generated_at") if report else None,
    }


@router.get("/dataset")
async def dataset() -> dict[str, Any]:
    rows = load_dataset()
    return {"count": len(rows), "questions": rows}


@router.get("/report")
async def report() -> dict[str, Any]:
    payload = _read_json(evaluation_directory() / "latest.json")
    if payload is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No completed evaluation yet. Run scripts/run_week4_evaluation.py.",
        )
    return payload


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except (json.JSONDecodeError, OSError) as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Cannot read evaluation artifact {path.name}: {exc}",
        ) from exc
