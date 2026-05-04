from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import ValidationError

from backend.config import get_env
from backend.llm_client import LLMClientError, analyze_with_llm, get_llm_runtime, is_provider_configured, resolve_runtime_overrides
from backend.models import AnalyzeRequest, AnalyzeResponse, ErrorResponse, ReportRecord
from backend.prompt_builder import SYSTEM_PROMPT, build_user_prompt
from backend.scraper import build_raw_text_page, scrape_url
from backend.storage import get_report as get_persisted_report
from backend.storage import init_db, list_reports as list_persisted_reports
from backend.storage import report_exists, save_report


BASE_DIR = Path(__file__).resolve().parent.parent
FRONTEND_DIR = BASE_DIR / "frontend"


@asynccontextmanager
async def lifespan(_app: FastAPI):
    init_db()
    yield


app = FastAPI(
    title="SiteAudit AI",
    description="Instant consulting-grade website analysis powered by LLMs.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")


def _report_url(report_id: str) -> str:
    return f"/report/{report_id}"


def _dashboard_url() -> str:
    return "/dashboard"


def _report_summary(record: ReportRecord) -> dict[str, object]:
    return {
        "report_id": record.id,
        "created_at": record.created_at,
        "source_url": record.source.source_url,
        "source_title": record.source.title,
        "input_type": "url" if record.request.url else "raw_text",
        "business_context": record.request.business_context,
        "provider": record.request.llm_provider,
        "model": record.request.model,
        "overall_score": record.report.overall_score,
        "executive_summary": record.report.executive_summary,
        "critical_issues_count": len(record.report.critical_issues),
        "quick_wins_count": len(record.report.quick_wins),
        "report_url": _report_url(record.id),
    }

@app.get("/", response_class=FileResponse)
def index() -> FileResponse:
    return FileResponse(FRONTEND_DIR / "index.html")


@app.get("/dashboard", response_class=FileResponse)
def dashboard() -> FileResponse:
    return FileResponse(FRONTEND_DIR / "dashboard.html")


@app.get("/health")
def health() -> dict[str, object]:
    runtime = get_llm_runtime()
    return {
        "status": "ok",
        "provider": runtime["provider"],
        "model": runtime["model"],
        "credentials_configured": is_provider_configured(),
    }


@app.get("/report/{report_id}", response_class=FileResponse)
def report_view(report_id: str) -> FileResponse:
    if not report_exists(report_id):
        raise HTTPException(status_code=404, detail="Report not found.")
    return FileResponse(FRONTEND_DIR / "report.html")


@app.get("/api/report/{report_id}", responses={404: {"model": ErrorResponse}})
def get_report(report_id: str) -> dict[str, object]:
    record = get_persisted_report(report_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Report not found.")

    return {
        "report_id": report_id,
        "created_at": record.created_at,
        "request": record.request.model_dump(),
        "source": record.source.model_dump(),
        "report": record.report.model_dump(),
    }


@app.get("/api/reports")
def list_reports() -> dict[str, object]:
    items = list_persisted_reports()
    return {
        "dashboard_url": _dashboard_url(),
        "count": len(items),
        "reports": [_report_summary(record) for record in items],
    }


@app.post("/analyze", response_model=AnalyzeResponse, responses={400: {"model": ErrorResponse}, 502: {"model": ErrorResponse}})
def analyze(request: AnalyzeRequest) -> AnalyzeResponse:
    try:
        source = scrape_url(request.url) if request.url else build_raw_text_page(request.raw_text or "", request.business_context)
    except (ValueError, httpx.HTTPError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    try:
        user_prompt = build_user_prompt(source, request.business_context)
        runtime = resolve_runtime_overrides(
            request.llm_provider,
            request.model,
            request.api_key,
        )
        report = analyze_with_llm(
            SYSTEM_PROMPT,
            user_prompt,
            provider_override=runtime["provider"],
            model_override=runtime["model"],
            api_key_override=runtime["api_key"],
        )
    except LLMClientError as exc:
        status_code = 429 if exc.context.get("status_code") == 429 else 502
        raise HTTPException(status_code=status_code, detail={"message": exc.message, "context": exc.context}) from exc
    except ValidationError as exc:
        raise HTTPException(status_code=502, detail={"message": "Invalid LLM response.", "context": exc.errors()}) from exc

    report_id = str(uuid4())
    stored_request = AnalyzeRequest(
        url=request.url,
        raw_text=request.raw_text,
        business_context=request.business_context,
        llm_provider=runtime["provider"],
        model=runtime["model"],
    )
    record = ReportRecord(
        id=report_id,
        report=report,
        request=stored_request,
        source=source,
        created_at=datetime.now(timezone.utc).isoformat(),
    )
    save_report(record)

    return AnalyzeResponse(report_id=report_id, report_url=_report_url(report_id), report=report)


if __name__ == "__main__":
    import uvicorn

    port = int(get_env("PORT", "8000") or "8000")
    uvicorn.run("backend.main:app", host="0.0.0.0", port=port, reload=True)
