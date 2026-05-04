from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse
from uuid import uuid4

import httpx
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import ValidationError

from backend.config import get_env
from backend.llm_client import LLMClientError, analyze_with_llm, get_llm_runtime, is_provider_configured, resolve_runtime_overrides
from backend.models import AnalyzeRequest, AnalyzeResponse, ErrorResponse, ReportRecord
from backend.pdf_export import build_report_pdf
from backend.prompt_builder import SYSTEM_PROMPT, build_user_prompt
from backend.scraper import build_raw_text_page, scrape_url
from backend.storage import get_report as get_persisted_report
from backend.storage import init_db, list_reports as list_persisted_reports
from backend.storage import report_exists, save_report
from backend.visual_capture import capture_url_screenshot


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


def _normalize_domain(url: Optional[str]) -> Optional[str]:
    if not url:
        return None
    parsed = urlparse(url)
    hostname = parsed.netloc or parsed.path
    hostname = hostname.lower().strip()
    if hostname.startswith("www."):
        hostname = hostname[4:]
    return hostname or None


def _matches_query(record: ReportRecord, query: str) -> bool:
    haystack = " ".join(
        [
            record.source.source_url or "",
            record.source.title or "",
            record.request.business_context or "",
            record.report.executive_summary or "",
            record.request.focus_page_label or "",
            record.request.special_attention or "",
        ]
    ).lower()
    return query.lower() in haystack


def _filter_reports(
    records: list[ReportRecord],
    *,
    query: Optional[str] = None,
    provider: Optional[str] = None,
    min_score: Optional[float] = None,
    max_score: Optional[float] = None,
    focus_label: Optional[str] = None,
    domain: Optional[str] = None,
) -> list[ReportRecord]:
    items = records
    if query:
        items = [record for record in items if _matches_query(record, query)]
    if provider:
        items = [
            record
            for record in items
            if (record.request.llm_provider or "").lower() == provider.lower()
        ]
    if min_score is not None:
        items = [record for record in items if float(record.report.overall_score) >= min_score]
    if max_score is not None:
        items = [record for record in items if float(record.report.overall_score) <= max_score]
    if focus_label:
        items = [
            record
            for record in items
            if (record.request.focus_page_label or "").lower() == focus_label.lower()
        ]
    if domain:
        normalized = domain.lower()
        items = [
            record
            for record in items
            if (_normalize_domain(record.source.source_url) or "") == normalized
        ]
    return items


def _report_summary(record: ReportRecord) -> dict[str, object]:
    domain = _normalize_domain(record.source.source_url)
    return {
        "report_id": record.id,
        "created_at": record.created_at,
        "domain": domain,
        "source_url": record.source.source_url,
        "source_title": record.source.title,
        "input_type": "url" if record.request.url else "raw_text",
        "business_context": record.request.business_context,
        "provider": record.request.llm_provider,
        "model": record.request.model,
        "focus_page_label": record.request.focus_page_label,
        "overall_score": record.report.overall_score,
        "executive_summary": record.report.executive_summary,
        "evidence_count": len(record.report.evidence),
        "visual_findings_count": len(record.report.visual_findings),
        "critical_issues_count": len(record.report.critical_issues),
        "quick_wins_count": len(record.report.quick_wins),
        "report_url": _report_url(record.id),
        "pdf_url": f"/api/report/{record.id}/pdf",
    }


def _list_filter_metadata(records: list[ReportRecord]) -> dict[str, list[str]]:
    providers = sorted(
        {
            record.request.llm_provider
            for record in records
            if record.request.llm_provider
        }
    )
    domains = sorted(
        {
            domain
            for domain in (_normalize_domain(record.source.source_url) for record in records)
            if domain
        }
    )
    focus_labels = sorted(
        {
            record.request.focus_page_label
            for record in records
            if record.request.focus_page_label
        }
    )
    return {"providers": providers, "domains": domains, "focus_page_labels": focus_labels}

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


@app.get("/api/report/{report_id}/pdf", responses={404: {"model": ErrorResponse}})
def get_report_pdf(report_id: str) -> Response:
    record = get_persisted_report(report_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Report not found.")
    pdf_bytes = build_report_pdf(record)
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="siteaudit-{report_id}.pdf"'},
    )


@app.get("/api/reports")
def list_reports(
    q: Optional[str] = Query(default=None),
    provider: Optional[str] = Query(default=None),
    min_score: Optional[float] = Query(default=None, ge=1, le=10),
    max_score: Optional[float] = Query(default=None, ge=1, le=10),
    focus_label: Optional[str] = Query(default=None),
    domain: Optional[str] = Query(default=None),
) -> dict[str, object]:
    all_items = list_persisted_reports()
    items = _filter_reports(
        all_items,
        query=q,
        provider=provider,
        min_score=min_score,
        max_score=max_score,
        focus_label=focus_label,
        domain=domain.lower() if domain else None,
    )
    return {
        "dashboard_url": _dashboard_url(),
        "filters": _list_filter_metadata(all_items),
        "count": len(items),
        "reports": [_report_summary(record) for record in items],
    }


@app.get("/api/reports/compare", responses={404: {"model": ErrorResponse}})
def compare_reports(domain: str = Query(..., min_length=1)) -> dict[str, object]:
    normalized_domain = domain.lower().strip()
    items = _filter_reports(
        list_persisted_reports(),
        domain=normalized_domain,
    )
    if not items:
        raise HTTPException(status_code=404, detail="No reports found for that domain.")

    ordered = sorted(items, key=lambda record: record.created_at)
    first = ordered[0]
    latest = ordered[-1]
    return {
        "domain": normalized_domain,
        "count": len(ordered),
        "reports": [_report_summary(record) for record in ordered],
        "comparison": {
            "first_report_id": first.id,
            "latest_report_id": latest.id,
            "overall_score_delta": round(float(latest.report.overall_score) - float(first.report.overall_score), 1),
            "critical_issues_delta": len(latest.report.critical_issues) - len(first.report.critical_issues),
            "quick_wins_delta": len(latest.report.quick_wins) - len(first.report.quick_wins),
        },
    }


@app.post("/analyze", response_model=AnalyzeResponse, responses={400: {"model": ErrorResponse}, 502: {"model": ErrorResponse}})
def analyze(request: AnalyzeRequest) -> AnalyzeResponse:
    try:
        source = scrape_url(request.url) if request.url else build_raw_text_page(request.raw_text or "", request.business_context)
        focus_source = scrape_url(request.focus_page_url) if request.focus_page_url else None
    except (ValueError, httpx.HTTPError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    visual_inputs = []
    if request.url:
        main_snapshot = capture_url_screenshot(request.url, "Main page")
        if main_snapshot is not None:
            visual_inputs.append(main_snapshot)
    if request.focus_page_url:
        focus_snapshot = capture_url_screenshot(
            request.focus_page_url,
            request.focus_page_label or "Focus page",
        )
        if focus_snapshot is not None:
            visual_inputs.append(focus_snapshot)

    try:
        user_prompt = build_user_prompt(
            source,
            request.business_context,
            focus_page=focus_source,
            focus_page_label=request.focus_page_label,
            special_attention=request.special_attention,
        )
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
            visual_inputs=visual_inputs or None,
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
        focus_page_url=request.focus_page_url,
        focus_page_label=request.focus_page_label,
        special_attention=request.special_attention,
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
