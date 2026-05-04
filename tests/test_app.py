from __future__ import annotations

from fastapi.testclient import TestClient

from backend.main import app
from backend.models import AuditReport
from backend.storage import clear_reports


client = TestClient(app)


def sample_report() -> AuditReport:
    return AuditReport.model_validate(
        {
            "executive_summary": "The site is clear enough to understand but lacks urgency and proof.",
            "overall_score": 6.4,
            "scores": {
                "value_proposition": {"score": 6, "rationale": "The offer is understandable but not differentiated."},
                "messaging_tone": {"score": 7, "rationale": "The tone broadly fits a professional audience."},
                "ux_structure": {"score": 6, "rationale": "The page flow is usable but not sharp."},
                "cta_effectiveness": {"score": 5, "rationale": "Calls-to-action are visible but generic."},
                "trust_signals": {"score": 7, "rationale": "Some proof exists, but it could be stronger."},
                "seo_readiness": {"score": 7, "rationale": "Basic metadata and structure are present."},
            },
            "critical_issues": [
                {
                    "title": "Weak CTA language",
                    "severity": "MED",
                    "description": "The main CTA does not create much momentum.",
                    "recommendation": "Use action-specific CTA copy tied to the core outcome.",
                }
            ],
            "quick_wins": [
                {
                    "action": "Rewrite the hero CTA",
                    "estimated_impact": "Higher click-through on the primary action",
                    "effort": "LOW",
                }
            ],
            "competitive_positioning_note": "The business feels credible, but sharper positioning would help it stand out.",
        }
    )


def teardown_function() -> None:
    clear_reports()


def test_health_endpoint() -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.json()["provider"] == "anthropic"
    assert response.json()["model"] == "claude-sonnet-4-20250514"
    assert response.json()["credentials_configured"] is False


def test_index_page_loads() -> None:
    response = client.get("/")

    assert response.status_code == 200
    assert "SiteAudit AI" in response.text
    assert "/dashboard" in response.text
    assert "Session API key" in response.text


def test_dashboard_page_loads() -> None:
    response = client.get("/dashboard")

    assert response.status_code == 200
    assert "Audit overview" in response.text
    assert "Most used provider" in response.text


def test_analyze_requires_input() -> None:
    response = client.post("/analyze", json={})

    assert response.status_code == 422


def test_analyze_raw_text_success(monkeypatch) -> None:
    def fake_analyze_with_llm(system_prompt: str, user_prompt: str, **kwargs) -> AuditReport:
        assert "Business Context: B2B SaaS for HR teams" in user_prompt
        assert "Page Body (truncated to 3000 chars):" in user_prompt
        assert kwargs["provider_override"] == "openai"
        assert kwargs["model_override"] == "gpt-4.1-mini"
        assert kwargs["api_key_override"] == "session-key"
        return sample_report()

    monkeypatch.setattr("backend.main.analyze_with_llm", fake_analyze_with_llm)

    response = client.post(
        "/analyze",
        json={
            "raw_text": "We help HR teams automate onboarding and compliance workflows.",
            "business_context": "B2B SaaS for HR teams",
            "llm_provider": "openai",
            "model": "gpt-4.1-mini",
            "api_key": "session-key",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["report"]["overall_score"] == 6.4
    assert payload["report_id"]
    assert payload["report_url"].startswith("/report/")


def test_report_routes_return_saved_report(monkeypatch) -> None:
    monkeypatch.setattr("backend.main.analyze_with_llm", lambda *_args, **_kwargs: sample_report())

    create_response = client.post(
        "/analyze",
        json={"raw_text": "A short homepage draft for testing."},
    )
    report_id = create_response.json()["report_id"]

    page_response = client.get(f"/report/{report_id}")
    api_response = client.get(f"/api/report/{report_id}")

    assert page_response.status_code == 200
    assert api_response.status_code == 200
    assert api_response.json()["report_id"] == report_id
    assert api_response.json()["report"]["executive_summary"]
    assert "api_key" not in api_response.json()["request"]
    assert "Audit heatmap" in page_response.text
    assert "Export PDF" in page_response.text


def test_reports_dashboard_api_returns_newest_first(monkeypatch) -> None:
    monkeypatch.setattr("backend.main.analyze_with_llm", lambda *_args, **_kwargs: sample_report())

    first = client.post("/analyze", json={"raw_text": "First draft."}).json()
    second = client.post("/analyze", json={"raw_text": "Second draft."}).json()

    response = client.get("/api/reports")

    assert response.status_code == 200
    payload = response.json()
    assert payload["count"] == 2
    assert payload["reports"][0]["report_id"] == second["report_id"]
    assert payload["reports"][1]["report_id"] == first["report_id"]
    assert payload["reports"][0]["report_url"] == f"/report/{second['report_id']}"
    assert payload["reports"][0]["provider"] == "anthropic"
    assert payload["reports"][0]["model"] == "claude-sonnet-4-20250514"


def test_reports_persist_across_client_instances(monkeypatch) -> None:
    monkeypatch.setattr("backend.main.analyze_with_llm", lambda *_args, **_kwargs: sample_report())

    create_response = client.post("/analyze", json={"raw_text": "Persistence check."})
    report_id = create_response.json()["report_id"]

    fresh_client = TestClient(app)
    persisted_response = fresh_client.get(f"/api/report/{report_id}")
    listing_response = fresh_client.get("/api/reports")

    assert persisted_response.status_code == 200
    assert listing_response.status_code == 200
    assert persisted_response.json()["report_id"] == report_id
    assert listing_response.json()["count"] >= 1


def test_report_not_found() -> None:
    response = client.get("/api/report/missing")

    assert response.status_code == 404
    assert response.json()["detail"] == "Report not found."


def test_health_endpoint_can_report_openai_provider(monkeypatch) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_MODEL", "gpt-4.1-mini")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["provider"] == "openai"
    assert response.json()["model"] == "gpt-4.1-mini"
    assert response.json()["credentials_configured"] is True
