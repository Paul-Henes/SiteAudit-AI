from __future__ import annotations

from fastapi.testclient import TestClient

from backend.llm_client import _coerce_audit_payload
from backend.main import app
from backend.models import AuditReport, ScrapedPage
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
            "evidence": [
                {
                    "source": "Hero headline",
                    "excerpt": "Automate onboarding and compliance workflows",
                    "why_it_matters": "The offer is understandable, but the outcome is still broad and not strongly differentiated.",
                },
                {
                    "source": "Primary CTA",
                    "excerpt": "Start now",
                    "why_it_matters": "The action is visible, but the wording does not lower hesitation or clarify what happens next.",
                },
            ],
            "visual_findings": [
                {
                    "area": "Hero hierarchy",
                    "observation": "The page introduces the product quickly, but the headline and supporting copy compete too evenly for attention.",
                    "impact": "Visitors have to parse more before understanding the main value proposition.",
                    "recommendation": "Create a stronger visual separation between the main promise and supporting context.",
                }
            ],
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
            "special_focus": {
                "target_page": "Exercise page",
                "attention_area": "Drop-off and conversion friction",
                "assessment": "The exercise page carries intent, but the messaging and action flow create too much hesitation before a visitor commits.",
                "friction_points": [
                    "The page does not quickly explain what happens next after the user clicks through.",
                    "The CTA language is not strong enough for the level of commitment being asked."
                ],
                "recommended_improvements": [
                    "Clarify the immediate outcome and next step above the fold.",
                    "Use a more explicit CTA that reduces uncertainty and increases commitment."
                ]
            },
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
    assert "Domain comparison" in response.text


def test_analyze_requires_input() -> None:
    response = client.post("/analyze", json={})

    assert response.status_code == 422


def test_analyze_raw_text_success(monkeypatch) -> None:
    def fake_analyze_with_llm(system_prompt: str, user_prompt: str, **kwargs) -> AuditReport:
        assert "Business Context: B2B SaaS for HR teams" in user_prompt
        assert "Focus Page Label: Exercise page" in user_prompt
        assert "Special Attention Request: Why users may drop off on the exercise page" in user_prompt
        assert "SCORING CALIBRATION" in user_prompt
        assert "STYLE EXAMPLES" in user_prompt
        assert "VISUAL ANALYSIS GUIDANCE" in user_prompt
        assert "Pay extra attention to user hesitation" in user_prompt
        assert "--- EXERCISE PAGE ---" in user_prompt
        assert kwargs["provider_override"] == "openai"
        assert kwargs["model_override"] == "gpt-4.1-mini"
        assert kwargs["api_key_override"] == "session-key"
        assert len(kwargs["visual_inputs"]) == 1
        assert kwargs["visual_inputs"][0].label == "Exercise page"
        return sample_report()

    monkeypatch.setattr("backend.main.analyze_with_llm", fake_analyze_with_llm)
    monkeypatch.setattr(
        "backend.main.scrape_url",
        lambda url: ScrapedPage(
            source_url=url,
            title="Exercise page",
            meta_description="Focused conversion page",
            h1_tags=["Start your exercise"],
            primary_ctas=["Start now"],
            body_text="Exercise page copy and supporting details.",
            inferred_mobile_performance_signals="viewport meta tag present",
        ),
    )
    monkeypatch.setattr(
        "backend.main.capture_url_screenshot",
        lambda url, label: type(
            "Snapshot",
            (),
            {"label": label, "media_type": "image/png", "base64_data": "dGVzdA==", "data_url": "data:image/png;base64,dGVzdA=="},
        )(),
    )

    response = client.post(
        "/analyze",
        json={
            "raw_text": "We help HR teams automate onboarding and compliance workflows.",
            "business_context": "B2B SaaS for HR teams",
            "focus_page_url": "https://example.com/exercise",
            "focus_page_label": "Exercise page",
            "special_attention": "Why users may drop off on the exercise page",
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
    assert payload["report"]["special_focus"]["target_page"] == "Exercise page"
    assert payload["report"]["evidence"][0]["source"] == "Hero headline"


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
    assert "Download PDF" in page_response.text
    assert "Special attention area" in page_response.text
    assert api_response.json()["report"]["visual_findings"][0]["area"] == "Hero hierarchy"


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
    assert payload["reports"][0]["pdf_url"] == f"/api/report/{second['report_id']}/pdf"
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


def test_coerce_audit_payload_normalizes_common_openai_schema_drift() -> None:
    payload = _coerce_audit_payload(
        {
            "executive_summary": "Clear offer, but weak proof and CTA momentum.",
            "overall_score": "6.7",
            "scores": {
                "value_proposition": {"score": "6", "rationale": "Mostly clear."},
                "messaging_tone": {"score": "7", "rationale": "Reasonably aligned."},
                "ux_structure": {"score": "6", "rationale": "Readable but not sharp."},
                "cta_effectiveness": {"score": "5", "rationale": "The CTA is generic."},
                "trust_signals": {"score": "6", "rationale": "Proof is limited."},
                "seo_readiness": {"score": "7", "rationale": "Surface structure exists."},
            },
            "evidence": {
                "source": "Hero",
                "excerpt": "Mostly clear.",
                "why_it_matters": "The message is still generic.",
            },
            "visual_findings": {
                "area": "Hero hierarchy",
                "observation": "The layout feels crowded.",
                "impact": "The promise is not immediately obvious.",
                "recommendation": "Increase whitespace around the main promise.",
            },
            "critical_issues": [
                {
                    "title": "Weak CTA",
                    "severity": "med",
                    "description": "The CTA does not create enough momentum.",
                    "recommendation": "Use a more specific next step.",
                }
            ],
            "quick_wins": [
                {
                    "action": "Rewrite hero CTA",
                    "estimated_impact": "Better click-through",
                    "effort": "low",
                }
            ],
            "competitive_positioning_note": "The market position feels credible but still generic.",
            "special_focus": {
                "target_page": "Exercise page",
                "attention_area": "Drop-off",
                "assessment": "There is hesitation before commitment.",
                "friction_points": "Users do not understand what happens next.",
                "recommended_improvements": "Clarify the next step above the fold.",
            },
        }
    )

    report = AuditReport.model_validate(payload)

    assert report.overall_score == 6.7
    assert report.scores.cta_effectiveness.score == 5
    assert report.critical_issues[0].severity.value == "MED"
    assert report.quick_wins[0].effort.value == "LOW"
    assert report.special_focus is not None
    assert report.evidence[0].source == "Hero"
    assert report.visual_findings[0].area == "Hero hierarchy"
    assert report.special_focus.friction_points == ["Users do not understand what happens next."]
    assert report.special_focus.recommended_improvements == ["Clarify the next step above the fold."]


def test_coerce_audit_payload_adds_null_special_focus_when_missing() -> None:
    payload = _coerce_audit_payload(sample_report().model_dump(exclude={"special_focus"}))

    report = AuditReport.model_validate(payload)

    assert report.special_focus is None


def test_report_pdf_endpoint_returns_pdf(monkeypatch) -> None:
    monkeypatch.setattr("backend.main.analyze_with_llm", lambda *_args, **_kwargs: sample_report())

    report_id = client.post("/analyze", json={"raw_text": "PDF export check."}).json()["report_id"]
    response = client.get(f"/api/report/{report_id}/pdf")

    assert response.status_code == 200
    assert response.headers["content-type"] == "application/pdf"
    assert response.content.startswith(b"%PDF")


def test_reports_dashboard_api_supports_filters_and_compare(monkeypatch) -> None:
    monkeypatch.setattr("backend.main.analyze_with_llm", lambda *_args, **_kwargs: sample_report())
    monkeypatch.setattr(
        "backend.main.scrape_url",
        lambda url: ScrapedPage(
            source_url=url,
            title="Example page",
            meta_description="Example description",
            h1_tags=["Example heading"],
            primary_ctas=["Start now"],
            body_text="Example page body",
            inferred_mobile_performance_signals="viewport meta tag present",
        ),
    )
    monkeypatch.setattr("backend.main.capture_url_screenshot", lambda *_args, **_kwargs: None)
    client.post(
        "/analyze",
        json={
            "url": "https://example.com",
            "focus_page_label": "Pricing page",
        },
    )
    client.post(
        "/analyze",
        json={
            "url": "https://example.com/pricing",
            "llm_provider": "openai",
            "model": "gpt-4.1-mini",
        },
    )

    filtered = client.get("/api/reports", params={"provider": "openai"})
    compare = client.get("/api/reports/compare", params={"domain": "example.com"})

    assert filtered.status_code == 200
    assert filtered.json()["count"] == 1
    assert filtered.json()["filters"]["domains"] == ["example.com"]
    assert compare.status_code == 200
    assert compare.json()["domain"] == "example.com"
    assert compare.json()["count"] == 2
