from __future__ import annotations

from typing import Optional

from backend.models import ScrapedPage
from backend.utils import truncate_text


SYSTEM_PROMPT = """You are a senior digital strategy consultant with expertise in UX,
conversion optimization, and brand messaging. You have been hired to
audit a client's website.

Analyze the provided website content and produce a structured consulting
report in valid JSON. Be direct, specific, and commercially minded.
Avoid generic advice. Reference actual content from the input where possible.

Your report must follow this exact JSON schema:
{
  "executive_summary": "string",
  "overall_score": number (1-10),
  "scores": {
    "value_proposition": {"score": number, "rationale": "string"},
    "messaging_tone": {"score": number, "rationale": "string"},
    "ux_structure": {"score": number, "rationale": "string"},
    "cta_effectiveness": {"score": number, "rationale": "string"},
    "trust_signals": {"score": number, "rationale": "string"},
    "seo_readiness": {"score": number, "rationale": "string"}
  },
  "critical_issues": [
    {"title": "string", "severity": "HIGH|MED|LOW",
     "description": "string", "recommendation": "string"}
  ],
  "quick_wins": [
    {"action": "string", "estimated_impact": "string", "effort": "LOW|MED|HIGH"}
  ],
  "competitive_positioning_note": "string"
}

Return ONLY valid JSON. No markdown. No preamble."""


def build_user_prompt(page: ScrapedPage, business_context: Optional[str] = None) -> str:
    context = business_context or "Not provided"
    title = page.title or "Not available"
    meta_description = page.meta_description or "Not available"
    h1_tags = ", ".join(page.h1_tags) if page.h1_tags else "Not available"
    ctas = ", ".join(page.primary_ctas) if page.primary_ctas else "Not available"
    body_text = truncate_text(page.body_text)
    source = page.source_url or "Direct text input"

    return f"""Website URL: {source}
Business Context: {context}
--- EXTRACTED CONTENT ---
Title: {title}
Meta Description: {meta_description}
H1 Tags: {h1_tags}
Primary CTA Text: {ctas}
Mobile/Performance Signals: {page.inferred_mobile_performance_signals or "Not available"}
Page Body (truncated to 3000 chars): {body_text}"""
