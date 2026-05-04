from __future__ import annotations

from typing import Optional

from backend.models import ScrapedPage
from backend.utils import truncate_text


SCORING_RUBRIC = """SCORING CALIBRATION
- 9-10 = exceptional: unusually clear, persuasive, and commercially sharp
- 7-8 = strong: clearly above average, but still has meaningful room to improve
- 5-6 = adequate: functional, but leaking clarity, trust, or conversion momentum
- 3-4 = weak: obvious strategic or conversion problems
- 1-2 = broken: severe messaging, UX, or trust failure

DIMENSION RUBRICS
- value_proposition:
  Assess whether the offer is immediately understandable, outcome-led, specific, and differentiated.
- messaging_tone:
  Assess whether the language matches the likely audience, level of sophistication, and buying intent.
- ux_structure:
  Assess hierarchy, scannability, flow, navigation cues, and whether the page makes the next step obvious.
- cta_effectiveness:
  Assess clarity of the primary action, CTA prominence, specificity of CTA copy, and friction from competing actions.
- trust_signals:
  Assess proof, credibility, reassurance, testimonials, authority, security, and risk-reduction cues.
- seo_readiness:
  Assess titles, meta description, headings, search intent alignment, and keyword clarity at a surface level.

AUDIT RULES
- Be commercially strict. Do not give balanced fluff.
- Prioritize what most affects conversion and buyer confidence.
- Reference actual page elements, copy, headings, CTAs, or missing signals wherever possible.
- Include 3-6 evidence items grounded in real copy, headings, CTA labels, or visible trust elements.
- Quote short excerpts where useful so the report feels auditable rather than generic.
- When recommending improvements, explain the highest-leverage fix first.
- If something is generic, say exactly why it is generic.
- If a focus page or attention request exists, use the special_focus section to directly answer it with practical recommendations.
- If screenshots are attached, use them to assess visual hierarchy, CTA prominence, layout clarity, whitespace, and trust presentation. Do not infer technical performance metrics from images alone.
- If screenshots are attached, include 2-4 visual_findings. If no screenshots are attached, return an empty visual_findings array."""


FEW_SHOT_EXAMPLES = """STYLE EXAMPLES

EXAMPLE 1
Context:
- Main page is a B2B SaaS homepage
- Headline is feature-led and generic
- CTA is repeated as "Learn more"
- No visible proof near the top of the page

Good audit behavior:
- Score value_proposition and cta_effectiveness harshly, not politely
- Call out that the headline does not communicate a business outcome
- Explicitly say that "Learn more" is weak because it creates low intent and no urgency
- Recommend a sharper hero rewrite and one clearly prioritized CTA

Example output characteristics:
- executive_summary is commercially blunt
- critical_issues prioritize value proposition and CTA hierarchy first
- quick_wins are specific, realistic, and tied to conversion impact
- evidence items quote the actual weak headline, CTA, or missing proof signal
- visual_findings stay empty if no screenshots are attached

EXAMPLE 2
Context:
- Main page introduces the product well enough
- Focus page is a pricing page
- Plans are visible but audience fit is unclear
- Visitor may hesitate because there is little reassurance around commitment
- Special attention request asks why users are not converting on pricing

Good audit behavior:
- Use special_focus to directly answer the pricing concern
- Explain where friction happens on the focus page, not just the homepage
- Name missing reassurance, weak plan framing, and unclear next-step confidence
- Recommend concrete improvements such as audience labels, outcome framing, and stronger pricing CTA copy
- Include evidence items that support the claim with visible wording or structure

Example output characteristics:
- special_focus.assessment directly addresses the user's attention request
- friction_points are concrete and page-specific
- recommended_improvements read like actions a growth or product team could implement this sprint
- evidence and visual_findings make the audit feel inspectable rather than hand-wavy"""


SYSTEM_PROMPT = """You are a senior digital strategy consultant with expertise in UX,
conversion optimization, and brand messaging. You have been hired to
audit a client's website.

Analyze the provided website content and produce a structured consulting
report in valid JSON. Be direct, specific, and commercially minded.
Avoid generic advice. Reference actual content from the input where possible.
Use the scoring calibration and dimension rubrics provided in the user prompt.
Be harder on weak messaging, weak CTA logic, and weak trust than a typical assistant.

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
  "evidence": [
    {"source": "string", "excerpt": "string", "why_it_matters": "string"}
  ],
  "visual_findings": [
    {"area": "string", "observation": "string", "impact": "string", "recommendation": "string"}
  ],
  "critical_issues": [
    {"title": "string", "severity": "HIGH|MED|LOW",
     "description": "string", "recommendation": "string"}
  ],
  "quick_wins": [
    {"action": "string", "estimated_impact": "string", "effort": "LOW|MED|HIGH"}
  ],
  "competitive_positioning_note": "string",
  "special_focus": null OR {
    "target_page": "string",
    "attention_area": "string",
    "assessment": "string",
    "friction_points": ["string"],
    "recommended_improvements": ["string"]
  }
}

If no focus page or special attention request is provided, set "special_focus" to null.
If a focus page or special attention request is provided, "special_focus" must be specific,
actionable, and directly address what needs to improve most in that area.
The evidence array should capture the strongest proof points behind your conclusions.
The visual_findings array should stay empty if no screenshots are attached.

Return ONLY valid JSON. No markdown. No preamble."""


def _infer_focus_guidance(
    focus_page: Optional[ScrapedPage],
    focus_page_label: Optional[str],
    special_attention: Optional[str],
) -> str:
    if focus_page is None and not special_attention:
        return "Not applicable."

    source = " ".join(
        part
        for part in [
            focus_page_label or "",
            focus_page.source_url if focus_page else "",
            special_attention or "",
        ]
        if part
    ).lower()

    if any(token in source for token in ["pricing", "plan", "subscription"]):
        return "Pay extra attention to plan clarity, value communication, objection handling, price anchoring, and CTA confidence."
    if any(token in source for token in ["exercise", "workout", "program", "lesson"]):
        return "Pay extra attention to user hesitation, clarity of the next step, motivation, expectation-setting, and conversion friction before commitment."
    if any(token in source for token in ["demo", "book", "call", "consultation"]):
        return "Pay extra attention to lead intent, friction before booking, clarity of the promised outcome, and reassurance around what happens next."
    if any(token in source for token in ["landing", "campaign", "ad"]):
        return "Pay extra attention to message match, focus, distraction, conversion flow, and how well the page supports one primary action."
    if any(token in source for token in ["about", "team", "company"]):
        return "Pay extra attention to credibility, trust-building narrative, authority, and whether the page earns belief."

    return "Pay extra attention to the journey from the main page into the focus page, the clarity of the focus page itself, and the specific friction described in the attention request."


def _format_page_block(label: str, page: ScrapedPage) -> str:
    title = page.title or "Not available"
    meta_description = page.meta_description or "Not available"
    h1_tags = ", ".join(page.h1_tags) if page.h1_tags else "Not available"
    ctas = ", ".join(page.primary_ctas) if page.primary_ctas else "Not available"
    body_text = truncate_text(page.body_text)
    source = page.source_url or "Direct text input"

    return f"""--- {label.upper()} ---
URL: {source}
Title: {title}
Meta Description: {meta_description}
H1 Tags: {h1_tags}
Primary CTA Text: {ctas}
Mobile/Performance Signals: {page.inferred_mobile_performance_signals or "Not available"}
Page Body (truncated to 3000 chars): {body_text}"""


def build_user_prompt(
    page: ScrapedPage,
    business_context: Optional[str] = None,
    focus_page: Optional[ScrapedPage] = None,
    focus_page_label: Optional[str] = None,
    special_attention: Optional[str] = None,
) -> str:
    context = business_context or "Not provided"
    focus_label = focus_page_label or "Focus page"
    focus_request = special_attention or "Not provided"
    focus_guidance = _infer_focus_guidance(focus_page, focus_page_label, special_attention)

    sections = [_format_page_block("Main page", page)]
    if focus_page is not None:
        sections.append(_format_page_block(focus_label, focus_page))

    return f"""Website URL: {page.source_url or "Direct text input"}
Business Context: {context}
Focus Page Label: {focus_label if focus_page is not None else "Not provided"}
Special Attention Request: {focus_request}
Focus Audit Guidance: {focus_guidance}

{SCORING_RUBRIC}

{FEW_SHOT_EXAMPLES}

VISUAL ANALYSIS GUIDANCE
- Screenshots may be attached for the main page and the focus page.
- Use screenshots to evaluate hierarchy, visual emphasis, CTA visibility, clutter, and trust presentation.
- Use the extracted text for exact wording and the screenshots for design, layout, and scannability judgments.

Analyze the main page as the anchor context. If a focus page is provided, pay special attention to:
1. how well the main page sets up the user for that page,
2. how strong the focus page itself is,
3. what specifically needs to improve in the focus area.

{chr(10).join(sections)}"""
