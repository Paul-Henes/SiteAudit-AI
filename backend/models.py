from __future__ import annotations

from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Severity(str, Enum):
    HIGH = "HIGH"
    MED = "MED"
    LOW = "LOW"


class Effort(str, Enum):
    LOW = "LOW"
    MED = "MED"
    HIGH = "HIGH"


class ScoreDetail(BaseModel):
    model_config = ConfigDict(extra="forbid")

    score: float = Field(..., ge=1, le=10)
    rationale: str = Field(..., min_length=1)


class AuditScores(BaseModel):
    model_config = ConfigDict(extra="forbid")

    value_proposition: ScoreDetail
    messaging_tone: ScoreDetail
    ux_structure: ScoreDetail
    cta_effectiveness: ScoreDetail
    trust_signals: ScoreDetail
    seo_readiness: ScoreDetail


class CriticalIssue(BaseModel):
    model_config = ConfigDict(extra="forbid")

    title: str = Field(..., min_length=1)
    severity: Severity
    description: str = Field(..., min_length=1)
    recommendation: str = Field(..., min_length=1)


class QuickWin(BaseModel):
    model_config = ConfigDict(extra="forbid")

    action: str = Field(..., min_length=1)
    estimated_impact: str = Field(..., min_length=1)
    effort: Effort


class EvidenceItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source: str = Field(..., min_length=1)
    excerpt: str = Field(..., min_length=1)
    why_it_matters: str = Field(..., min_length=1)


class VisualFinding(BaseModel):
    model_config = ConfigDict(extra="forbid")

    area: str = Field(..., min_length=1)
    observation: str = Field(..., min_length=1)
    impact: str = Field(..., min_length=1)
    recommendation: str = Field(..., min_length=1)


class SpecialFocus(BaseModel):
    model_config = ConfigDict(extra="forbid")

    target_page: str = Field(..., min_length=1)
    attention_area: str = Field(..., min_length=1)
    assessment: str = Field(..., min_length=1)
    friction_points: list[str] = Field(..., min_length=1)
    recommended_improvements: list[str] = Field(..., min_length=1)


class AuditReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    executive_summary: str = Field(..., min_length=1)
    overall_score: float = Field(..., ge=1, le=10)
    scores: AuditScores
    evidence: list[EvidenceItem] = Field(default_factory=list)
    visual_findings: list[VisualFinding] = Field(default_factory=list)
    critical_issues: list[CriticalIssue]
    quick_wins: list[QuickWin]
    competitive_positioning_note: str = Field(..., min_length=1)
    special_focus: Optional[SpecialFocus] = None


class AnalyzeRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    url: Optional[str] = None
    raw_text: Optional[str] = None
    business_context: Optional[str] = None
    focus_page_url: Optional[str] = None
    focus_page_label: Optional[str] = None
    special_attention: Optional[str] = None
    llm_provider: Optional[str] = None
    model: Optional[str] = None
    api_key: Optional[str] = Field(default=None, exclude=True, repr=False)

    @model_validator(mode="after")
    def validate_input(self) -> "AnalyzeRequest":
        if not (self.url or self.raw_text):
            raise ValueError("Provide either a website URL or raw text.")
        return self


class AnalyzeResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    report_id: str
    report_url: str
    report: AuditReport


class ScrapedPage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_url: Optional[str] = None
    title: str = ""
    meta_description: str = ""
    h1_tags: list[str] = Field(default_factory=list)
    primary_ctas: list[str] = Field(default_factory=list)
    body_text: str = ""
    inferred_mobile_performance_signals: str = ""


class ReportRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    report: AuditReport
    request: AnalyzeRequest
    source: ScrapedPage
    created_at: str


class ErrorResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    detail: Any
    context: Optional[dict[str, Any]] = None
