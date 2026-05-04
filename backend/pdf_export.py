from __future__ import annotations

from io import BytesIO

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from backend.models import ReportRecord


def _escape(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def _bullet_list(items: list[str], style: ParagraphStyle) -> list[Paragraph]:
    return [Paragraph(f"&bull; {_escape(item)}", style) for item in items]


def build_report_pdf(record: ReportRecord) -> bytes:
    buffer = BytesIO()
    document = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        leftMargin=18 * mm,
        rightMargin=18 * mm,
        topMargin=18 * mm,
        bottomMargin=18 * mm,
        title=f"SiteAudit report {record.id}",
    )

    styles = getSampleStyleSheet()
    title = ParagraphStyle(
        "ReportTitle",
        parent=styles["Heading1"],
        fontName="Helvetica-Bold",
        fontSize=26,
        leading=30,
        textColor=colors.HexColor("#111827"),
        spaceAfter=8,
    )
    h2 = ParagraphStyle(
        "SectionTitle",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=14,
        leading=18,
        textColor=colors.HexColor("#111827"),
        spaceBefore=12,
        spaceAfter=8,
    )
    body = ParagraphStyle(
        "Body",
        parent=styles["BodyText"],
        fontName="Helvetica",
        fontSize=10.5,
        leading=15,
        textColor=colors.HexColor("#374151"),
        alignment=TA_LEFT,
    )
    muted = ParagraphStyle(
        "Muted",
        parent=body,
        textColor=colors.HexColor("#6B7280"),
        fontSize=9.5,
        leading=13.5,
    )
    bullet = ParagraphStyle(
        "Bullet",
        parent=body,
        leftIndent=10,
        firstLineIndent=0,
        spaceAfter=4,
    )

    source_label = record.source.source_url or "Direct text input"
    focus_label = record.request.focus_page_label or "Not provided"
    focus_url = record.request.focus_page_url or "Not provided"
    special_attention = record.request.special_attention or "Not provided"

    story: list = [
        Paragraph("SiteAudit AI", muted),
        Paragraph("Website audit report", title),
        Paragraph(_escape(record.report.executive_summary), body),
        Spacer(1, 8),
        Paragraph(
            _escape(
                f"Source: {source_label} | Created: {record.created_at} | "
                f"Provider: {record.request.llm_provider or 'default'} | Model: {record.request.model or 'default'}"
            ),
            muted,
        ),
        Spacer(1, 14),
    ]

    metrics_table = Table(
        [
            ["Overall score", f"{record.report.overall_score:.1f} / 10"],
            ["Critical issues", str(len(record.report.critical_issues))],
            ["Quick wins", str(len(record.report.quick_wins))],
            ["Focused page", focus_label],
        ],
        colWidths=[48 * mm, 110 * mm],
    )
    metrics_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F9FAFB")),
                ("TEXTCOLOR", (0, 0), (-1, -1), colors.HexColor("#111827")),
                ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
                ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 10),
                ("ROWBACKGROUNDS", (0, 0), (-1, -1), [colors.HexColor("#FFFFFF"), colors.HexColor("#F9FAFB")]),
                ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#D1D5DB")),
                ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#E5E7EB")),
                ("LEFTPADDING", (0, 0), (-1, -1), 8),
                ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                ("TOPPADDING", (0, 0), (-1, -1), 7),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
            ]
        )
    )
    story.extend([metrics_table, Spacer(1, 10)])

    story.extend(
        [
            Paragraph("Audit context", h2),
            Paragraph(_escape(f"Business context: {record.request.business_context or 'Not provided'}"), body),
            Paragraph(_escape(f"Focus page URL: {focus_url}"), body),
            Paragraph(_escape(f"Special attention: {special_attention}"), body),
        ]
    )

    story.append(Paragraph("Dimension scores", h2))
    score_rows = [["Dimension", "Score", "Rationale"]]
    for label, detail in record.report.scores.model_dump().items():
        score_rows.append(
            [
                label.replace("_", " ").title(),
                f"{float(detail['score']):.1f}",
                detail["rationale"],
            ]
        )
    score_table = Table(score_rows, colWidths=[52 * mm, 20 * mm, 98 * mm], repeatRows=1)
    score_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#F3F4F6")),
                ("TEXTCOLOR", (0, 0), (-1, -1), colors.HexColor("#111827")),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#E5E7EB")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )
    story.extend([score_table, Spacer(1, 8)])

    story.append(Paragraph("Evidence highlights", h2))
    if record.report.evidence:
        for item in record.report.evidence:
            story.extend(
                [
                    Paragraph(f"<b>{_escape(item.source)}</b>", body),
                    Paragraph(f"“{_escape(item.excerpt)}”", muted),
                    Paragraph(_escape(item.why_it_matters), body),
                    Spacer(1, 6),
                ]
            )
    else:
        story.append(Paragraph("No evidence items were returned in this audit.", muted))

    story.append(Paragraph("Visual findings", h2))
    if record.report.visual_findings:
        for item in record.report.visual_findings:
            story.extend(
                [
                    Paragraph(f"<b>{_escape(item.area)}</b>", body),
                    Paragraph(_escape(item.observation), body),
                    Paragraph(f"<b>Impact:</b> {_escape(item.impact)}", body),
                    Paragraph(f"<b>Recommendation:</b> {_escape(item.recommendation)}", body),
                    Spacer(1, 6),
                ]
            )
    else:
        story.append(Paragraph("No screenshot-based visual findings were returned in this audit.", muted))

    story.append(Paragraph("Critical issues", h2))
    if record.report.critical_issues:
        for issue in record.report.critical_issues:
            story.extend(
                [
                    Paragraph(f"<b>{_escape(issue.title)} ({_escape(issue.severity.value)})</b>", body),
                    Paragraph(_escape(issue.description), body),
                    Paragraph(f"<b>Recommendation:</b> {_escape(issue.recommendation)}", body),
                    Spacer(1, 6),
                ]
            )
    else:
        story.append(Paragraph("No critical issues were returned in this audit.", muted))

    story.append(Paragraph("Quick wins", h2))
    if record.report.quick_wins:
        for win in record.report.quick_wins:
            story.extend(
                [
                    Paragraph(
                        f"<b>{_escape(win.action)}</b> ({_escape(win.effort.value)} effort)",
                        body,
                    ),
                    Paragraph(_escape(win.estimated_impact), body),
                    Spacer(1, 6),
                ]
            )
    else:
        story.append(Paragraph("No quick wins were returned in this audit.", muted))

    story.append(Paragraph("Competitive positioning note", h2))
    story.append(Paragraph(_escape(record.report.competitive_positioning_note), body))

    if record.report.special_focus is not None:
        story.append(Paragraph("Special attention area", h2))
        story.append(Paragraph(_escape(record.report.special_focus.assessment), body))
        story.append(Spacer(1, 4))
        story.extend(_bullet_list(record.report.special_focus.friction_points, bullet))
        story.append(Spacer(1, 4))
        story.extend(_bullet_list(record.report.special_focus.recommended_improvements, bullet))

    document.build(story)
    return buffer.getvalue()
