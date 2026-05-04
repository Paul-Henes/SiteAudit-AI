from __future__ import annotations

import json
import time
from typing import Any, Optional

import anthropic
from anthropic import Anthropic
import openai
from openai import OpenAI
from pydantic import BaseModel, ValidationError

from backend.config import get_env
from backend.models import AuditReport
from backend.visual_capture import VisualSnapshot


ANTHROPIC_MODEL_NAME = "claude-sonnet-4-20250514"
OPENAI_MODEL_NAME = "gpt-4.1"
MAX_TOKENS = 1800
RETRY_DELAY_SECONDS = 2


class LLMClientError(Exception):
    def __init__(self, message: str, *, context: Optional[dict[str, Any]] = None) -> None:
        super().__init__(message)
        self.message = message
        self.context = context or {}


def _get_provider() -> str:
    provider = (get_env("LLM_PROVIDER", "anthropic") or "anthropic").strip().lower()
    if provider not in {"anthropic", "openai"}:
        raise LLMClientError(
            "LLM_PROVIDER must be either 'anthropic' or 'openai'.",
            context={"provider": provider},
        )
    return provider


def _normalize_provider(provider: str) -> str:
    normalized = provider.strip().lower()
    if normalized not in {"anthropic", "openai"}:
        raise LLMClientError(
            "LLM provider must be either 'anthropic' or 'openai'.",
            context={"provider": provider},
        )
    return normalized


def get_llm_runtime() -> dict[str, str]:
    provider = _get_provider()
    model = (
        (get_env("ANTHROPIC_MODEL", ANTHROPIC_MODEL_NAME) or ANTHROPIC_MODEL_NAME).strip()
        if provider == "anthropic"
        else (get_env("OPENAI_MODEL", OPENAI_MODEL_NAME) or OPENAI_MODEL_NAME).strip()
    )
    return {"provider": provider, "model": model}


def is_provider_configured() -> bool:
    provider = _get_provider()
    if provider == "openai":
        return bool(get_env("OPENAI_API_KEY"))
    return bool(get_env("ANTHROPIC_API_KEY"))


def resolve_runtime_overrides(
    provider_override: Optional[str] = None,
    model_override: Optional[str] = None,
    api_key_override: Optional[str] = None,
) -> dict[str, str]:
    provider = _normalize_provider(provider_override) if provider_override else _get_provider()
    default_model = ANTHROPIC_MODEL_NAME if provider == "anthropic" else OPENAI_MODEL_NAME
    model = (model_override or get_env("ANTHROPIC_MODEL" if provider == "anthropic" else "OPENAI_MODEL", default_model) or default_model).strip()
    api_key = (api_key_override or get_env("ANTHROPIC_API_KEY" if provider == "anthropic" else "OPENAI_API_KEY") or "").strip()
    return {"provider": provider, "model": model, "api_key": api_key}


def _get_anthropic_client(api_key_override: Optional[str] = None) -> Anthropic:
    api_key = (api_key_override or get_env("ANTHROPIC_API_KEY") or "").strip()
    if not api_key:
        raise LLMClientError(
            "ANTHROPIC_API_KEY is not configured.",
            context={"hint": "Add ANTHROPIC_API_KEY to your environment before analyzing content."},
        )
    return Anthropic(api_key=api_key, max_retries=0)


def _get_openai_client(api_key_override: Optional[str] = None) -> OpenAI:
    api_key = (api_key_override or get_env("OPENAI_API_KEY") or "").strip()
    if not api_key:
        raise LLMClientError(
            "OPENAI_API_KEY is not configured.",
            context={"hint": "Add OPENAI_API_KEY to your environment before analyzing content."},
        )
    return OpenAI(api_key=api_key, max_retries=0)


def _extract_anthropic_text(response: Any) -> str:
    chunks: list[str] = []
    for block in response.content:
        if getattr(block, "type", "") == "text":
            chunks.append(block.text)
    return "".join(chunks).strip()


def _parse_report(response_text: str) -> AuditReport:
    payload = json.loads(response_text)
    return AuditReport.model_validate(_coerce_audit_payload(payload))


def _coerce_string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    text = str(value).strip()
    return [text] if text else []


def _coerce_enum_like(value: Any) -> Any:
    if value is None:
        return None
    if hasattr(value, "value"):
        return str(value.value).strip().upper()
    text = str(value).strip()
    if "." in text:
        text = text.split(".")[-1]
    return text.upper()


def _coerce_audit_payload(payload: Any) -> Any:
    if not isinstance(payload, dict):
        return payload

    normalized = dict(payload)

    try:
        if "overall_score" in normalized:
            normalized["overall_score"] = float(normalized["overall_score"])
    except (TypeError, ValueError):
        pass

    scores = normalized.get("scores")
    if isinstance(scores, dict):
        coerced_scores: dict[str, Any] = {}
        for key, value in scores.items():
            if isinstance(value, dict):
                score_detail = dict(value)
                try:
                    if "score" in score_detail:
                        score_detail["score"] = float(score_detail["score"])
                except (TypeError, ValueError):
                    pass
                coerced_scores[key] = score_detail
            else:
                coerced_scores[key] = value
        normalized["scores"] = coerced_scores

    evidence = normalized.get("evidence")
    if evidence is None:
        normalized["evidence"] = []
    elif not isinstance(evidence, list):
        normalized["evidence"] = [evidence]

    visual_findings = normalized.get("visual_findings")
    if visual_findings is None:
        normalized["visual_findings"] = []
    elif not isinstance(visual_findings, list):
        normalized["visual_findings"] = [visual_findings]

    issues = normalized.get("critical_issues")
    if issues is None:
        normalized["critical_issues"] = []
    elif isinstance(issues, list):
        coerced_issues: list[Any] = []
        for issue in issues:
            if isinstance(issue, dict):
                item = dict(issue)
                if "severity" in item and item["severity"] is not None:
                    item["severity"] = _coerce_enum_like(item["severity"])
                coerced_issues.append(item)
            else:
                coerced_issues.append(issue)
        normalized["critical_issues"] = coerced_issues

    quick_wins = normalized.get("quick_wins")
    if quick_wins is None:
        normalized["quick_wins"] = []
    elif isinstance(quick_wins, list):
        coerced_quick_wins: list[Any] = []
        for quick_win in quick_wins:
            if isinstance(quick_win, dict):
                item = dict(quick_win)
                if "effort" in item and item["effort"] is not None:
                    item["effort"] = _coerce_enum_like(item["effort"])
                coerced_quick_wins.append(item)
            else:
                coerced_quick_wins.append(quick_win)
        normalized["quick_wins"] = coerced_quick_wins

    if "special_focus" not in normalized:
        normalized["special_focus"] = None
    elif isinstance(normalized["special_focus"], dict):
        special_focus = dict(normalized["special_focus"])
        special_focus["friction_points"] = _coerce_string_list(special_focus.get("friction_points"))
        special_focus["recommended_improvements"] = _coerce_string_list(
            special_focus.get("recommended_improvements")
        )
        normalized["special_focus"] = special_focus

    return normalized


def _serialize_for_error(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return value.model_dump()
    if isinstance(value, dict):
        return value
    if isinstance(value, list):
        return value
    return str(value)


def analyze_with_claude(
    system_prompt: str,
    user_prompt: str,
    *,
    model_name: Optional[str] = None,
    api_key_override: Optional[str] = None,
    visual_inputs: Optional[list[VisualSnapshot]] = None,
) -> AuditReport:
    client = _get_anthropic_client(api_key_override)
    last_error: Optional[LLMClientError] = None

    for attempt in range(2):
        try:
            active_model = (model_name or get_env("ANTHROPIC_MODEL", ANTHROPIC_MODEL_NAME) or ANTHROPIC_MODEL_NAME).strip()
            content: list[dict[str, Any]] | str
            if visual_inputs:
                content_blocks: list[dict[str, Any]] = [{"type": "text", "text": user_prompt}]
                for snapshot in visual_inputs:
                    content_blocks.append({"type": "text", "text": f"{snapshot.label} screenshot"})
                    content_blocks.append(
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": snapshot.media_type,
                                "data": snapshot.base64_data,
                            },
                        }
                    )
                content = content_blocks
            else:
                content = user_prompt
            response = client.messages.create(
                model=active_model,
                max_tokens=MAX_TOKENS,
                system=system_prompt,
                messages=[{"role": "user", "content": content}],
            )

            response_text = _extract_anthropic_text(response)
            report = _parse_report(response_text)
            return report
        except anthropic.RateLimitError as exc:
            last_error = LLMClientError(
                "Anthropic rate limit exceeded while generating the audit.",
                context={"attempt": attempt + 1, "status_code": 429},
            )
        except (json.JSONDecodeError, ValidationError) as exc:
            last_error = LLMClientError(
                "Anthropic returned malformed audit JSON.",
                context={"attempt": attempt + 1, "error": str(exc)},
            )
        except anthropic.APIConnectionError as exc:
            last_error = LLMClientError(
                "Could not reach Anthropic API.",
                context={"attempt": attempt + 1, "error": str(exc)},
            )
        except anthropic.APIStatusError as exc:
            last_error = LLMClientError(
                "Anthropic API returned an unexpected status.",
                context={
                    "attempt": attempt + 1,
                    "status_code": exc.status_code,
                    "error": str(exc),
                },
            )
        except Exception as exc:
            last_error = LLMClientError(
                "Unexpected error while generating the audit.",
                context={"attempt": attempt + 1, "error": str(exc)},
            )

        if attempt == 0:
            time.sleep(RETRY_DELAY_SECONDS)

    if last_error is None:
        raise LLMClientError("Unknown error while generating the audit.")
    raise last_error


def analyze_with_openai(
    system_prompt: str,
    user_prompt: str,
    *,
    model_name: Optional[str] = None,
    api_key_override: Optional[str] = None,
    visual_inputs: Optional[list[VisualSnapshot]] = None,
) -> AuditReport:
    client = _get_openai_client(api_key_override)
    last_error: Optional[LLMClientError] = None

    for attempt in range(2):
        response: Any = None
        parsed_output: Any = None
        try:
            active_model = (model_name or get_env("OPENAI_MODEL", OPENAI_MODEL_NAME) or OPENAI_MODEL_NAME).strip()
            user_content: list[dict[str, Any]] = [{"type": "input_text", "text": user_prompt}]
            for snapshot in visual_inputs or []:
                user_content.append({"type": "input_text", "text": f"{snapshot.label} screenshot"})
                user_content.append(
                    {
                        "type": "input_image",
                        "image_url": snapshot.data_url,
                        "detail": "high",
                    }
                )
            response = client.responses.parse(
                model=active_model,
                instructions=system_prompt,
                input=[{"role": "user", "content": user_content}],
                max_output_tokens=MAX_TOKENS,
                text_format=AuditReport,
            )

            parsed_output = getattr(response, "output_parsed", None)
            if parsed_output is None:
                response_text = getattr(response, "output_text", "").strip()
                raise LLMClientError(
                    "OpenAI returned no structured audit payload.",
                    context={"attempt": attempt + 1, "response_text": response_text},
                )

            if isinstance(parsed_output, AuditReport):
                return parsed_output

            return AuditReport.model_validate(_coerce_audit_payload(parsed_output))
        except openai.RateLimitError:
            last_error = LLMClientError(
                "OpenAI rate limit exceeded while generating the audit.",
                context={"attempt": attempt + 1, "status_code": 429},
            )
        except ValidationError as exc:
            parsed_preview = None
            response_text = None
            try:
                parsed_preview = _serialize_for_error(parsed_output)  # type: ignore[name-defined]
            except Exception:
                parsed_preview = None
            try:
                response_text = getattr(response, "output_text", None)  # type: ignore[name-defined]
            except Exception:
                response_text = None
            last_error = LLMClientError(
                "OpenAI returned malformed audit JSON.",
                context={
                    "attempt": attempt + 1,
                    "error": str(exc),
                    "parsed_output": parsed_preview,
                    "response_text": response_text,
                },
            )
        except openai.APIConnectionError as exc:
            last_error = LLMClientError(
                "Could not reach OpenAI API.",
                context={"attempt": attempt + 1, "error": str(exc)},
            )
        except openai.APIStatusError as exc:
            response_body = None
            if getattr(exc, "response", None) is not None:
                try:
                    response_body = exc.response.json()
                except Exception:
                    response_body = getattr(exc.response, "text", None)
            last_error = LLMClientError(
                "OpenAI API returned an unexpected status.",
                context={
                    "attempt": attempt + 1,
                    "status_code": exc.status_code,
                    "error": str(exc),
                    "response_body": response_body,
                },
            )
        except LLMClientError as exc:
            last_error = exc
        except Exception as exc:
            last_error = LLMClientError(
                "Unexpected error while generating the audit.",
                context={"attempt": attempt + 1, "error": str(exc)},
            )

        if attempt == 0:
            time.sleep(RETRY_DELAY_SECONDS)

    if last_error is None:
        raise LLMClientError("Unknown error while generating the audit.")
    raise last_error


def analyze_with_llm(
    system_prompt: str,
    user_prompt: str,
    *,
    provider_override: Optional[str] = None,
    model_override: Optional[str] = None,
    api_key_override: Optional[str] = None,
    visual_inputs: Optional[list[VisualSnapshot]] = None,
) -> AuditReport:
    runtime = resolve_runtime_overrides(provider_override, model_override, api_key_override)
    provider = runtime["provider"]
    if provider == "openai":
        return analyze_with_openai(
            system_prompt,
            user_prompt,
            model_name=runtime["model"],
            api_key_override=runtime["api_key"],
            visual_inputs=visual_inputs,
        )
    return analyze_with_claude(
        system_prompt,
        user_prompt,
        model_name=runtime["model"],
        api_key_override=runtime["api_key"],
        visual_inputs=visual_inputs,
    )
