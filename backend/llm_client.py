from __future__ import annotations

import json
import time
from typing import Any, Optional

import anthropic
from anthropic import Anthropic
import openai
from openai import OpenAI
from pydantic import ValidationError

from backend.config import get_env
from backend.models import AuditReport


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
    return AuditReport.model_validate(payload)


def _openai_response_format() -> dict[str, Any]:
    schema = AuditReport.model_json_schema()
    return {
        "type": "json_schema",
        "name": "site_audit_report",
        "schema": schema,
        "strict": True,
    }


def analyze_with_claude(
    system_prompt: str,
    user_prompt: str,
    *,
    model_name: Optional[str] = None,
    api_key_override: Optional[str] = None,
) -> AuditReport:
    client = _get_anthropic_client(api_key_override)
    last_error: Optional[LLMClientError] = None

    for attempt in range(2):
        try:
            active_model = (model_name or get_env("ANTHROPIC_MODEL", ANTHROPIC_MODEL_NAME) or ANTHROPIC_MODEL_NAME).strip()
            response = client.messages.create(
                model=active_model,
                max_tokens=MAX_TOKENS,
                system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}],
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
) -> AuditReport:
    client = _get_openai_client(api_key_override)
    last_error: Optional[LLMClientError] = None

    for attempt in range(2):
        try:
            active_model = (model_name or get_env("OPENAI_MODEL", OPENAI_MODEL_NAME) or OPENAI_MODEL_NAME).strip()
            response = client.responses.create(
                model=active_model,
                input=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                max_output_tokens=MAX_TOKENS,
                text={"format": _openai_response_format()},
            )

            response_text = getattr(response, "output_text", "").strip()
            if not response_text:
                raise LLMClientError(
                    "OpenAI returned an empty response.",
                    context={"attempt": attempt + 1},
                )

            report = _parse_report(response_text)
            return report
        except openai.RateLimitError:
            last_error = LLMClientError(
                "OpenAI rate limit exceeded while generating the audit.",
                context={"attempt": attempt + 1, "status_code": 429},
            )
        except (json.JSONDecodeError, ValidationError) as exc:
            last_error = LLMClientError(
                "OpenAI returned malformed audit JSON.",
                context={"attempt": attempt + 1, "error": str(exc)},
            )
        except openai.APIConnectionError as exc:
            last_error = LLMClientError(
                "Could not reach OpenAI API.",
                context={"attempt": attempt + 1, "error": str(exc)},
            )
        except openai.APIStatusError as exc:
            last_error = LLMClientError(
                "OpenAI API returned an unexpected status.",
                context={
                    "attempt": attempt + 1,
                    "status_code": exc.status_code,
                    "error": str(exc),
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
) -> AuditReport:
    runtime = resolve_runtime_overrides(provider_override, model_override, api_key_override)
    provider = runtime["provider"]
    if provider == "openai":
        return analyze_with_openai(
            system_prompt,
            user_prompt,
            model_name=runtime["model"],
            api_key_override=runtime["api_key"],
        )
    return analyze_with_claude(
        system_prompt,
        user_prompt,
        model_name=runtime["model"],
        api_key_override=runtime["api_key"],
    )
