# SiteAudit AI
> Paste a URL. Get a consulting report in seconds.

SiteAudit AI is a website audit dashboard that turns URLs or pasted copy
into structured, consulting-grade reports with scored dimensions,
critical issues, quick wins, heatmaps, and PDF-ready output.

Built for founders, consultants, and growth teams who want sharper
website analysis without the usual vague AI feedback loop.

<img width="829" height="743" alt="Screenshot 2026-05-04 at 12 26 21" src="https://github.com/user-attachments/assets/1460bfd2-a5c0-4b54-a65d-14e2a2315ccd" />


## What it does
- Scrapes and parses website content
- Runs it through a structured LLM analysis pipeline
- Returns scored dimensions, critical issues, quick wins,
  and an executive summary

<img width="405" height="637" alt="Screenshot 2026-05-04 at 12 22 27" src="https://github.com/user-attachments/assets/2537e94d-50ac-48de-a852-2b7081c0f7e4" />

## Why it's different
Most 'AI website checkers' return vague suggestions.
SiteAudit AI outputs a structured JSON report modeled on
consulting deliverables: severity-rated issues, effort-tagged
quick wins, and scored dimensions with rationale.

## Stack
FastAPI · Anthropic/OpenAI API · BeautifulSoup · Vanilla JS

## Run locally
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
uvicorn backend.main:app --reload
```

Open [http://localhost:8000](http://localhost:8000).

## Environment
Set your provider and matching API key before calling `/analyze`.

Anthropic:
```bash
export LLM_PROVIDER=anthropic
export ANTHROPIC_API_KEY=your_anthropic_api_key
```

OpenAI:
```bash
export LLM_PROVIDER=openai
export OPENAI_API_KEY=your_openai_api_key
```

You can also create a local `.env` file in the project root. The app now loads `.env` automatically on startup, so keys stay server-side and never need to be entered in the browser UI.

For internal or local testing, the audit form also supports an optional temporary session API key. That key is sent only with the current `/analyze` request, is not persisted in saved reports, and is not stored by the frontend app. It should still be treated as a sensitive value because it passes through the browser for that request.

## API
- `POST /analyze` accepts either `url` or `raw_text`, plus optional `business_context`
- `GET /api/report/{id}` returns the stored JSON report
- `GET /report/{id}` renders the report UI

## Notes
- Anthropic support keeps the exact model specified in the brief by default: `claude-sonnet-4-20250514`
- OpenAI support uses structured JSON output with `gpt-4.1` by default and can be changed via `OPENAI_MODEL`
- `GET /health` reports the active provider, active model, and whether matching server-side credentials are configured
- Scraped body text is truncated to 3000 characters before prompt assembly
- Reports are now persisted in SQLite so dashboard history survives app restarts
- The default database path is `data/siteaudit.db` and can be overridden with `SITEAUDIT_DB_PATH`
