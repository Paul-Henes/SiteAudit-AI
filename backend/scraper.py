from __future__ import annotations

from typing import Optional

import httpx
from bs4 import BeautifulSoup

from backend.models import ScrapedPage
from backend.utils import clean_multiline_text, clean_whitespace, normalize_url, unique_preserve_order


SCRAPE_TIMEOUT_SECONDS = 15.0
USER_AGENT = (
    "Mozilla/5.0 (compatible; SiteAuditAI/1.0; +https://example.com/bot)"
)


def fetch_url(url: str) -> str:
    normalized_url = normalize_url(url)

    headers = {
        "User-Agent": USER_AGENT,
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    }

    with httpx.Client(
        follow_redirects=True,
        timeout=SCRAPE_TIMEOUT_SECONDS,
        headers=headers,
    ) as client:
        response = client.get(normalized_url)
        response.raise_for_status()
        return response.text


def extract_meta(soup: BeautifulSoup) -> tuple[str, str, list[str], list[str], str]:
    title = clean_whitespace(soup.title.string if soup.title and soup.title.string else "")

    meta_description = ""
    meta_tag = soup.find("meta", attrs={"name": "description"}) or soup.find(
        "meta", attrs={"property": "og:description"}
    )
    if meta_tag and meta_tag.get("content"):
        meta_description = clean_whitespace(meta_tag["content"])

    h1_tags = unique_preserve_order([tag.get_text(" ", strip=True) for tag in soup.find_all("h1")])[:5]

    cta_candidates = []
    for element in soup.select("a, button, input[type='submit'], input[type='button']"):
        text = ""
        if element.name == "input":
            text = element.get("value", "")
        else:
            text = element.get_text(" ", strip=True)
        if len(text) >= 2:
            cta_candidates.append(text)

    primary_ctas = unique_preserve_order(cta_candidates)[:8]

    viewport = "viewport meta tag present" if soup.find("meta", attrs={"name": "viewport"}) else "no viewport meta tag"
    responsive_images = "responsive images present" if soup.select("img[srcset], picture source") else "no responsive images detected"
    script_count = len(soup.find_all("script"))
    inferred_signals = f"{viewport}; {responsive_images}; {script_count} script tags detected"

    return title, meta_description, h1_tags, primary_ctas, inferred_signals


def extract_visible_text(soup: BeautifulSoup) -> str:
    working = BeautifulSoup(str(soup), "html.parser")

    for selector in ("script", "style", "noscript", "template", "svg", "iframe", "footer", "nav"):
        for tag in working.select(selector):
            tag.decompose()

    main_region = working.find("main") or working.body or working
    text = main_region.get_text("\n", strip=True)
    return clean_multiline_text(text)


def scrape_url(url: str) -> ScrapedPage:
    normalized_url = normalize_url(url)
    raw_html = fetch_url(normalized_url)
    soup = BeautifulSoup(raw_html, "html.parser")

    title, meta_description, h1_tags, primary_ctas, inferred_signals = extract_meta(soup)
    body_text = extract_visible_text(soup)

    return ScrapedPage(
        source_url=normalized_url,
        title=title,
        meta_description=meta_description,
        h1_tags=h1_tags,
        primary_ctas=primary_ctas,
        body_text=body_text,
        inferred_mobile_performance_signals=inferred_signals,
    )


def build_raw_text_page(raw_text: str, business_context: Optional[str] = None) -> ScrapedPage:
    title = "Direct text input"
    return ScrapedPage(
        source_url=None,
        title=title,
        meta_description="",
        h1_tags=[],
        primary_ctas=[],
        body_text=clean_multiline_text(raw_text),
        inferred_mobile_performance_signals="Not available for pasted text input",
    )
