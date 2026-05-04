from __future__ import annotations

import base64
from dataclasses import dataclass
from typing import Optional


SCREENSHOT_VIEWPORT = {"width": 1440, "height": 1600}


@dataclass
class VisualSnapshot:
    label: str
    media_type: str
    base64_data: str

    @property
    def data_url(self) -> str:
        return f"data:{self.media_type};base64,{self.base64_data}"


def capture_url_screenshot(url: str, label: str) -> Optional[VisualSnapshot]:
    try:
        from playwright.sync_api import sync_playwright
    except Exception:
        return None

    try:
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(headless=True)
            page = browser.new_page(viewport=SCREENSHOT_VIEWPORT)
            page.goto(url, wait_until="domcontentloaded", timeout=20000)
            page.wait_for_timeout(1200)
            screenshot_bytes = page.screenshot(full_page=False, type="png")
            browser.close()
    except Exception:
        return None

    return VisualSnapshot(
        label=label,
        media_type="image/png",
        base64_data=base64.b64encode(screenshot_bytes).decode("utf-8"),
    )
