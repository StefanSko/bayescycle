from __future__ import annotations

import socket
import subprocess
import sys
import time
import urllib.request
from collections.abc import Callable, Generator
from pathlib import Path
from typing import Any

import pytest
from playwright.sync_api import Browser, Page, Playwright, sync_playwright

PLAYGROUND_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def base_url() -> Generator[str]:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]

    url = f"http://127.0.0.1:{port}"
    process = subprocess.Popen(
        [sys.executable, "-m", "http.server", str(port), "--bind", "127.0.0.1"],
        cwd=PLAYGROUND_ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        for _ in range(100):
            if process.poll() is not None:
                pytest.fail("playground static server exited during startup")
            try:
                with urllib.request.urlopen(url, timeout=0.1):
                    break
            except OSError:
                time.sleep(0.05)
        else:
            pytest.fail("playground static server did not become ready")
        yield url
    finally:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)


@pytest.fixture(scope="session")
def playwright_instance() -> Generator[Playwright]:
    with sync_playwright() as instance:
        yield instance


@pytest.fixture(scope="session")
def browser(playwright_instance: Playwright) -> Generator[Browser]:
    instance = playwright_instance.chromium.launch(headless=True)
    yield instance
    instance.close()


@pytest.fixture
def page(browser: Browser) -> Generator[Page]:
    instance = browser.new_page()
    yield instance
    instance.close()


def _run_suite(
    page: Page, base_url: str, name: str, timeout_ms: int = 30_000
) -> list[dict[str, Any]]:
    page.goto(f"{base_url}/tests/harness.html?suite={name}")
    page.wait_for_function("window.__done === true", timeout=timeout_ms)
    return page.evaluate("window.__results")


@pytest.fixture
def run_suite() -> Callable[[Page, str, str], list[dict[str, Any]]]:
    return _run_suite


class OriginIsolation:
    def __init__(self, page: Page, origin: str) -> None:
        self.origin = origin
        self.urls: list[str] = []
        page.on("request", lambda request: self.urls.append(request.url))

    def assert_only_origin(self) -> None:
        non_origin = [url for url in self.urls if not url.startswith(self.origin)]
        assert not non_origin, f"non-origin requests observed: {non_origin}"


@pytest.fixture
def origin_isolation(page: Page, base_url: str) -> OriginIsolation:
    return OriginIsolation(page, base_url)
