"""
Web scraper for SHL product catalog (skeleton).

Planned responsibilities:
- Crawl SHL catalog pages (or a known list of URLs)
- Extract assessment metadata (name, url, test_type, description, duration, etc.)
- Normalize and store results into `catalog.json`

WARNING:
- Ensure scraping complies with SHL's Terms of Service and robots.txt.
- Consider rate limiting, caching, and user-agent identification.

TODO:
- Identify authoritative SHL catalog entry points and parsing rules
- Implement robust HTML parsing with fallbacks
- Add retries, exponential backoff, and polite delays
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests
from bs4 import BeautifulSoup


@dataclass
class ScraperConfig:
    """
    Scraper configuration.

    TODO:
    - Add base URL(s), pagination rules, and request headers
    - Add timeout and retry strategy
    """

    output_path: Path = Path("catalog.json")
    timeout_s: int = 30


class SHLCatalogScraper:
    """
    Scrapes SHL assessments into a local JSON catalog file.

    TODO:
    - Implement `scrape()` to return a list of dict items
    - Implement `save_catalog()` to persist results
    """

    def __init__(self, config: Optional[ScraperConfig] = None) -> None:
        self.config = config or ScraperConfig()

        # TODO: configure session headers, retries, and caching
        self._session = requests.Session()

    def scrape(self) -> List[Dict[str, Any]]:
        """
        Scrape the SHL catalog.

        Returns:
            A list of normalized catalog items.

        TODO:
        - Fetch listing pages
        - Extract detail page links
        - Parse each assessment page into a structured record
        """
        raise NotImplementedError

    def parse_assessment_page(self, html: str, url: str) -> Dict[str, Any]:
        """
        Parse an assessment detail page.

        TODO:
        - Extract fields needed by the recommender (at minimum: name, url, test_type)
        - Extract optional fields for richer ranking (description, job family, etc.)
        """
        soup = BeautifulSoup(html, "html.parser")
        _ = soup  # TODO: remove once parsing is implemented
        raise NotImplementedError

    def save_catalog(self, items: List[Dict[str, Any]]) -> None:
        """
        Save scraped items to `catalog.json`.
        """
        self.config.output_path.write_text(json.dumps(items, indent=2), encoding="utf-8")

