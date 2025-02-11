import logging

import pytest
import requests
from src.scraping.getmatch import GetmatchJobScraper


@pytest.fixture
def scraper():
    logger = logging.getLogger("test_logger")
    return GetmatchJobScraper(logger, "test_output", num_pages=1)


@pytest.fixture
def sample_job_card_html():
    with open("tests/scraping/fixtures/getmatch_job_card.html", encoding="utf-8") as f:
        return f.read()


@pytest.fixture
def sample_job_page_html():
    with open("tests/scraping/fixtures/getmatch_job_page.html", encoding="utf-8") as f:
        return f.read()


def test_parse_salary():
    scraper = GetmatchJobScraper(logging.getLogger("test"), "test")

    # Test various salary formats
    assert scraper.parse_salary("250 000 — 300 000 ₽/мес на руки") == (250000, 300000)
    assert scraper.parse_salary("от 250 000 ₽/мес на руки") == (250000, None)
    assert scraper.parse_salary("до 300 000 ₽/мес на руки") == (None, 300000)
    assert scraper.parse_salary("") == (None, None)
    assert scraper.parse_salary(None) == (None, None)


def test_get_job_urls(scraper, sample_job_card_html):
    # Mock the requests.get response
    class MockResponse:
        def __init__(self):
            self.text = sample_job_card_html

        def raise_for_status(self):
            pass

    # Store original requests.get
    original_get = requests.get

    try:
        requests.get = lambda *args, **kwargs: MockResponse()
        urls, dates = scraper.get_job_urls(1)
        assert len(urls) > 0
        assert all(url.startswith("https://getmatch.ru/vacancies/") for url in urls)
        assert all(date.count("-") == 2 for date in dates)
    finally:
        # Restore original requests.get
        requests.get = original_get


def test_job_description_parsing(scraper, sample_job_page_html):
    class MockResponse:
        def __init__(self):
            self.text = sample_job_page_html

        def raise_for_status(self):
            pass

    scraper.get_job_description.__globals__["requests"].get = lambda *args, **kwargs: MockResponse()

    job_data = scraper.get_job_description("https://getmatch.ru/vacancies/test", "2024-01-01")

    # Test extracted fields
    assert isinstance(job_data, dict)
    assert "title" in job_data
    assert "company_name" in job_data
    assert "salary_text" in job_data
    assert "location" in job_data
    assert "work_format" in job_data
    assert isinstance(job_data["description_text"], str)
    assert isinstance(job_data["skills"], list)


def test_error_handling(scraper):
    # Mock requests.get to raise an exception
    def mock_error(*args, **kwargs):
        raise requests.RequestException("Test error")

    original_get = requests.get
    try:
        requests.get = mock_error
        job_data = scraper.get_job_description("https://invalid.url", "2024-01-01")
        assert "error" in job_data
        assert job_data["url"] == "https://invalid.url"
        assert "Test error" in job_data["error"]
    finally:
        requests.get = original_get


def test_salary_parsing_edge_cases(scraper):
    # Test edge cases for salary parsing
    test_cases = [
        ("250000₽/мес на руки", (250000, 250000)),
        ("от 100 500 до 200 500 ₽/мес на руки", (100500, 200500)),
        ("Invalid salary text", (None, None)),
        ("1000-2000 ₽/мес на руки", (1000, 2000)),
    ]

    for salary_text, expected in test_cases:
        assert scraper.parse_salary(salary_text) == expected


def test_output_format(scraper, sample_job_card_html, sample_job_page_html):
    # Mock responses
    class MockResponse:
        def __init__(self, html):
            self.text = html

        def raise_for_status(self):
            pass

    def mock_get(url, *args, **kwargs):
        if "vacancies?" in url:
            return MockResponse(sample_job_card_html)
        return MockResponse(sample_job_page_html)

    scraper.get_job_urls.__globals__["requests"].get = mock_get
    scraper.get_job_description.__globals__["requests"].get = mock_get

    # Test CSV output
    scraper.output_format = "csv"
    scraper.scrape()

    # Test JSON output
    scraper.output_format = "json"
    scraper.scrape()
