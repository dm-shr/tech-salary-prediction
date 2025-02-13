import json
import logging
from datetime import datetime
from datetime import timedelta

import pytest

from src.scraping.headhunter import HeadhunterJobScraper


# Helper function to load fixtures
def load_fixture(filename):
    fixture_path = "tests/scraping/fixtures/" + filename
    with open(fixture_path) as f:
        return json.load(f)


@pytest.fixture
def mock_logger():
    return logging.getLogger("test")


@pytest.fixture
def today():
    return datetime.now().strftime("%Y-%m-%d")


@pytest.fixture
def yesterday():
    return (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")


@pytest.fixture
def scraper(mock_logger, yesterday, today):
    return HeadhunterJobScraper(
        logger=mock_logger,
        output_filename_base="test_data",
        start_date=yesterday,
        end_date=today,
        per_page=2,
        max_pages=1,
    )


@pytest.fixture
def mock_vacancy_response(today):
    response = load_fixture("headhunter_vacancy_response.json")
    # Update the date to today
    for item in response["items"]:
        item["published_at"] = f"{today}T12:00:00+0300"
    return response


@pytest.fixture
def mock_vacancy_details(today):
    details = load_fixture("headhunter_vacancy_details.json")
    details["published_at"] = f"{today}T12:00:00+0300"
    return details


def test_clean_html_tags(scraper):
    html = "<p>Test <b>text</b> with <i>tags</i></p>"
    expected = "Test text with tags"
    assert scraper.clean_html_tags(html) == expected


def test_process_experience(scraper):
    assert scraper.process_experience("noExperience") == (0, 0)
    # Convert map object to tuple for comparison
    result = scraper.process_experience("between3And6")
    assert tuple(result) == (3, 6)
    assert scraper.process_experience("moreThan6") == (6, -1)
    assert scraper.process_experience(None) == (None, None)


def test_get_vacancies(scraper, mocker, mock_vacancy_response, yesterday, today):
    mock_get = mocker.patch("requests.get")
    mock_get.return_value.status_code = 200
    mock_get.return_value.json.return_value = mock_vacancy_response

    result = scraper.get_vacancies(yesterday, today, 0)
    assert result == mock_vacancy_response
    assert len(result["items"]) == 1


def test_get_vacancy_details(scraper, mocker, mock_vacancy_details):
    mock_get = mocker.patch("requests.get")
    mock_get.return_value.status_code = 200
    mock_get.return_value.json.return_value = mock_vacancy_details

    result = scraper.get_vacancy_details("1234567")
    assert result == mock_vacancy_details
    assert result["name"] == "Python Developer"


def test_remove_duplicates(scraper, today):
    vacancies = [
        {"id": "1", "published_at": f"{today}T12:00:00+0300", "name": "Old Version"},
        {"id": "1", "published_at": f"{today}T13:00:00+0300", "name": "New Version"},
    ]
    result = scraper.remove_duplicates(vacancies)
    assert len(result) == 1
    assert result[0]["name"] == "New Version"


def test_get_russian_area_ids(scraper, mocker):
    mock_response = [{"id": "113", "areas": [{"id": "1", "areas": [{"id": "2"}]}]}]
    mock_get = mocker.patch("requests.get")
    mock_get.return_value.status_code = 200
    mock_get.return_value.json.return_value = mock_response

    result = scraper.get_russian_area_ids()
    assert isinstance(result, set)
    assert "1" in result
    assert "2" in result
