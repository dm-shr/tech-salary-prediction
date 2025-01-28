import logging
import re
from unittest.mock import patch

import pandas as pd
import pytest

from src.preprocessing.main import JobDataPreProcessor


# Fixtures
@pytest.fixture
def mock_config():
    return {
        "is_test": True,
        "preprocessing": {
            "input_filename_base": {"getmatch": "test_getmatch", "headhunter": "test_headhunter"},
            "merged_path": "test_merged",
            "historical_data_path": "test_historical.csv",
            "output_path": "test_output.csv",
            "salary_outliers": {
                "bottom_percentile": 0.001,
                "top_percentile": 1.0,
            },
            "drift_thresholds": {
                "ks_pvalue_threshold": 0.05,
                "js_divergence_threshold": 0.1,
                "psi_threshold": 0.2,
            },
        },
    }


@pytest.fixture
def mock_getmatch_data():
    return pd.DataFrame(
        {
            "description_text": [
                "Нужен Python-разработчик, онлайн",
                "Нужен Senior Java разработчик в офис",
            ],
            "company_name": ["CompanyA", "CompanyB"],
            "title": ["Python Dev", "Java Dev"],
            "location": ["Moscow", "Saint Petersburg"],
            "skills": ["['Python', 'SQL']", "['Java', 'Spring']"],
            "level": ["Middle", "Senior"],
            "salary_text": ["от 150000 ₽", "200000-300000 EUR"],
            "salary_from": [150000, 200000],
            "salary_to": [None, 300000],
            "url": ["url1", "url2"],
            "published_date": ["2024-01-01", "2024-01-02"],
        }
    )


@pytest.fixture
def mock_headhunter_data():
    return pd.DataFrame(
        {
            "description": [
                "Нужен Python-разработчик, офис в г. Москва",
                "Нужен Java разработчик на удаленку",
            ],
            "company": ["CompanyC", "CompanyD"],
            "title": ["Python Dev", "Java Dev"],
            "area": ["Moscow", None],
            "skills": ["Python, SQL", "Java, Spring"],
            "salary_from": [160000, 180000],
            "salary_to": [200000, 250000],
            "currency": ["RUR", "RUR"],
            "url": ["url3", "url4"],
            "published_date": ["2024-01-03", "2024-01-04"],
        }
    )


@pytest.fixture
def logger():
    return logging.getLogger("test_logger")


def test_get_currency():
    assert JobDataPreProcessor.get_currency("150000 ₽") == "RUR"
    assert JobDataPreProcessor.get_currency("5000 EUR") == "EUR"
    assert JobDataPreProcessor.get_currency("$4000") == "USD"
    assert JobDataPreProcessor.get_currency("3000 £") == "GBP"
    assert JobDataPreProcessor.get_currency("No currency") is None


def test_list_to_string():
    assert JobDataPreProcessor.list_to_string("['Python', 'SQL']") == "Python, SQL"
    assert JobDataPreProcessor.list_to_string("Already a string") == "Already a string"


def test_which_language():
    assert JobDataPreProcessor.which_language("Требуется Python разработчик") == "ru"
    assert JobDataPreProcessor.which_language("Looking for Python developer") == "en"
    assert JobDataPreProcessor.which_language("12345") == "unknown"


def test_replace_salary_patterns():
    test_cases = [
        (
            "Зарплата от 150000 рублей до 200000 рублей",
            "Зарплата от [NUMBER] рублей до [NUMBER] рублей",
        ),
        ("ЗП 150 000 ₽ - 200 000 ₽", "ЗП [NUMBER] рублей - [NUMBER] рублей".replace("  ", " ")),
        ("Оклад: от 120000₽", "Оклад: от [NUMBER] рублей"),
        ("Зарплата 100500 рублей + бонусы 50000", "Зарплата [NUMBER] рублей + бонусы 50000"),
        ("Плата от 80 000 до 120 000₽", "Плата от [NUMBER]до [NUMBER] рублей"),
        ("Salary 1234567 RUB", "Salary [NUMBER] RUB"),  # Testing large numbers
        ("Text without salary", "Text without salary"),  # Testing text without salary
        (
            "ЗП до 150000₽ после испытательного срока",
            "ЗП до [NUMBER] рублей после испытательного срока",
        ),
    ]

    for original, expected in test_cases:
        result = JobDataPreProcessor.replace_salary_patterns(original)
        result = re.sub(r"\s+", " ", result)  # normalize spaces
        expected = re.sub(r"\s+", " ", expected)  # normalize spaces
        assert result == expected, f"\nExpected: {expected}\nGot: {result}\nFor input: {original}"


@pytest.mark.parametrize("exists_param", [True, False])
@patch("os.path.exists")
@patch("pandas.read_csv")
def test_process(
    mock_read_csv,
    mock_exists,
    exists_param,
    mock_config,
    mock_getmatch_data,
    mock_headhunter_data,
    logger,
):
    """
    Parameters order matters:
    1. Parametrize arguments come first (exists_param)
    2. Then patch decorators from bottom to top (mock_read_csv, mock_exists)
    3. Then fixture arguments
    """
    mock_exists.return_value = exists_param
    # Provide historical data for the True case
    if exists_param:
        historical_data = pd.DataFrame(
            {
                "description": ["Old job post"],
                "company": ["OldCo"],
                "title": ["Old title"],
                "published_date": ["2023-12-01"],
                "salary_from": [100000],
                "salary_to": [200000],
                "currency": ["RUR"],
                "log_salary_from": [11.51],
            }
        )
        mock_read_csv.side_effect = [mock_getmatch_data, mock_headhunter_data, historical_data]
    else:
        mock_read_csv.side_effect = [mock_getmatch_data, mock_headhunter_data]

    with patch("src.preprocessing.main.load_config", return_value=mock_config), patch(
        "src.preprocessing.main.current_week_info", return_value={"week_number": 1, "year": 2024}
    ), patch("pandas.DataFrame.to_csv") as mock_to_csv:

        processor = JobDataPreProcessor(logger)
        processor.process()

        # Verify that to_csv was called
        assert mock_to_csv.call_count >= 2


def test_init_processor(mock_config, logger):
    with patch("src.preprocessing.main.load_config", return_value=mock_config), patch(
        "src.preprocessing.main.current_week_info", return_value={"week_number": 1, "year": 2024}
    ):

        processor = JobDataPreProcessor(logger)
        assert "2024" in processor.getmatch_path
        assert "2024" in processor.headhunter_path


@pytest.mark.parametrize(
    "input_data,expected_count",
    [
        (pd.DataFrame({"salary_from": [100, 200, 300, 400, 500]}), 3),  # Normal case
        (pd.DataFrame({"salary_from": [100, 100, 100, 100, 100]}), 5),  # All same values
        (pd.DataFrame({"salary_from": [1, 1000000, 2, 3, 4]}), 3),  # With outliers
    ],
)
def test_salary_outlier_removal(input_data, expected_count, logger):
    bottom_percentile = 0.1
    top_percentile = 0.9

    result = JobDataPreProcessor.remove_salary_outliers(
        input_data,
        bottom_percentile,
        top_percentile,
    )
    assert len(result) == expected_count


def test_remove_description_duplicates():
    # Test data with duplicates
    test_data = pd.DataFrame(
        {
            "description": [
                "Python developer needed",  # original
                "Python developer needed",  # exact duplicate
                "Python developer needed ",  # duplicate with extra space, should be removed
                "Java developer needed",  # unique
                "Ruby developer needed",  # unique
            ],
            "salary": [
                100,
                200,
                300,
                400,
                500,
            ],  # different salaries to ensure we're matching on description only
        }
    )

    result = JobDataPreProcessor.remove_description_duplicates(test_data)

    # Check that duplicates were removed
    assert len(result) == 2  # Should only have 2 rows (Java, Ruby, neither Python due to conflict)
    assert (
        "Python developer needed" not in result["description"].values
    )  # All Python duplicates should be removed
    assert "Java developer needed" in result["description"].values  # Unique entry should remain
    assert "Ruby developer needed" in result["description"].values  # Unique entry should remain
    assert "description_hash" not in result.columns  # Hash column should be removed
