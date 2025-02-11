import logging
import re
import time
from datetime import datetime
from typing import Dict
from typing import List
from typing import Tuple

import pandas as pd
import requests
from bs4 import BeautifulSoup
from src.utils.utils import current_week_info


class GetmatchJobScraper:
    def __init__(
        self,
        logger: logging.Logger,
        output_filename_base: str,
        num_pages: int = 5,
        output_format: str = "csv",
    ):
        self.filename_base = output_filename_base
        self.base_url = "https://getmatch.ru/vacancies"
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        self.num_pages = num_pages
        self.output_format = output_format

        self.logger = logger

    def parse_salary(self, salary_text: str) -> tuple:
        """
        Parses the salary string and returns the minimum and maximum values

        Examples of input data:
        - "250 000 — 300 000 ₽/month after taxes"
        - "from 250 000 ₽/month after taxes"
        - "up to 300 000 ₽/month after taxes"
        """
        if not salary_text:
            return None, None

        # Remove all extra spaces and convert to lowercase
        salary_text = salary_text.lower().strip()

        # Remove "₽/month after taxes" and similar endings
        salary_text = re.sub(r"₽/мес.*$", "", salary_text)

        # remove spaces between digits
        salary_text = re.sub(r"\s(?=\d)", "", salary_text)

        # try to find two numbers (range)
        range_match = re.findall(r"\d+", salary_text)

        if "от" in salary_text and len(range_match) == 1:
            return int(range_match[0]), None
        elif "до" in salary_text and len(range_match) == 1:
            return None, int(range_match[0])
        elif len(range_match) >= 2:
            return int(range_match[0]), int(range_match[1])
        elif len(range_match) == 1:
            return int(range_match[0]), int(range_match[0])

        return None, None

    def get_job_description(self, job_url: str, published_at) -> Dict:
        """Extracts full job description from the job URL"""
        try:
            response = requests.get(job_url, headers=self.headers)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, "html.parser")

            job_data = {
                "published_date": published_at,
                "url": job_url,
                "title": None,
                "company_name": None,
                "salary_text": None,
                "salary_from": None,
                "salary_to": None,
                "location": None,
                "work_format": None,
                "specialization": None,
                "level": None,
                "company_logo_url": None,
                "description_text": [],
                "skills": [],
                "posted_date": None,
            }

            # Title
            title_elem = soup.find("h1")
            if title_elem:
                job_data["title"] = title_elem.text.strip()

            # Company
            company_elem = soup.find("h2")
            if company_elem and company_elem.find("a"):
                job_data["company_name"] = company_elem.find("a").text.strip()

            # Salary
            salary_elem = soup.find("h3")
            if salary_elem:
                salary_text = salary_elem.text.strip()
                job_data["salary_text"] = salary_text
                job_data["salary_from"], job_data["salary_to"] = self.parse_salary(salary_text)

            # Location
            location_container = soup.find("div", class_="b-vacancy-locations")
            if location_container:
                locations = location_container.find_all("span", class_="g-label")
                for loc in locations:
                    if "📍" in loc.text:
                        job_data["location"] = loc.text.replace("📍", "").strip()
                    else:
                        job_data["work_format"] = loc.text.strip()

            # Grade/spec
            specs_container = soup.find("div", class_="b-specs")
            if specs_container:
                rows = specs_container.find_all("div", class_="row")
                for row in rows:
                    term = row.find("div", class_="b-term")
                    value = row.find("div", class_="b-value")
                    if term and value:
                        term_text = term.text.strip().lower()
                        if "специализация" in term_text:
                            job_data["specialization"] = value.text.strip()
                        elif "уровень" in term_text:
                            job_data["level"] = value.text.strip()

            # company logo
            logo_elem = soup.find("img", {"alt": lambda x: x and "logo" in x.lower()})
            if logo_elem:
                job_data["company_logo_url"] = logo_elem.get("src")

            # job description
            description_sections = [
                ("b-vacancy-short-description", "Краткое описание"),
                ("b-vacancy-description", "Полное описание"),
            ]

            for class_name, section_name in description_sections:
                section = soup.find("section", class_=class_name)
                if section:
                    for elem in section.stripped_strings:
                        if elem.strip():
                            job_data["description_text"].append(elem.strip())

            # skills
            stack_container = soup.find("div", class_="b-vacancy-stack-container")
            if stack_container:
                skills = [
                    skill.text.strip()
                    for skill in stack_container.find_all("span", class_="g-label")
                ]
                job_data["skills"] = skills

            # combine description
            job_data["description_text"] = "\n".join(job_data["description_text"])

            return job_data

        except requests.RequestException as e:
            self.logger.error(f"Error when fetching {job_url}: {e}")
            return {"url": job_url, "error": str(e)}
        except Exception as e:
            self.logger.error(f"Unexpected error when fetching {job_url}: {e}")
            return {"url": job_url, "error": str(e)}

    def get_job_urls(self, page: int = 1) -> Tuple[List[str], List[str]]:
        """Getting list of jobs URLs from the page"""
        params = {"p": page}
        try:
            response = requests.get(self.base_url, params=params, headers=self.headers)
            response.raise_for_status()

            soup = BeautifulSoup(response.text, "html.parser")
            job_cards = soup.find_all("div", class_="b-vacancy-card")

            job_urls = []
            published_at_dates = []
            for card in job_cards:
                # get published_at date from element div class="b-vacancy-card-header__publish-date"
                published_at_elem = card.find("div", class_="b-vacancy-card-header__publish-date")
                published_at = datetime.now().strftime("%Y-%m-%d")
                if published_at_elem:
                    published_at = published_at_elem.text.strip()
                    if "сегодня" in published_at:
                        published_at = datetime.now().strftime("%Y-%m-%d")
                    else:  # format of '16 января 2025 г.', convert it
                        day, month, year, _ = published_at.split()
                        month = {
                            "января": "01",
                            "февраля": "02",
                            "марта": "03",
                            "апреля": "04",
                            "мая": "05",
                            "июня": "06",
                            "июля": "07",
                            "августа": "08",
                            "сентября": "09",
                            "октября": "10",
                            "ноября": "11",
                            "декабря": "12",
                        }[month]
                        published_at = f"{day}.{month}.{year}"
                        published_at = datetime.strptime(published_at, "%d.%m.%Y").strftime(
                            "%Y-%m-%d"
                        )

                published_at_dates.append(published_at)
                title_elem = card.find("h3")
                if title_elem:
                    link_elem = title_elem.find("a")
                    if link_elem and link_elem.get("href"):
                        job_urls.append("https://getmatch.ru" + link_elem.get("href"))

            return job_urls, published_at_dates
        except Exception as e:
            self.logger.error(f"Error when fetching job list from page {page}: {e}")
            return []

    def scrape(self):
        """Collecting job descriptions from multiple pages"""
        all_jobs = []

        for page in range(1, self.num_pages + 1):
            self.logger.info(f"Parsing page {page}...")
            job_urls, published_at_dates = self.get_job_urls(page)

            # for url in job_urls:
            for url, published_at in zip(job_urls, published_at_dates):
                self.logger.info(f"Getting job description: {url}")
                job_data = self.get_job_description(url, published_at)
                if job_data and "error" not in job_data:
                    all_jobs.append(job_data)
                time.sleep(1)

            time.sleep(2)

        if all_jobs:
            week_info = current_week_info()
            output_filename = (
                f"{self.filename_base}_week_{week_info['week_number']}_year_{week_info['year']}"
            )

            if self.output_format == "csv":
                df = pd.DataFrame(all_jobs)
                filename = f"{output_filename}.csv"
                df.to_csv(filename, index=False, encoding="utf-8")
                self.logger.info(f"Saved {len(all_jobs)} job descriptions to {filename}")
            elif self.output_format == "json":
                filename = f"{output_filename}.json"
                pd.DataFrame(all_jobs).to_json(
                    filename, orient="records", force_ascii=False, indent=2
                )
                self.logger.info(f"Saved {len(all_jobs)} job descriptions to {filename}")
            else:
                self.logger.error("Unknown output format. Use 'csv' or 'json'.")

        else:
            self.logger.error("Could not collect job data")
