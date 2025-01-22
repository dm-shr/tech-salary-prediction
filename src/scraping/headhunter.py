import csv
import os
import random
import re
import time
from datetime import datetime
from datetime import timedelta

import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

from src.utils.utils import current_week_info  # dict with keys 'week_number' and 'year'


class HeadhunterJobScraper:
    def __init__(
        self,
        output_filename_base="data",
        start_date=None,
        end_date=None,
        per_page=5,
        max_pages=2,
    ):
        week_info = current_week_info()
        self.filename = (
            f"{output_filename_base}_week_{week_info['week_number']}_year_{week_info['year']}"
        )
        self.DATE_FORMAT = "%Y-%m-%d"
        if isinstance(start_date, str):
            self.START_DATE = datetime.strptime(start_date, self.DATE_FORMAT).strftime(
                self.DATE_FORMAT
            )
        else:
            self.START_DATE = (datetime.now() - timedelta(days=1)).strftime(self.DATE_FORMAT)

        if isinstance(end_date, str):
            self.END_DATE = datetime.strptime(end_date, self.DATE_FORMAT).strftime(self.DATE_FORMAT)
        else:
            self.END_DATE = datetime.now().strftime(self.DATE_FORMAT)

        self.PER_PAGE = per_page
        self.MAX_PAGES = max_pages
        self.PROFESSIONAL_ROLES = [
            "156",
            "160",
            "10",
            "12",
            "150",
            "25",
            "165",
            "34",
            "36",
            "73",
            "155",
            "96",
            "164",
            "104",
            "157",
            "107",
            "112",
            "113",
            "148",
            "114",
            "116",
            "121",
            "124",
            "125",
            "126",
        ]
        self.HEADERS = {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:92.0) Gecko/20100101 Firefox/92.0",
            "Accept-Language": "ru-RU,ru;q=0.9,en-US;q=0.8,en;q=0.7",
            "Accept-Encoding": "gzip, deflate, br",
            "Connection": "keep-alive",
        }

    def clean_html_tags(self, text):
        soup = BeautifulSoup(text, "html.parser")
        return " ".join(soup.get_text(separator=" ").split())

    def get_russian_area_ids(self):
        url = "https://api.hh.ru/areas"
        response = requests.get(url, headers=self.HEADERS)
        if response.status_code == 200:
            areas = response.json()
            russian_area_ids = set()

            def extract_ids(area):
                if area["id"] == "113":
                    for sub_area in area["areas"]:
                        russian_area_ids.add(sub_area["id"])
                        for sub_sub_area in sub_area["areas"]:
                            russian_area_ids.add(sub_sub_area["id"])
                else:
                    for sub_area in area["areas"]:
                        extract_ids(sub_area)

            for area in areas:
                extract_ids(area)
            return russian_area_ids
        return set()

    def get_vacancies(self, date_from, date_to, page):
        url = f"https://api.hh.ru/vacancies?only_with_salary=true&date_from={date_from}&date_to={date_to}&page={page}&per_page={self.PER_PAGE}"
        for role in self.PROFESSIONAL_ROLES:
            url += f"&professional_role={role}"

        for _ in range(5):
            try:
                response = requests.get(url, headers=self.HEADERS, timeout=60)
                if response.status_code == 200:
                    return response.json()
                time.sleep(1)
            except requests.RequestException as e:
                print(f"Request failed: {e}. Retrying...")
                time.sleep(1)
        return {"items": []}

    def get_vacancy_details(self, vacancy_id):
        url = f"https://api.hh.ru/vacancies/{vacancy_id}"
        for _ in range(5):
            try:
                response = requests.get(url, headers=self.HEADERS, timeout=60)
                if response.status_code == 200:
                    return response.json()
                time.sleep(1)
            except requests.RequestException as e:
                print(f"Request failed: {e}. Retrying...")
                time.sleep(1)
        return {"id": vacancy_id, "error": "Failed after retries"}

    def process_experience(self, experience_id):
        if not experience_id:
            return None, None
        if experience_id == "noExperience":
            return 0, 0
        match = re.match(r"between(\d+)And(\d+)", experience_id)
        if match:
            return map(int, match.groups())
        match = re.match(r"moreThan(\d+)", experience_id)
        if match:
            return int(match.group(1)), -1
        return None, None

    def collect_vacancies(self):
        current_date = datetime.strptime(self.START_DATE, self.DATE_FORMAT)
        end_date = datetime.strptime(self.END_DATE, self.DATE_FORMAT)
        all_vacancies = []

        with tqdm(desc="Collecting vacancies") as pbar:
            while current_date <= end_date:
                date_from = current_date.strftime(self.DATE_FORMAT)
                date_to = (current_date + timedelta(days=1)).strftime(self.DATE_FORMAT)
                page = 0

                while True:
                    vacancies = self.get_vacancies(date_from, date_to, page)
                    if not vacancies["items"]:
                        break
                    all_vacancies.extend(vacancies["items"])
                    if len(vacancies["items"]) < self.PER_PAGE or page >= self.MAX_PAGES - 1:
                        break
                    page += 1
                    pbar.update(len(vacancies["items"]))

                current_date += timedelta(days=1)

        return all_vacancies

    def remove_duplicates(self, vacancies):
        vacancy_map = {}
        for vacancy in tqdm(vacancies, desc="Removing duplicates"):
            vid = vacancy["id"]
            if vid not in vacancy_map:
                vacancy_map[vid] = vacancy
            else:
                existing_date = datetime.strptime(
                    vacancy_map[vid]["published_at"], "%Y-%m-%dT%H:%M:%S%z"
                )
                new_date = datetime.strptime(vacancy["published_at"], "%Y-%m-%dT%H:%M:%S%z")
                if new_date > existing_date:
                    vacancy_map[vid] = vacancy
        return list(vacancy_map.values())

    def process_vacancies(self, vacancies):
        processed_vacancies = []
        for vacancy in tqdm(vacancies, desc="Processing vacancies"):
            details = self.get_vacancy_details(vacancy["id"])
            if "error" not in details:
                processed_vacancies.append(details)
            time.sleep(0.5 + random.random() * 0.5)
        return processed_vacancies

    def save_to_csv(self, vacancies, filepath):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        with open(filepath, mode="w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(
                [
                    "published_date",
                    "url",
                    "title",
                    "area",
                    "company",
                    "skills",
                    "description",
                    "salary_from",
                    "salary_to",
                    "currency",
                    "experience_from",
                    "experience_to",
                ]
            )

            for vacancy in vacancies:
                if vacancy.get("salary") is not None:
                    experience = vacancy.get("experience", {}).get("id", "")
                    exp_from, exp_to = self.process_experience(experience)
                    published_at = vacancy.get(
                        "published_at", None
                    )  # format = 2024-10-31T14:45:21+0300, convert to a datetime object
                    if published_at is None:
                        published_at = datetime.now().strftime("%d.%m.%Y")
                    else:
                        published_at = datetime.strptime(
                            published_at, "%Y-%m-%dT%H:%M:%S%z"
                        ).strftime("%d.%m.%Y")

                    writer.writerow(
                        [
                            published_at,
                            vacancy.get("alternate_url", ""),
                            vacancy.get("name", ""),
                            vacancy.get("area", {}).get("name", ""),
                            vacancy.get("employer", {}).get("name", ""),
                            ", ".join(skill["name"] for skill in vacancy.get("key_skills", [])),
                            self.clean_html_tags(vacancy.get("description", "")),
                            vacancy["salary"].get("from", ""),
                            vacancy["salary"].get("to", ""),
                            vacancy["salary"].get("currency", ""),
                            exp_from,
                            exp_to,
                        ]
                    )

    def scrape(self):
        print("Starting data collection...")
        russian_area_ids = self.get_russian_area_ids()
        vacancies = self.collect_vacancies()

        print("Processing and filtering vacancies...")
        unique_vacancies = self.remove_duplicates(vacancies)
        russian_vacancies = [v for v in unique_vacancies if v["area"]["id"] in russian_area_ids]

        processed_vacancies = self.process_vacancies(russian_vacancies)

        output_file = f"{self.filename}.csv"
        self.save_to_csv(processed_vacancies, output_file)
        print(f"Data collection completed. Results saved to {output_file}")


# for testing purposes
if __name__ == "__main__":
    scraper = HeadhunterJobScraper(output_filename_base="data/raw/hh/raw", max_pages=1, per_page=5)
    scraper.scrape()
