import hashlib
import logging
import os
import re
from datetime import datetime
from datetime import timedelta
from typing import Optional

import numpy as np
import pandas as pd

from src.utils.utils import current_week_info  # dict with keys 'week_number' and 'year'
from src.utils.utils import load_config


class JobDataPreProcessor:
    def __init__(
        self,
        logger: logging.Logger,
    ):
        self.logger = logger
        self.logger.info("Starting data preprocessing...")

        # Load configuration
        config = load_config()
        week_info = current_week_info()

        self.is_test = config["is_test"]

        # Construct input paths with week and year
        base_getmatch = config["preprocessing"]["input_filename_base"]["getmatch"]
        base_headhunter = config["preprocessing"]["input_filename_base"]["headhunter"]
        self.getmatch_path = (
            f"{base_getmatch}_week_{week_info['week_number']}_year_{week_info['year']}.csv"
        )
        self.headhunter_path = (
            f"{base_headhunter}_week_{week_info['week_number']}_year_{week_info['year']}.csv"
        )

        # Construct merged and output paths
        self.merged_path = f"{config['preprocessing']['merged_path']}_week_{week_info['week_number']}_year_{week_info['year']}.csv"
        self.historical_data_path = config["preprocessing"]["historical_data_path"]
        self.output_path = config["preprocessing"]["output_path"]
        self.bottom_percentile = config["preprocessing"]["salary_outliers"]["bottom_percentile"]
        self.top_percentile = config["preprocessing"]["salary_outliers"]["top_percentile"]

    @staticmethod
    def get_currency(salary_text: str) -> Optional[str]:
        X = str(salary_text).upper()
        if any(currency in X for currency in ["€", "EUR"]):
            return "EUR"
        elif any(currency in X for currency in ["£", "GBP"]):
            return "GBP"
        elif any(currency in X for currency in ["$", "USD"]):
            return "USD"
        elif any(currency in X for currency in ["₽", "RUB", "RUR"]):
            return "RUR"
        return None

    @staticmethod
    def list_to_string(x: str) -> str:
        try:
            x = eval(x)
            return ", ".join(x)
        except Exception:
            return x

    @staticmethod
    def which_language(description: str) -> str:
        cyrrilic_letters = len(re.findall(r"[а-яА-Я]", description))
        total_letters = len(re.findall(r"[a-zA-Zа-яА-Я]", description))
        if total_letters == 0:
            return "unknown"
        return "ru" if cyrrilic_letters / total_letters > 0.5 else "en"

    @staticmethod
    def replace_salary_patterns(text: str) -> str:
        pattern_v1 = re.compile(r"\b\d{1,3}( )?\d{3}( )?\d{3}\b|\b\d{3}( )?\d{3}\b")
        text = pattern_v1.sub("[NUMBER]", text)

        pattern_v2 = re.compile(
            r"((оклад|плата|от|до)\W{,3}\d[\d|\W]+\d)\D(?![%лгшчк])", flags=re.IGNORECASE
        )
        text = pattern_v2.sub(lambda m: f"{m.group(2)} [NUMBER]", text)
        return text.replace("₽", " рублей")

    @staticmethod
    def remove_description_duplicates(df: pd.DataFrame) -> pd.DataFrame:
        """Remove duplicate job descriptions using MD5 hashing.

        Args:
            df: DataFrame containing 'description' column

        Returns:
            DataFrame with duplicate descriptions removed
        """
        # Step 1: Add hashes
        df["description_hash"] = df["description"].apply(
            lambda x: hashlib.md5(x.replace(" ", "").encode()).hexdigest()
        )
        # Step 2: Remove duplicates and cleanup
        df = df.drop_duplicates(subset="description_hash", keep=False)
        df = df.drop(columns=["description_hash"])
        return df

    def process(self):
        self.logger.info("Loading datasets...")
        getmatch = pd.read_csv(self.getmatch_path)
        headhunter = pd.read_csv(self.headhunter_path)

        # Process Getmatch
        self.logger.info("Processing Getmatch data...")
        getmatch = getmatch.drop(columns=["Unnamed: 0"], errors="ignore")
        getmatch["source"] = "getmatch"
        getmatch["currency"] = getmatch["salary_text"].apply(self.get_currency)

        getmatch_short = getmatch.rename(
            columns={"description_text": "description", "company_name": "company", "level": "grade"}
        )[
            [
                "published_date",
                "title",
                "location",
                "company",
                "skills",
                "grade",
                "description",
                "salary_from",
                "salary_to",
                "currency",
                "source",
                "url",
            ]
        ]

        getmatch_short["skills"] = getmatch_short["skills"].apply(self.list_to_string)

        # Process headhunter
        self.logger.info("Processing headhunter data...")
        headhunter["source"] = "headhunter"
        headhunter.rename(columns={"area": "location"}, inplace=True)
        headhunter = headhunter[[col for col in headhunter.columns if col != "url"] + ["url"]]

        # Merge datasets
        self.logger.info("Merging datasets...")
        merged_data = pd.concat([getmatch_short, headhunter], ignore_index=True)
        self.logger.info(f"Length of merged data: {len(merged_data)}")

        # Save merged data
        self.logger.info(f"Saving merged data to {self.merged_path}...")
        merged_data.to_csv(self.merged_path, index=False)

        # Fill missing locations
        self.logger.info("Filling missing locations...")
        remote_mask = merged_data["location"].isna() & merged_data["description"].str.contains(
            "удаленн|удаленка|remote|удалённ|удалёнка", case=False
        )
        merged_data.loc[remote_mask, "location"] = "удаленно"
        merged_data["location"] = merged_data["location"].replace([None, ""], np.nan)
        merged_data.fillna({"location": "неизвестно"}, inplace=True)

        # Fill missing skills
        self.logger.info("Filling missing skills...")
        merged_data["skills"] = merged_data["skills"].replace([None, ""], np.nan)
        merged_data.fillna({"skills": "Не указаны"}, inplace=True)

        # Filter by currency
        self.logger.info("Filtering by currency (RUR)...")
        merged_data = merged_data[merged_data["currency"] == "RUR"]

        # Filter by description language
        self.logger.info("Filtering by language (Russian)...")
        merged_data["description_language"] = merged_data["description"].apply(self.which_language)
        merged_data = merged_data[merged_data["description_language"] == "ru"]

        # Remove duplicates
        self.logger.info("Removing duplicate descriptions...")
        merged_data = self.remove_description_duplicates(merged_data)

        # Drop rows with empty salary_from
        self.logger.info("Dropping rows with empty salary_from...")
        merged_data.dropna(subset=["salary_from"], inplace=True)

        # Rescale salaries
        self.logger.info("Rescaling salaries to thousands...")
        merged_data["salary_from"] /= 1000
        merged_data["salary_to"] /= 1000

        # Remove salary outliers
        self.logger.info("Removing salary outliers...")
        bottom_threshold = merged_data["salary_from"].quantile(self.bottom_percentile)
        top_threshold = merged_data["salary_from"].quantile(self.top_percentile)

        merged_data = merged_data[
            (merged_data["salary_from"] >= bottom_threshold)
            & (merged_data["salary_from"] <= top_threshold)
        ]

        # Log-transform salaries
        self.logger.info("Log-transforming salaries...")
        merged_data["log_salary_from"] = np.log(merged_data["salary_from"])
        merged_data["log_salary_to"] = np.log(merged_data["salary_to"])

        # Remove salary information from descriptions
        self.logger.info("Removing salary information from descriptions...")
        merged_data["description_no_numbers"] = merged_data["description"].apply(
            self.replace_salary_patterns
        )

        self.logger.info(f"Length of current data after cleaning: {len(merged_data)}")

        # Merge with historical data and filter
        self.logger.info("Loading and merging with historical data...")
        if os.path.exists(self.historical_data_path):
            historical_data = pd.read_csv(self.historical_data_path)
            self.logger.info(f"Length of historical data: {len(historical_data)}")
            merged_data["description"] = merged_data["description"]
            merged_data = pd.concat([historical_data, merged_data], ignore_index=True)

            # Filter out entries older than 6 months
            merged_data["published_date"] = pd.to_datetime(merged_data["published_date"])
            cutoff_date = datetime.now() - timedelta(days=180)
            merged_data = merged_data[merged_data["published_date"] >= cutoff_date]
            self.logger.info(f"Length of data after filtering old entries: {len(merged_data)}")

            # Sort by date and remove duplicates, keeping latest entries
            merged_data = merged_data.sort_values("published_date", ascending=False)
            merged_data.drop_duplicates(
                subset=["description", "company", "title"], keep="first", inplace=True
            )
            self.logger.info(f"Length of data after duplicates removal: {len(merged_data)}")

        self.logger.info(f"Length of data after cleaning: {len(merged_data)}")
        self.logger.info("Saving processed data...")
        merged_data.to_csv(self.historical_data_path, index=False)
        merged_data.to_csv(self.output_path, index=False)

        self.logger.info("Processing completed.")


def main(logger: logging.Logger):
    processor = JobDataPreProcessor(logger)
    processor.process()
