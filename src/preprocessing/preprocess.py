import pandas as pd
import re
import hashlib
import numpy as np
from typing import Optional

from src.utils.utils import load_config, setup_logging

# Configure logging
logger = setup_logging()
logger.info("Starting data preprocessing...")

class JobDataPreProcessor:
    def __init__(self,):
        # Load configuration
        config = load_config()

        self.is_test = config['is_test']

        self.getmatch_path = config['preprocessing']['input_paths']['getmatch']
        self.hh_path = config['preprocessing']['input_paths']['hh']
        self.merged_path = config['preprocessing']['merged_path']
        self.output_path = config['preprocessing']['output_path']

        self.bottom_percentile = config['preprocessing']['salary_outliers']['bottom_percentile']
        self.top_percentile = config['preprocessing']['salary_outliers']['top_percentile']

    @staticmethod
    def get_currency(salary_text: str) -> Optional[str]:
        X = str(salary_text).upper()
        if any(currency in X for currency in ['€', 'EUR']):
            return 'EUR'
        elif any(currency in X for currency in ['£', 'GBP']):
            return 'GBP'
        elif any(currency in X for currency in ['$', 'USD']):
            return 'USD'
        elif any(currency in X for currency in ['₽', 'RUB', 'RUR']):
            return 'RUR'
        return None

    @staticmethod
    def list_to_string(x: str) -> str:
        try:
            x = eval(x)
            return ', '.join(x)
        except:
            return x

    @staticmethod
    def which_language(description: str) -> str:
        cyrrilic_letters = len(re.findall(r'[а-яА-Я]', description))
        total_letters = len(re.findall(r'[a-zA-Zа-яА-Я]', description))
        if total_letters == 0:
            return 'unknown'
        return 'ru' if cyrrilic_letters / total_letters > 0.5 else 'en'

    @staticmethod
    def replace_salary_patterns(text: str) -> str:
        pattern_v1 = re.compile(r'\b\d{1,3}( )?\d{3}( )?\d{3}\b|\b\d{3}( )?\d{3}\b')
        text = pattern_v1.sub('[NUMBER]', text)

        pattern_v2 = re.compile(r'((оклад|плата|от|до)\W{,3}\d[\d|\W]+\d)\D(?![%лгшчк])', flags=re.IGNORECASE)
        text = pattern_v2.sub(lambda m: f"{m.group(2)} [NUMBER]", text)
        return text.replace('₽', ' рублей')

    def process(self):
        logger.info("Loading datasets...")
        getmatch = pd.read_csv(self.getmatch_path)
        hh = pd.read_csv(self.hh_path)

        # Process Getmatch
        logger.info("Processing Getmatch data...")
        getmatch = getmatch.drop(columns=['Unnamed: 0'], errors='ignore')
        getmatch['source'] = 'getmatch'
        getmatch['currency'] = getmatch['salary_text'].apply(self.get_currency)

        getmatch_short = getmatch.rename(columns={
            'description_text': 'description',
            'company_name': 'company',
            'level': 'grade'
        })[[
            'title', 'location', 'company', 'skills', 'grade', 'description',
            'salary_from', 'salary_to', 'currency', 'source', 'url'
        ]]

        getmatch_short['skills'] = getmatch_short['skills'].apply(self.list_to_string)

        # Process HH
        logger.info("Processing HH data...")
        hh['source'] = 'hh'
        hh.rename(columns={'area': 'location'}, inplace=True)
        hh = hh[[col for col in hh.columns if col != 'url'] + ['url']]

        # Merge datasets
        logger.info("Merging datasets...")
        merged_data = pd.concat([getmatch_short, hh], ignore_index=True)
        logger.info(f"Length of merged data: {len(merged_data)}")

        # Save merged data
        logger.info(f"Saving merged data to {self.merged_path}...")
        merged_data.to_csv(self.merged_path, index=False)

        # Fill missing locations
        logger.info("Filling missing locations...")
        remote_mask = merged_data['location'].isna() & merged_data['description'].str.contains('удаленн|удаленка|remote|удалённ|удалёнка', case=False)
        merged_data.loc[remote_mask, 'location'] = 'удаленно'
        merged_data['location'] = merged_data['location'].replace([None, ''], np.nan)
        merged_data.fillna({'location': 'неизвестно'}, inplace=True)

        # Fill missing skills
        logger.info("Filling missing skills...")
        merged_data['skills'] = merged_data['skills'].replace([None, ''], np.nan)
        merged_data.fillna({'skills': 'Не указаны'}, inplace=True)

        # Filter by currency
        logger.info("Filtering by currency (RUR)...")
        merged_data = merged_data[merged_data['currency'] == 'RUR']

        # Filter by description language
        logger.info("Filtering by language (Russian)...")
        merged_data['description_language'] = merged_data['description'].apply(self.which_language)
        merged_data = merged_data[merged_data['description_language'] == 'ru']

        # Remove duplicates
        logger.info("Removing duplicate descriptions...")
        merged_data['description_hash'] = merged_data['description'].apply(lambda x: hashlib.md5(x.encode()).hexdigest())
        merged_data = merged_data.drop_duplicates(subset='description_hash', keep=False)
        merged_data.drop(columns=['description_hash'], inplace=True)

        # Drop rows with empty salary_from
        logger.info("Dropping rows with empty salary_from...")
        merged_data.dropna(subset=['salary_from'], inplace=True)

        # Rescale salaries
        logger.info("Rescaling salaries to thousands...")
        merged_data['salary_from'] /= 1000
        merged_data['salary_to'] /= 1000

        # Remove salary outliers
        logger.info("Removing salary outliers...")
        bottom_threshold = merged_data['salary_from'].quantile(self.bottom_percentile)
        top_threshold = merged_data['salary_from'].quantile(self.top_percentile)
        
        merged_data = merged_data[
            (merged_data['salary_from'] >= bottom_threshold) &
            (merged_data['salary_from'] <= top_threshold)
        ]

        # Log-transform salaries
        logger.info("Log-transforming salaries...")
        merged_data['log_salary_from'] = np.log(merged_data['salary_from'])
        merged_data['log_salary_to'] = np.log(merged_data['salary_to'])

        # Remove salary information from descriptions
        logger.info("Removing salary information from descriptions...")
        merged_data['description_no_numbers'] = merged_data['description'].apply(self.replace_salary_patterns)

        # Save the processed data
        logger.info(f"Length of data after cleaning: {len(merged_data)}")
        logger.info(f"Saving processed data to {self.output_path}...")

        merged_data.to_csv(self.output_path, index=False)
        logger.info("Processing completed.")

if __name__ == "__main__":
    processor = JobDataPreProcessor()
    processor.process()
