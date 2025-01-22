import re
from typing import Optional
from typing import Tuple

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer

from src.utils.utils import current_week_info  # dict with keys 'week_number' and 'year'
from src.utils.utils import load_config
from src.utils.utils import setup_logging


# Configure logging
logger = setup_logging()
logger.info("Starting feature building...")


class FeatureBuilder:
    def __init__(self):
        """Initialize the FeatureBuilder with configuration and data loading."""
        try:
            # Load configuration
            config = load_config()
            self.config = config

            self.is_test = config["is_test"]

            # Get current week info
            week_info = current_week_info()
            week_suffix = f"_week_{week_info['week_number']}_year_{week_info['year']}"

            # Get paths from configuration and append week suffix
            preprocessed_base = config["features"]["preprocessed_data_base"]
            self.preprocessed_data_path = f"{preprocessed_base}{week_suffix}.csv"

            output_base = config["features"]["output_base"]
            self.output_file_path = f"{output_base}{week_suffix}.csv"

            self.target_name = config["features"]["target_name"]
            target_base = config["features"]["target_base"]
            self.target_output_path = f"{target_base}{week_suffix}"

            # Update transformer paths with week suffix
            transformer_base = config["features"]["transformer"]["features_base"]
            for item in config["features"]["transformer"]["feature_processing"]:
                item["path"] = f"{transformer_base}/{item['name']}{week_suffix}.pt"
            self.transformer_feature_processing = config["features"]["transformer"][
                "feature_processing"
            ]

            # Update catboost path with week suffix
            catboost_base = config["features"]["catboost"]["features_base"]
            self.catboost_features_path = f"{catboost_base}{week_suffix}.csv"

            # Extract features from nested dictionary of features

            def extract_features(nested_dict):

                features = []
                for _, value in nested_dict.items():
                    if isinstance(value, dict):
                        features.extend(extract_features(value))
                    else:
                        features.extend(value)
                return features

            self.catboost_features = extract_features(config["features"]["features"]["catboost"])
            # get text features
            self.text_features = (
                config["features"]["features"]["transformer"]["text"]
                + config["features"]["features"]["bi_gru_cnn"]["text"]
            )
            self.text_features = list(set(self.text_features))

            self.add_query_prefix = config["features"]["transformer"][
                "add_query_prefix"
            ]  # whether to add 'query: ' prefix to text features

            logger.info(f"Input file path: {self.preprocessed_data_path}")
            logger.info(f"Output file path: {self.output_file_path}")

            # Load tokenizer
            tokenizer_name = (
                config["features"]["transformer"]["tokenizer"]
                if not self.is_test
                else config["features"]["transformer"]["tokenizer_test"]
            )
            self.tokenizer_transformer = AutoTokenizer.from_pretrained(tokenizer_name)

            # Load the data from the CSV file
            self.data = pd.read_csv(self.preprocessed_data_path)
            logger.info(f"Data loaded successfully with {len(self.data)} rows.")

            # NOTE: REMOVE LATER #
            self.data["location"] = self.data["location"].replace([None, ""], np.nan)
            self.data.fillna({"location": "неизвестно"}, inplace=True)
            self.data["skills"] = self.data["skills"].replace([None, ""], np.nan)
            self.data.fillna({"skills": "неизвестно"}, inplace=True)
            # NOTE: REMOVE LATER #

        except Exception as e:
            logger.error(f"Failed to initialize FeatureBuilder: {str(e)}")
            raise

    def merge_skills_and_descriptions(self) -> None:
        """Merge skills and descriptions to create a new feature."""
        try:
            logger.info("Merging skills and descriptions...")
            self.data["description_no_numbers_with_skills"] = (
                self.data["description_no_numbers"] + " " + self.data["skills"]
            )
            logger.info("Skills and descriptions merged successfully.")
        except Exception as e:
            logger.error(f"Failed to merge skills and descriptions: {str(e)}")
            raise

    @staticmethod
    def extract_numbers(text: str) -> Tuple[Optional[int], Optional[int]]:
        """
        Extract years of experience from text.

        Args:
            text (str): Input text containing experience information

        Returns:
            Tuple[Optional[int], Optional[int]]: Minimum and maximum years of experience
        """
        try:
            logger.debug("Extracting numbers from text...")
            if not isinstance(text, str):
                return None, None

            text = text.replace("ё", "е").replace("–", "-")

            # Patterns to be replaced with numbers
            word_patterns = {
                1: [r"один", r"одного", r"одним"],
                2: ["два", "двух", "двум", "двумя", "полутора", "полтора"],
                3: ["три", "трех", "трем", "тремя"],
                4: ["четыре", "четырех", "четырьмя"],
                5: ["пять", "пяти", "пятью"],
                6: ["шесть", "шести", "шестью"],
                7: ["семь", "семи", "семью"],
                8: ["восемь", "восьми", "восемью"],
                9: ["девять", "девяти", "девятью"],
                10: ["десять", "десяти", "десятью"],
            }

            # Replace word patterns with numbers
            for number, patterns in word_patterns.items():
                for pattern in patterns:
                    text = re.sub(pattern + r"\W", f"{number} ", text, flags=re.IGNORECASE)

            # Replace specific patterns
            replacements = [
                (r"(от|до|более|больше|менее)\s*год(а)", r"\1 1 год"),
                (r"\d{1,2}\s{0,2}месяц", "1 год"),
                (r"[2-9]\d{1,3}", ""),
            ]

            for pattern, replacement in replacements:
                text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)

            # Extract experience ranges
            experience_patterns = [
                r"от\s*года",
                r"(?:(от|более|не менее)\s{0,5})(\d{1,2})(?:-(?:го|й|х|и|ти))?(?:\s*(?:до|–|-)\s*(\d{1,2})(?:-(?:го|й|х|и))?)",
                r"(?:(от|более|не менее)\s{0,5})(\d{1,2})\s{0,5}(год|лет)",
                r"(\d{1,2})\+\s{0,5}(год|лет)",
                r"(\d{1,2})(?:-?(?:го|й|х|и|ти))?\s*(год|лет)",
            ]

            all_matches = []
            for pattern in experience_patterns:
                matches = re.findall(pattern, text, re.IGNORECASE)
                all_matches.extend([match for match in matches if match])

            # Process matches
            numbers = set()
            for match in all_matches:
                numbers.update(int(num) for num in re.findall(r"\d+", " ".join(match)))

            if not numbers:
                return None, None

            min_exp = min(numbers)
            max_exp = max(numbers)

            return min_exp, -1 if min_exp == max_exp else max_exp

        except Exception as e:
            logger.error(f"Error extracting numbers from text: {str(e)}")
            return None, None

    def fill_missing_experience(self) -> None:
        """Fill missing experience values based on description."""
        try:
            logger.info("Filling missing experience data...")
            empty_experience_mask = self.data["experience_from"].isna()

            # Apply extract_numbers to descriptions where experience is missing
            extracted_values = self.data.loc[empty_experience_mask, "description"].apply(
                self.extract_numbers
            )

            self.data.loc[empty_experience_mask, "experience_from"] = extracted_values.apply(
                lambda x: x[0]
            )
            self.data.loc[empty_experience_mask, "experience_to"] = extracted_values.apply(
                lambda x: x[1]
            )

            logger.info(f"Filled {empty_experience_mask.sum()} missing experience values.")
        except Exception as e:
            logger.error(f"Failed to fill missing experience: {str(e)}")
            raise

    def fill_experience_by_grade(self) -> None:
        """Fill missing experience values based on median experience per grade."""
        try:
            logger.info("Filling missing experience based on grade...")
            empty_experience_mask = self.data["experience_from"].isna()
            empty_grade_mask = self.data["grade"].isna()

            # Calculate median experience by grade
            grouped_by_grade = (
                self.data[~empty_experience_mask & ~empty_grade_mask]
                .groupby("grade")[["experience_from", "experience_to"]]
                .median()
            )

            # Fill missing values based on grade medians
            for grade, (experience_from, experience_to) in grouped_by_grade.iterrows():
                mask = (self.data["grade"] == grade) & empty_experience_mask
                self.data.loc[mask, ["experience_from", "experience_to"]] = (
                    experience_from,
                    experience_to,
                )

            logger.info("Missing experience based on grade filled successfully.")
        except Exception as e:
            logger.error(f"Failed to fill experience by grade: {str(e)}")
            raise

    def adjust_experience_upper_bound(self) -> None:
        """Adjust the upper bound of experience to 10 if not specified."""
        try:
            logger.info("Adjusting upper bound of experience...")
            self.data["experience_to_adjusted_10"] = self.data["experience_to"].apply(
                lambda x: 10 if x == -1 else x
            )
            logger.info("Upper bound of experience adjusted successfully.")
        except Exception as e:
            logger.error(f"Failed to adjust experience upper bound: {str(e)}")
            raise

    def add_description_size_feature(self) -> None:
        """Add a feature for the description length (in number of words)."""
        try:
            logger.info("Adding description size (word count)...")
            self.data["description_size"] = (
                self.data["description_no_numbers_with_skills"].str.split().str.len()
            )
            logger.info("Description size added successfully.")
        except Exception as e:
            logger.error(f"Failed to add description size: {str(e)}")
            raise

    def add_title_company_location_skills_source_feature(self) -> None:
        """Add a feature for combined title, company, location, skills, and source."""
        try:
            logger.info("Adding combined title, company, location, skills, and source feature...")
            template = (
                "Позиция: {position}\n"
                "Компания: {company}\n"
                "Место: {location}\n"
                "Навыки: {skills}\n"
                "Источник: {source}"
            )
            self.data["title_company_location_skills_source"] = self.data.apply(
                lambda x: template.format(
                    position=x["title"],
                    company=x["company"],
                    location=x["location"],
                    skills=x["skills"],
                    source=x["source"],
                ),
                axis=1,
            )
            logger.info("Title, company, location, skills, and source feature added successfully.")
        except Exception as e:
            logger.error(
                f"Failed to add title, company, location, skills, and source feature: {str(e)}"
            )
            raise

    def add_query_prefix_to_text_features(self) -> None:
        """Add 'query: ' prefix to text features."""
        try:
            logger.info("Adding 'query: ' prefix to text features...")
            for feature in self.text_features:
                self.data[feature] = self.data[feature].apply(lambda x: f"query: {x}")
            logger.info("Prefix added successfully.")
        except Exception as e:
            logger.error(f"Failed to add 'query: ' prefix to text features: {str(e)}")
            raise

    def tokenize_and_save_transformer_features_to_pt(self) -> None:
        """Tokenize text features for a transformer model and save them separately as Torch files."""
        try:
            logger.info("Tokenizing and saving text features...")
            for feature_processing_dict in self.transformer_feature_processing:
                feature = feature_processing_dict["name"]
                max_len = feature_processing_dict["max_len"]

                feature_path = feature_processing_dict["path"]

                logger.info(f"Processing feature: {feature}")
                tokenized_data = self.tokenizer_transformer(
                    self.data[feature].tolist(),
                    max_length=max_len,
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt",
                )

                torch.save(tokenized_data, feature_path)

                logger.info(f"Tokenized data for '{feature}' saved to {feature_path}")
        except Exception as e:
            logger.error(f"Failed to tokenize and save text features: {str(e)}")
            raise

    def process_features(self) -> None:
        """Process all features in sequence."""
        try:
            logger.info("Processing all features...")
            self.merge_skills_and_descriptions()
            self.fill_missing_experience()
            self.fill_experience_by_grade()
            self.adjust_experience_upper_bound()
            self.add_description_size_feature()
            self.add_title_company_location_skills_source_feature()
            if self.add_query_prefix:
                self.add_query_prefix_to_text_features()
            if self.is_test:
                test_size = self.config["features"]["test_size"]
                # drop test_size % of rows
                self.data = self.data.sample(frac=test_size, random_state=42)
            logger.info("All features processed successfully.")
        except Exception as e:
            logger.error(f"Failed to process features: {str(e)}")
            raise

    def save_target(self) -> None:
        """Save the processed target data to a Torch and CSV files."""
        try:
            logger.info("Saving target data to PT file...")
            torch.save(self.data[[self.target_name]].values, self.target_output_path + ".pt")
            logger.info(f"Target data saved successfully to {self.target_output_path + '.pt'}.")
            logger.info("Saving target data to CSV file...")
            self.data[[self.target_name]].to_csv(self.target_output_path + ".csv", index=False)
            logger.info(f"Target data saved successfully to {self.target_output_path + '.csv'}.")
        except Exception as e:
            logger.error(f"Failed to save target data: {str(e)}")
            raise

    def save_catboost_features_to_csv(self) -> None:
        """Save the processed features data to a CSV file for CatBoost."""
        try:
            logger.info("Saving features data to CSV file for CatBoost...")
            self.data[self.catboost_features].to_csv(self.catboost_features_path, index=False)
            logger.info(f"Features data saved successfully to {self.catboost_features_path}.")
        except Exception as e:
            logger.error(f"Failed to save features data to CSV: {str(e)}")
            raise

    def run(self) -> None:
        """Main method to run the entire feature building process."""
        try:
            logger.info("Starting feature building process...")
            self.process_features()
            self.tokenize_and_save_transformer_features_to_pt()
            self.save_catboost_features_to_csv()
            self.save_target()
            logger.info("Feature building process completed successfully.")
        except Exception as e:
            logger.error(f"Feature building process failed: {str(e)}")
            raise


if __name__ == "__main__":
    try:
        feature_builder = FeatureBuilder()
        feature_builder.run()
    except Exception as e:
        logger.critical(f"Application failed: {str(e)}")
        raise
