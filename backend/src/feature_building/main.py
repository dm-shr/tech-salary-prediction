import logging
import os
import re
from typing import Literal
from typing import Optional
from typing import Tuple

import pandas as pd
from dotenv import load_dotenv

from src.preprocessing.main import JobDataPreProcessor
from src.utils.utils import current_week_info  # dict with keys 'week_number' and 'year'
from src.utils.utils import load_config
from src.utils.utils import setup_logging

try:
    from google import genai
    from pydantic import BaseModel

    class TranslationResult(BaseModel):
        company: str
        description: str
        location: str

    translation_enabled = True
    print("google-generativeai is installed. Translation will be enabled.")
except ImportError:
    print("google-generativeai is not installed. Translation will be disabled.")
    translation_enabled = False


load_dotenv()


class FeatureBuilder:
    def __init__(
        self,
        logger: logging.Logger,
        is_inference: bool = False,
    ):
        """Initialize the FeatureBuilder with configuration and data loading."""
        try:
            self.logger = logger
            self.is_inference = is_inference
            # Load configuration
            config = load_config()
            self.config = config
            self.is_test = config["is_test"]
            self.transformer_enabled = config["models"]["transformer"]["enabled"]

            # Initialize transformer-specific components if enabled
            if self.transformer_enabled:
                self.logger.info("Initializing transformer dependencies...")
                import torch
                from transformers import AutoTokenizer

                self.torch = torch

                tokenizer_name = (
                    config["features"]["transformer"]["tokenizer"]
                    if not self.is_test
                    else config["features"]["transformer"]["tokenizer_test"]
                )
                self.tokenizer_transformer = AutoTokenizer.from_pretrained(tokenizer_name)

            # Get current week info
            week_info = current_week_info()
            week_suffix = f"_week_{week_info['week_number']}_year_{week_info['year']}"

            # Get paths from configuration and append week suffix if needed
            preprocessed_base = config["features"]["preprocessed_data_base"]
            self.preprocessed_data_path = f"{preprocessed_base}.csv"

            output_base = config["features"]["output_base"]
            self.output_file_path = f"{output_base}{week_suffix}.csv"

            self.target_name = config["features"]["target_name"]
            target_base = config["features"]["target_base"]
            self.target_output_path = f"{target_base}{week_suffix}"

            # Update transformer paths only if enabled
            if self.transformer_enabled:
                transformer_base = config["features"]["transformer"]["features_base"]
                for item in config["features"]["transformer"]["feature_processing"]:
                    item["path"] = f"{transformer_base}/{item['name']}{week_suffix}.pt"
                self.transformer_feature_processing = config["features"]["transformer"][
                    "feature_processing"
                ]
            self.add_query_prefix = config["features"]["transformer"]["add_query_prefix"]

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
            self.text_features = []
            if self.transformer_enabled:
                self.text_features.extend(config["features"]["features"]["transformer"]["text"])
            self.text_features.extend(config["features"]["features"]["bi_gru_cnn"]["text"])
            self.text_features = list(set(self.text_features))

            self.logger.info(f"Input file path: {self.preprocessed_data_path}")
            self.logger.info(f"Output file path: {self.output_file_path}")

            # Initialize data as None for inference mode
            self.data = None
            if not is_inference:
                self.data = pd.read_csv(self.preprocessed_data_path)
                self.logger.info(f"Data loaded successfully with {len(self.data)} rows.")

        except Exception as e:
            self.logger.error(f"Failed to initialize FeatureBuilder: {str(e)}")
            raise

    async def _translate_with_gemini(
        self, company: str, description: str, location: str
    ) -> "TranslationResult":
        """Translates company, description, and location using Google Gemini API."""
        if not translation_enabled:
            self.logger.warning("Translation is disabled.")
            return TranslationResult(company=company, description=description, location=location)

        try:
            client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

            prompt = f"""
            You are an expert in localizing job market information for Russia.
            Given the following job posting details, your task is to:
            1.  Convert the company name to a well-known Russian company \
(e.g., Yandex, Avito, Sber, Mail, VK, EPAM, X5 Retail Group). Be creative!
            2.  Translate the job description into Russian.
            3.  Convert the location to a relevant location within Russia.

            Here are the details:
            - Company: {company}
            - Description: {description}
            - Location: {location}

            Provide the localized information in the following JSON format:
            {{
                "company": "Russian Company Name",
                "description": "Russian Translation of the Job Description",
                "location": "Location in Russia"
            }}
            """

            self.logger.info(f"Translation prompt: {prompt}")

            response = await client.aio.models.generate_content(
                model="gemini-1.5-flash",
                contents=prompt,
                config={
                    "response_mime_type": "application/json",
                    "response_schema": TranslationResult,
                },
            )

            self.logger.info(f"Translation response: {response.text}")

            translation_result: TranslationResult = response.parsed

            return translation_result

        except Exception as e:
            self.logger.error(f"Translation failed: {str(e)}")
            return TranslationResult(company=company, description=description, location=location)

    async def prepare_set_inference_data(
        self,
        title: str,
        company: str,
        location: str,
        description: str,
        skills: str,
        experience_from: int,
        experience_to: int,
        source: Literal["headhunter", "getmatch"] = "headhunter",
    ) -> pd.DataFrame:
        """Prepare input data for inference, translate if needed, and set the data attribute."""
        self.logger.info("Preparing inference data...")
        if self.is_inference and translation_enabled:
            self.logger.info("Translating data with Gemini API...")
            translation = await self._translate_with_gemini(company, description, location)
            company = translation.company
            description = translation.description
            location = translation.location

        input_dict = {
            "title": [title],
            "company": [company],
            "location": [location],
            "description": [description],
            "skills": [skills],
            "experience_from": [experience_from],
            "experience_to": [experience_to],
            "source": [source],
        }

        self.data = pd.DataFrame(input_dict)

    @property
    def data(self):
        """Get the current data."""
        return self._data

    @data.setter
    def data(self, value):
        """Set the data for processing."""
        if value is not None and not isinstance(value, pd.DataFrame):
            raise ValueError("Data must be a pandas DataFrame")
        self._data = value

    def merge_skills_and_descriptions(self) -> None:
        """Merge skills and descriptions to create a new feature."""
        try:
            self.logger.info("Merging skills and descriptions...")
            if self.is_inference:
                self.data["description_no_numbers"] = self.data["description"].apply(
                    JobDataPreProcessor.replace_salary_patterns
                )
            self.data["description_no_numbers_with_skills"] = (
                self.data["description_no_numbers"] + " " + self.data["skills"]
            )
            self.logger.info("Skills and descriptions merged successfully.")
        except Exception as e:
            self.logger.error(f"Failed to merge skills and descriptions: {str(e)}")
            raise

    def extract_numbers(self, text: str) -> Tuple[Optional[int], Optional[int]]:
        """
        Extract years of experience from text.

        Args:
            text (str): Input text containing experience information

        Returns:
            Tuple[Optional[int], Optional[int]]: Minimum and maximum years of experience
        """
        try:
            self.logger.debug("Extracting numbers from text...")
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
            self.logger.error(f"Error extracting numbers from text: {str(e)}")
            return None, None

    def process_experience(self) -> None:
        """Extracts experience from the description and fills missing values.
        When not in inference mode, fills missing experience values based on grade medians."""
        try:
            self.logger.info("Processing experience data...")
            # Fill missing experience from description
            empty_experience_mask = self.data["experience_from"].isna()
            extracted_values = self.data.loc[empty_experience_mask, "description"].apply(
                self.extract_numbers
            )
            self.data.loc[empty_experience_mask, "experience_from"] = extracted_values.apply(
                lambda x: x[0]
            )
            self.data.loc[empty_experience_mask, "experience_to"] = extracted_values.apply(
                lambda x: x[1]
            )
            self.logger.info(f"Filled {empty_experience_mask.sum()} missing experience values.")

            # Fill by grade if not in inference mode
            if not self.is_inference:
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
                self.logger.info("Missing experience based on grade filled successfully.")

            # Adjust upper bound
            self.data["experience_to_adjusted_10"] = self.data["experience_to"].apply(
                lambda x: 10 if x == -1 else x
            )
            self.logger.info("Experience processing completed successfully.")

        except Exception as e:
            self.logger.error(f"Failed to process experience data: {str(e)}")
            raise

    def add_description_size_feature(self) -> None:
        """Add a feature for the description length (in number of words)."""
        try:
            self.logger.info("Adding description size (word count)...")
            self.data["description_size"] = (
                self.data["description_no_numbers_with_skills"]
                .str.split()
                .str.len()
                .astype(int)  # Added explicit conversion to int
            )
            self.logger.info("Description size added successfully.")
        except Exception as e:
            self.logger.error(f"Failed to add description size: {str(e)}")
            raise

    def add_title_company_location_skills_source_feature(self) -> None:
        """Add a feature for combined title, company, location, skills, and source."""
        try:
            self.logger.info(
                "Adding combined title, company, location, skills, and source feature..."
            )
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
            self.logger.info(
                "Title, company, location, skills, and source feature added successfully."
            )
        except Exception as e:
            self.logger.error(
                f"Failed to add title, company, location, skills, and source feature: {str(e)}"
            )
            raise

    def add_query_prefix_to_text_features(self) -> None:
        """Add 'query: ' prefix to text features if transformer is enabled."""
        if not self.transformer_enabled:
            self.logger.info("Skipping query prefix addition - transformer is disabled")
            return

        try:
            self.logger.info("Adding 'query: ' prefix to text features...")
            for feature in self.text_features:
                self.data[feature] = self.data[feature].apply(lambda x: f"query: {x}")
            self.logger.info("Prefix added successfully.")
        except Exception as e:
            self.logger.error(f"Failed to add 'query: ' prefix to text features: {str(e)}")
            raise

    def process_and_get_transformer_features(self) -> Optional[dict]:
        """Process text features for transformer if enabled."""
        if not self.transformer_enabled:
            self.logger.info("Skipping transformer feature processing - transformer is disabled")
            return None

        try:
            self.logger.info("Processing transformer features...")
            features_dict = {}

            for feature_processing_dict in self.transformer_feature_processing:
                feature = feature_processing_dict["name"]
                max_len = feature_processing_dict["max_len"]

                tokenized_data = self.tokenizer_transformer(
                    self.data[feature].tolist(),
                    max_length=max_len,
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt",
                )

                if self.is_inference:
                    features_dict[feature] = tokenized_data
                else:
                    feature_path = feature_processing_dict["path"]
                    self.torch.save(tokenized_data, feature_path)
                    self.logger.info(f"Saved {feature} to {feature_path}")

            return features_dict if self.is_inference else None

        except Exception as e:
            self.logger.error(f"Failed to process transformer features: {str(e)}")
            raise

    def process_and_get_catboost_features(self) -> Optional[pd.DataFrame]:
        """Process and either save or return CatBoost features."""
        try:
            catboost_features = self.data[self.catboost_features]

            if not self.is_inference:
                catboost_features.to_csv(self.catboost_features_path, index=False)
                self.logger.info(f"Saved CatBoost features to {self.catboost_features_path}")

            return catboost_features if self.is_inference else None

        except Exception as e:
            self.logger.error(f"Failed to process CatBoost features: {str(e)}")
            raise

    def save_target(self) -> None:
        """When not in inference mode, save the target data."""
        try:
            if self.is_inference:
                return None

            target_values = self.data[[self.target_name]].values

            # Save CSV version regardless of transformer status
            self.data[[self.target_name]].to_csv(self.target_output_path + ".csv", index=False)

            if self.transformer_enabled:
                self.torch.save(target_values, self.target_output_path + ".pt")
                self.logger.info(f"Target data saved to {self.target_output_path} (.csv and .pt)")
            else:
                self.logger.info(
                    f"Target data saved to {self.target_output_path}.csv (skipped .pt - transformer disabled)"
                )
            return None

        except Exception as e:
            self.logger.error(f"Failed to process target: {str(e)}")
            raise

    def _build(self) -> None:
        """Build all features in sequence."""
        try:
            self.logger.info("Processing all features...")
            self.merge_skills_and_descriptions()
            self.process_experience()
            self.add_description_size_feature()
            self.add_title_company_location_skills_source_feature()  # Moved before query prefix addition
            if self.add_query_prefix:
                self.add_query_prefix_to_text_features()
            if self.is_test:
                test_size = self.config["features"]["test_size"]
                # drop test_size % of rows
                self.data = self.data.sample(frac=test_size, random_state=42)
            self.logger.info("All features processed successfully.")
        except Exception as e:
            self.logger.error(f"Failed to process features: {str(e)}")
            raise

    def build(self) -> Optional[dict]:
        """Run the feature building process and return features if in inference mode."""
        try:
            if self.data is None:
                raise ValueError("No data available for processing")

            self.logger.info("Starting feature building process...")
            self._build()

            results = {
                "catboost_features": self.process_and_get_catboost_features(),
                "transformer_features": self.process_and_get_transformer_features(),
            }

            if not self.is_inference:
                self.save_target()
                return None

            return results

        except Exception as e:
            self.logger.error(f"Feature building process failed: {str(e)}")
            raise


def main(logger: logging.Logger):
    try:
        feature_builder = FeatureBuilder(logger, is_inference=False)
        feature_builder.build()
    except Exception as e:
        logger.critical(f"Application failed: {str(e)}")
        raise


if __name__ == "__main__":
    logger = setup_logging()
    main(logger)
