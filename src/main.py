from src.feature_building.main import main as build_features
from src.preprocessing.main import main as preprocess_data
from src.scraping.main import main as scrape_data
from src.training.main import main as train_models
from src.utils.utils import current_week_info  # dict with keys 'week_number' and 'year'
from src.utils.utils import setup_logging


def main():
    # Setup logging
    logger = setup_logging("data-training-pipeline")

    # Determine current week and year
    week_info = current_week_info()
    current_week, current_year = week_info["week_number"], week_info["year"]

    try:
        logger.info(f"Starting data-training-pipeline for Week {current_week}, Year {current_year}")
        # 1. Data Scraping
        scrape_data(logger)

        # 2. Preprocessing
        preprocess_data(logger)

        # 3. Feature Building
        build_features(logger)

        # 4. Model Training
        train_models(logger)

        logger.info("Pipeline completed successfully")

    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)


if __name__ == "__main__":
    main()
