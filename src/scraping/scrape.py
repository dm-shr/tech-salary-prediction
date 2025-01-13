import os
import logging
from typing import Dict, Any

from src.utils.utils import load_config, setup_logging

from src.scraping.headhunter import HeadhunterJobScraper
from src.scraping.getmatch import GetmatchJobScraper


def create_directories(config: Dict[str, Any]):
    """Create necessary directories if they don't exist."""
    for source in ['headhunter', 'getmatch']:
        if config.get(source, {}).get('enabled', False):
            os.makedirs(config[source]['data_dir'], exist_ok=True)

def run_scrapers(config: Dict[str, Any], logger: logging.Logger):
    """Run enabled scrapers with their respective configurations."""
    # Run HeadHunter scraper if enabled
    if config.get('headhunter', {}).get('enabled', False):
        logger.info("Starting HeadHunter scraping...")
        try:
            hh_config = config['headhunter']
            scraper = HeadhunterJobScraper(
                data_dir=hh_config['data_dir'],
                start_date=hh_config['start_date'],
                end_date=hh_config['end_date'],
                per_page=hh_config['per_page'],
                max_pages=hh_config['max_pages']
            )
            scraper.scrape()
            logger.info("HeadHunter scraping completed successfully")
        except Exception as e:
            logger.error(f"Error during HeadHunter scraping: {str(e)}")

    # Run GetMatch scraper if enabled
    if config.get('getmatch', {}).get('enabled', False):
        logger.info("Starting GetMatch scraping...")
        try:
            gm_config = config['getmatch']
            scraper = GetmatchJobScraper(
                data_dir=gm_config['data_dir'],
                num_pages=gm_config['num_pages'],
                output_format=gm_config['output_format']
            )
            scraper.scrape()
            logger.info("GetMatch scraping completed successfully")
        except Exception as e:
            logger.error(f"Error during GetMatch scraping: {str(e)}")

def main():
    """Main function to orchestrate the scraping process."""
    logger = setup_logging()
    logger.info("Starting scraping process...")
    
    try:
        # Load configuration
        config = load_config()
        config = config.get('scraping', {})
        
        # Create necessary directories
        create_directories(config)
        
        # Run scrapers
        run_scrapers(config, logger)
        
        logger.info("Scraping process completed")
    except Exception as e:
        logger.error(f"Fatal error in scraping process: {str(e)}")
        raise

if __name__ == "__main__":
    main()
