import logging
import json
from config import config
import datetime
from pathlib import Path

from data.cache import JobCache, SearchCache
from data.database import Database
from data.dbcache_manager import DBCacheManager
from drivers.linkedin_driver import LinkedInIntegration


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def setup_output_dir():
    """Create output directory if it doesn't exist."""
    output_dir = Path(config.get("paths.output_dir"))
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir

def search_jobs():
    """Search for jobs based on configuration."""
    # Initialize the LinkedIn driver with credentials from config
    linkedin_config = config.get("credentials.linkedin")
    browser_config = config.get("selenium")

    try:
        linkedin_driver = LinkedInIntegration(
            email=linkedin_config.get("email"),
            password=linkedin_config.get("password"),
            headless=browser_config.get("headless", False)
        )

        # Explicitly call login before creating job searcher
        if not linkedin_driver.login():
            logger.error("Failed to log in to LinkedIn")
            return None
        db = Database(config.get("database.path"))
        job_cache = JobCache()
        search_cache = SearchCache()

        db_cache_manager = DBCacheManager(
            database=db,
            job_cache=job_cache,
            search_cache=search_cache
)
        # Create job searcher
        job_searcher = linkedin_driver.create_job_searcher(db)
        if job_searcher is None:
            logger.error("Failed to create job searcher")
            return None

        # Set up search parameters
        search_params = config["search_parameters"]
        job_title = search_params.get("job_title", "Software Engineer")
        location = search_params.get("location", "United States")
        max_jobs = search_params.get("max_jobs", 10)

        # Set up filters from config
        filters = {}
        if "search_filters" in config:
            filter_config = config["search_filters"]
            if "experience_level" in filter_config:
                filters["experience_level"] = filter_config["experience_level"]
            if "job_type" in filter_config:
                filters["job_type"] = filter_config["job_type"]
            if "date_posted" in filter_config:
                filters["date_posted"] = filter_config["date_posted"]
            if "remote" in filter_config and filter_config["remote"]:
                filters["workplace_type"] = ["Remote"]
            if "easy_apply" in filter_config:
                filters["easy_apply"] = filter_config["easy_apply"]

        logger.info(f"Searching for {job_title} jobs in {location}")
        job_listings = job_searcher.search_jobs(
            keywords=job_title,
            location=location,
            filters=filters,
            max_listings=max_jobs,
            scroll_pages=3  # Adjust as needed
        )

        return job_listings
    except Exception as e:
        logger.error(f"Error in job search: {e}")
        return None
    finally:
        # Ensure browser is closed
        if 'linkedin_driver' in locals() and linkedin_driver.driver:
            if hasattr(linkedin_driver.driver, 'quit'):
                linkedin_driver.driver.quit()
                logger.info("Browser closed successfully")

def main():
    """Main function to run the job search."""
    # Create output directory
    output_dir = setup_output_dir()

    # Search for jobs
    job_listings = search_jobs()

    if not job_listings:
        logger.error("No job listings found. Exiting.")
        return

    # Save results
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save job search results
    job_results_file = output_dir / f"job_search_{timestamp}.json"
    try:
        with open(job_results_file, 'w') as f:
            json.dump(job_listings, f, indent=2)
        logger.info(f"Saved {len(job_listings)} job listings to {job_results_file}")
    except Exception as e:
        logger.error(f"Error saving job search results: {e}")

    # Placeholder for application results (to be implemented)
    application_results = {
        "timestamp": timestamp,
        "total_jobs_found": len(job_listings),
        "jobs_applied": 0,
        "jobs_skipped": 0,
        "details": []
    }

    # Save application results
    app_results_file = output_dir / f"application_results_{timestamp}.json"
    try:
        with open(app_results_file, 'w') as f:
            json.dump(application_results, f, indent=2)
        logger.info(f"Results saved to {app_results_file}")
    except Exception as e:
        logger.error(f"Error saving application results: {e}")

    # Report summary
    logger.info(f"Application results: Found {len(job_listings)} jobs, Applied to {application_results['jobs_applied']}, Skipped {application_results['jobs_skipped']}")

if __name__ == "__main__":
    main()