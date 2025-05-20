import json
import logging
import os
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
import uuid
import asyncpg
from contextlib import asynccontextmanager

from dataModels.data_models import Job, Resume, SearchHistory, JobStatus

# Configure logging
logger = logging.getLogger(__name__)

class Database:
    """PostgreSQL database layer for storing job application data."""

    def __init__(self, db_url=None):
        """
        Initialize the database connection.

        Args:
            db_url: PostgreSQL connection string
        """
        self.db_url = db_url or os.environ.get('DATABASE_URL')
        if not self.db_url:
            raise ValueError("No database URL provided")

        # Connection pool will be initialized during application startup
        self.pool = None

    async def initialize_pool(self):
        """Initialize connection pool."""
        if self.pool is None:
            self.pool = await asyncpg.create_pool(
                dsn=self.db_url,
                min_size=1,
                max_size=10,
                command_timeout=60
            )
            logger.info("PostgreSQL connection pool initialized")

    async def close_pool(self):
        """Close connection pool."""
        if self.pool:
            await self.pool.close()
            self.pool = None
            logger.info("PostgreSQL connection pool closed")

    @asynccontextmanager
    async def get_connection(self):
        """Get a database connection from the pool."""
        if not self.pool:
            await self.initialize_pool()

        async with self.pool.acquire() as conn:
            yield conn

    async def initialize_db(self):
        """Initialize the database schema if it doesn't exist."""
        try:
            async with self.get_connection() as conn:
                # Drop foreign key constraints first
                await conn.execute('''
                ALTER TABLE IF EXISTS jobs DROP CONSTRAINT IF EXISTS fk_jobs_resume_id;
                ALTER TABLE IF EXISTS resumes DROP CONSTRAINT IF EXISTS fk_resumes_job_id;
                ALTER TABLE IF EXISTS search_job_mapping DROP CONSTRAINT IF EXISTS fk_search_job_mapping_search_id;
                ALTER TABLE IF EXISTS search_job_mapping DROP CONSTRAINT IF EXISTS fk_search_job_mapping_job_id;
                ''')

                # Drop existing tables in the correct order (dependencies first)
                await conn.execute('''
                DROP TABLE IF EXISTS search_job_mapping;
                DROP TABLE IF EXISTS search_history;
                DROP TABLE IF EXISTS resumes;
                DROP TABLE IF EXISTS jobs;
                ''')

                # Drop indices if they exist
                await conn.execute('''
                DROP INDEX IF EXISTS idx_jobs_user_id;
                DROP INDEX IF EXISTS idx_resumes_user_id;
                DROP INDEX IF EXISTS idx_search_history_user_id;
                DROP INDEX IF EXISTS idx_search_job_mapping_user_id;
                DROP INDEX IF EXISTS jobs_id_user_id_unique;
                DROP INDEX IF EXISTS resumes_id_user_id_unique;
                DROP INDEX IF EXISTS search_history_id_user_id_unique;
                DROP INDEX IF EXISTS idx_jobs_status;
                DROP INDEX IF EXISTS idx_jobs_company;
                DROP INDEX IF EXISTS idx_jobs_date_found;
                ''')

                # Create jobs table
                await conn.execute('''
                CREATE TABLE jobs (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    job_url TEXT NOT NULL,
                    status TEXT NOT NULL,
                    date_found TEXT,
                    applied_date TEXT,
                    rejected_date TEXT,
                    resume_id TEXT,
                    metadata JSONB
                )
                ''')

                # Add a unique constraint on (id, user_id) in jobs
                await conn.execute('''
                CREATE UNIQUE INDEX jobs_id_user_id_unique ON jobs(id, user_id)
                ''')

                # Create resumes table
                await conn.execute('''
                CREATE TABLE resumes (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    job_id TEXT,
                    file_path TEXT NOT NULL,
                    yaml_content TEXT NOT NULL,
                    date_created TEXT,
                    uploaded_to_simplify BOOLEAN DEFAULT FALSE
                )
                ''')

                # Add a unique constraint on (id, user_id) in resumes
                await conn.execute('''
                CREATE UNIQUE INDEX resumes_id_user_id_unique ON resumes(id, user_id)
                ''')

                # Create search_history table
                await conn.execute('''
                CREATE TABLE search_history (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    keywords TEXT NOT NULL,
                    location TEXT NOT NULL,
                    filters JSONB,
                    date_searched TEXT,
                    job_count INTEGER DEFAULT 0,
                    job_ids JSONB
                )
                ''')

                # Add a unique constraint on (id, user_id) in search_history
                await conn.execute('''
                CREATE UNIQUE INDEX search_history_id_user_id_unique ON search_history(id, user_id)
                ''')

                # Create search_job_mapping table
                await conn.execute('''
                CREATE TABLE search_job_mapping (
                    search_id TEXT,
                    job_id TEXT,
                    user_id TEXT NOT NULL,
                    PRIMARY KEY (search_id, job_id)
                )
                ''')

                # Create indexes for user_id on all tables
                await conn.execute('''
                CREATE INDEX idx_jobs_user_id ON jobs(user_id);
                CREATE INDEX idx_resumes_user_id ON resumes(user_id);
                CREATE INDEX idx_search_history_user_id ON search_history(user_id);
                CREATE INDEX idx_search_job_mapping_user_id ON search_job_mapping(user_id);
                ''')

                # Create index for common queries
                await conn.execute('''
                CREATE INDEX idx_jobs_status ON jobs(status);
                CREATE INDEX idx_jobs_company ON jobs(company);
                CREATE INDEX idx_jobs_date_found ON jobs(date_found);
                ''')

                # Now add the foreign key constraints
                await conn.execute('''
                ALTER TABLE jobs
                ADD CONSTRAINT fk_jobs_resume_id
                FOREIGN KEY (resume_id) REFERENCES resumes (id);
                ''')

                await conn.execute('''
                ALTER TABLE resumes
                ADD CONSTRAINT fk_resumes_job_id
                FOREIGN KEY (job_id) REFERENCES jobs (id);
                ''')

                await conn.execute('''
                ALTER TABLE search_job_mapping
                ADD CONSTRAINT fk_search_job_mapping_search_id
                FOREIGN KEY (search_id) REFERENCES search_history (id);
                ''')

                await conn.execute('''
                ALTER TABLE search_job_mapping
                ADD CONSTRAINT fk_search_job_mapping_job_id
                FOREIGN KEY (job_id) REFERENCES jobs (id);
                ''')

                logger.info("Database schema initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing database: {e}")
            raise

    # Job methods

    async def save_job(self, job: Job, user_id: str) -> bool:
        """
        Save a job to the database.

        Args:
            job: Job object to save
            user_id: ID of the user owning this job

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            async with self.get_connection() as conn:
                # Convert job object to dict for storage
                job_dict = job.to_dict()

                # Check if job already exists for this user
                existing = await conn.fetchval("SELECT id FROM jobs WHERE id = $1 AND user_id = $2", job_dict["id"], user_id)

                if existing:
                    # Update existing job
                    await conn.execute('''
                    UPDATE jobs
                    SET job_url = $1, status = $2, date_found = $3, applied_date = $4,
                        rejected_date = $5, resume_id = $6, metadata = $7
                    WHERE id = $8 AND user_id = $9
                    ''',
                                       job_dict["job_url"], job_dict["status"],
                                       job_dict["date_found"], job_dict["applied_date"], job_dict["rejected_date"],
                                       job_dict["resume_id"], json.dumps(job_dict.get("metadata", {})), job_dict["id"], user_id
                                       )
                    logger.info(f"Updated job: {job_dict['id']} for user: {user_id}")
                else:
                    # Insert new job
                    await conn.execute('''
                    INSERT INTO jobs (id, user_id, job_url, status, date_found, applied_date,
                                    rejected_date, resume_id, metadata)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                    ''',
                                       job_dict["id"], user_id, job_dict["job_url"], job_dict["status"],
                                       job_dict["date_found"], job_dict["applied_date"], job_dict["rejected_date"],
                                       job_dict["resume_id"], json.dumps(job_dict.get("metadata", {}))
                                       )
                    logger.info(f"Inserted new job: {job_dict['id']} for user: {user_id}")

                return True
        except Exception as e:
            logger.error(f"Error saving job: {e}")
            return False

    async def get_job(self, job_id: str, user_id: str) -> Optional[Job]:
        """
        Get a job by ID.

        Args:
            job_id: ID of the job to retrieve
            user_id: ID of the user who owns the job

        Returns:
            Job object if found, None otherwise
        """
        try:
            async with self.get_connection() as conn:
                row = await conn.fetchrow("SELECT * FROM jobs WHERE id = $1 AND user_id = $2", job_id, user_id)

                if row:
                    # Convert row to dict
                    job_dict = dict(row)

                    # Parse metadata JSON
                    if job_dict.get("metadata"):
                        job_dict["metadata"] = json.loads(job_dict["metadata"])
                    else:
                        job_dict["metadata"] = {}

                    # Extract only the fields needed for the new Job class
                    filtered_dict = {
                        "id": job_dict["id"],
                        "job_url": job_dict["job_url"],
                        "status": job_dict["status"],
                        "date_found": job_dict["date_found"],
                        "applied_date": job_dict["applied_date"],
                        "rejected_date": job_dict["rejected_date"],
                        "resume_id": job_dict.get("resume_id"),
                        "metadata": job_dict["metadata"]
                    }

                    return Job.from_dict(filtered_dict)
                else:
                    return None
        except Exception as e:
            logger.error(f"Error getting job {job_id} for user {user_id}: {e}")
            return None

    async def get_all_jobs(self, user_id: str, status: Optional[Union[JobStatus, str]] = None) -> List[Job]:
        """
        Get all jobs for a user, optionally filtered by status.

        Args:
            user_id: ID of the user who owns the jobs
            status: Filter jobs by this status

        Returns:
            List of Job objects
        """
        try:
            async with self.get_connection() as conn:
                if status:
                    # Convert enum to string if needed
                    status_str = str(status) if isinstance(status, JobStatus) else status
                    rows = await conn.fetch("SELECT * FROM jobs WHERE user_id = $1 AND status = $2", user_id, status_str)
                else:
                    rows = await conn.fetch("SELECT * FROM jobs WHERE user_id = $1", user_id)

                jobs = []
                for row in rows:
                    # Convert row to dict
                    job_dict = dict(row)

                    # Parse metadata JSON
                    if job_dict.get("metadata"):
                        job_dict["metadata"] = json.loads(job_dict["metadata"])
                    else:
                        job_dict["metadata"] = {}

                    # Extract only the fields needed for the new Job class
                    filtered_dict = {
                        "id": job_dict["id"],
                        "job_url": job_dict["job_url"],
                        "status": job_dict["status"],
                        "date_found": job_dict["date_found"],
                        "applied_date": job_dict["applied_date"],
                        "rejected_date": job_dict["rejected_date"],
                        "resume_id": job_dict.get("resume_id"),
                        "metadata": job_dict["metadata"]
                    }

                    jobs.append(Job.from_dict(filtered_dict))

                return jobs
        except Exception as e:
            logger.error(f"Error getting jobs for user {user_id}: {e}")
            return []

    async def job_exists(self, url: str, user_id: str) -> Optional[str]:
        """
        Check if a job with the given URL already exists in the database for this user.

        Args:
            url: URL of the job posting
            user_id: ID of the user who owns the job

        Returns:
            Job ID if it exists, None otherwise
        """
        try:
            async with self.get_connection() as conn:
                # Updated to use job_url instead of linkedin_url
                job_id = await conn.fetchval("SELECT id FROM jobs WHERE job_url = $1 AND user_id = $2", url, user_id)
                return job_id
        except Exception as e:
            logger.error(f"Error checking if job exists for user {user_id}: {e}")
            return None

    async def update_job_status(self, job_id: str, user_id: str, status: Union[JobStatus, str]) -> bool:
        """
        Update the status of a job.

        Args:
            job_id: ID of the job to update
            user_id: ID of the user who owns the job
            status: New status for the job

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            async with self.get_connection() as conn:
                # Convert enum to string if needed
                status_str = str(status) if isinstance(status, JobStatus) else status

                # Update status
                await conn.execute('''
                UPDATE jobs
                SET status = $1
                WHERE id = $2 AND user_id = $3
                ''', status_str, job_id, user_id)

                # Update appropriate date field based on the status
                if status_str == str(JobStatus.APPLIED):
                    await conn.execute('''
                    UPDATE jobs
                    SET applied_date = $1
                    WHERE id = $2 AND user_id = $3
                    ''', datetime.now().isoformat(), job_id, user_id)
                elif status_str == str(JobStatus.REJECTED):
                    await conn.execute('''
                    UPDATE jobs
                    SET rejected_date = $1
                    WHERE id = $2 AND user_id = $3
                    ''', datetime.now().isoformat(), job_id, user_id)

                logger.info(f"Updated job {job_id} status to {status_str} for user {user_id}")
                return True
        except Exception as e:
            logger.error(f"Error updating job status: {e}")
            return False

    # Resume methods

    async def save_resume(self, resume: Resume, user_id: str) -> bool:
        """
        Save a resume to the database.

        Args:
            resume: Resume object to save
            user_id: ID of the user who owns the resume

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            async with self.get_connection() as conn:
                # Convert resume object to dict for storage
                resume_dict = resume.to_dict()

                # Check if resume already exists for this user
                existing = await conn.fetchval("SELECT id FROM resumes WHERE id = $1 AND user_id = $2", resume_dict["id"], user_id)

                if existing:
                    # Update existing resume
                    await conn.execute('''
                    UPDATE resumes
                    SET job_id = $1, file_path = $2, yaml_content = $3,
                        date_created = $4, uploaded_to_simplify = $5
                    WHERE id = $6 AND user_id = $7
                    ''',
                                       resume_dict["job_id"], resume_dict["file_path"],
                                       resume_dict["yaml_content"], resume_dict["date_created"],
                                       resume_dict["uploaded_to_simplify"], resume_dict["id"], user_id
                                       )
                    logger.info(f"Updated resume: {resume_dict['id']} for user: {user_id}")
                else:
                    # Insert new resume
                    await conn.execute('''
                    INSERT INTO resumes (id, user_id, job_id, file_path, yaml_content,
                                        date_created, uploaded_to_simplify)
                    VALUES ($1, $2, $3, $4, $5, $6, $7)
                    ''',
                                       resume_dict["id"], user_id, resume_dict["job_id"], resume_dict["file_path"],
                                       resume_dict["yaml_content"], resume_dict["date_created"],
                                       resume_dict["uploaded_to_simplify"]
                                       )
                    logger.info(f"Inserted new resume: {resume_dict['id']} for user: {user_id}")

                return True
        except Exception as e:
            logger.error(f"Error saving resume: {e}")
            return False

    async def get_resume(self, resume_id: str, user_id: str) -> Optional[Resume]:
        """
        Get a resume by ID.

        Args:
            resume_id: ID of the resume to retrieve
            user_id: ID of the user who owns the resume

        Returns:
            Resume object if found, None otherwise
        """
        try:
            async with self.get_connection() as conn:
                row = await conn.fetchrow("SELECT * FROM resumes WHERE id = $1 AND user_id = $2", resume_id, user_id)

                if row:
                    # Convert row to dict
                    resume_dict = dict(row)
                    return Resume.from_dict(resume_dict)
                else:
                    return None
        except Exception as e:
            logger.error(f"Error getting resume {resume_id} for user {user_id}: {e}")
            return None

    async def get_resume_for_job(self, job_id: str, user_id: str) -> Optional[Resume]:
        """
        Get a resume associated with a specific job.

        Args:
            job_id: ID of the job
            user_id: ID of the user who owns the job and resume

        Returns:
            Resume object if found, None otherwise
        """
        try:
            async with self.get_connection() as conn:
                row = await conn.fetchrow("SELECT * FROM resumes WHERE job_id = $1 AND user_id = $2", job_id, user_id)

                if row:
                    # Convert row to dict
                    resume_dict = dict(row)
                    return Resume.from_dict(resume_dict)
                else:
                    return None
        except Exception as e:
            logger.error(f"Error getting resume for job {job_id} for user {user_id}: {e}")
            return None

    async def update_simplify_upload_status(self, resume_id: str, user_id: str, uploaded: bool) -> bool:
        """
        Update the Simplify upload status of a resume.

        Args:
            resume_id: ID of the resume to update
            user_id: ID of the user who owns the resume
            uploaded: Whether the resume has been uploaded to Simplify

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            async with self.get_connection() as conn:
                await conn.execute('''
                UPDATE resumes
                SET uploaded_to_simplify = $1
                WHERE id = $2 AND user_id = $3
                ''', uploaded, resume_id, user_id)

                logger.info(f"Updated resume {resume_id} Simplify upload status to {uploaded} for user {user_id}")
                return True
        except Exception as e:
            logger.error(f"Error updating resume Simplify upload status: {e}")
            return False

    # Search history methods

    async def get_cached_search_results(self, keywords: str, location: str, filters: Dict[str, Any], user_id: str) -> List[Dict[str, Any]]:
        """
        Get cached search results for given search parameters from the database.

        Args:
            keywords: Search keywords
            location: Search location
            filters: Search filters
            user_id: ID of the user who performed the search

        Returns:
            List of job dictionaries if found in cache, empty list otherwise
        """
        try:
            async with self.get_connection() as conn:
                # Generate search ID to find the cached search
                search_id = self._generate_search_key(keywords, location, filters)

                # First check if we have this search in our history for this user
                search_row = await conn.fetchrow('''
                SELECT id, job_count, date_searched 
                FROM search_history 
                WHERE id = $1 AND user_id = $2
                ''', search_id, user_id)

                if not search_row:
                    logger.debug(f"No cached search found for '{keywords}' in '{location}' for user {user_id}")
                    return []

                # Check if the search is recent (less than 24 hours old)
                date_searched = datetime.fromisoformat(search_row["date_searched"])
                cache_age = datetime.now() - date_searched

                # If cache is older than 24 hours, consider it stale
                if cache_age.total_seconds() > 86400:  # 24 hours in seconds
                    logger.info(f"Cached search results for '{keywords}' in '{location}' for user {user_id} are stale (older than 24 hours)")
                    return []

                # Get the job IDs associated with this search
                job_ids = []

                # First try to get job_ids directly from search_history
                job_ids_json = await conn.fetchval('''
                SELECT job_ids FROM search_history WHERE id = $1 AND user_id = $2
                ''', search_id, user_id)

                if job_ids_json:
                    try:
                        job_ids = json.loads(job_ids_json)
                    except json.JSONDecodeError:
                        logger.warning(f"Failed to parse job_ids JSON from search_history: {job_ids_json}")

                # If job_ids not found in search_history, try search_job_mapping
                if not job_ids:
                    try:
                        mapping_rows = await conn.fetch('''
                        SELECT job_id FROM search_job_mapping WHERE search_id = $1 AND user_id = $2
                        ''', search_id, user_id)

                        if mapping_rows:
                            job_ids = [row["job_id"] for row in mapping_rows]
                    except Exception as e:
                        logger.debug(f"Could not get job_ids from search_job_mapping: {e}")

                # If we couldn't find any job IDs, return empty list
                if not job_ids:
                    logger.warning(f"Search ID {search_id} for user {user_id} exists but has no associated job IDs")
                    return []

                # Now fetch the actual job data for these job IDs
                results = []
                for job_id in job_ids:
                    job = await self.get_job(job_id, user_id)
                    if job:
                        results.append(job.to_dict())

                logger.info(f"Retrieved {len(results)} cached jobs for search '{keywords}' in '{location}' for user {user_id}")
                return results
        except Exception as e:
            logger.error(f"Error retrieving cached search results: {e}")
            return []

    async def save_search_history(self, keywords: str, location: str, filters: Dict[str, Any], job_ids: List[str], user_id: str, search_id: str = None) -> bool:
        """
        Save a search history entry to the database.

        Args:
            keywords: Search keywords
            location: Search location
            filters: Filters used in the search
            job_ids: List of job IDs from the search results
            user_id: ID of the user who performed the search
            search_id: Optional custom search ID

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            async with self.get_connection() as conn:
                # Generate search ID if not provided
                if not search_id:
                    search_id = self._generate_search_key(keywords, location, filters)

                # Get current timestamp
                date_searched = datetime.now().isoformat()

                # Check if this search already exists for this user
                existing = await conn.fetchval("SELECT id FROM search_history WHERE id = $1 AND user_id = $2", search_id, user_id)

                if existing:
                    # Update existing search history
                    await conn.execute('''
                    UPDATE search_history
                    SET keywords = $1, location = $2, filters = $3,
                        date_searched = $4, job_count = $5, job_ids = $6
                    WHERE id = $7 AND user_id = $8
                    ''',
                                       keywords, location, json.dumps(filters),
                                       date_searched, len(job_ids), json.dumps(job_ids), search_id, user_id
                                       )
                    logger.info(f"Updated search history: {search_id} for user {user_id}")
                else:
                    # Insert new search history
                    await conn.execute('''
                    INSERT INTO search_history (id, user_id, keywords, location, filters,
                                              date_searched, job_count, job_ids)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                    ''',
                                       search_id, user_id, keywords, location, json.dumps(filters),
                                       date_searched, len(job_ids), json.dumps(job_ids)
                                       )
                    logger.info(f"Inserted new search history: {search_id} for user {user_id}")

                # Create mapping between search and jobs in search_job_mapping table
                try:
                    # Clear existing mappings
                    await conn.execute("DELETE FROM search_job_mapping WHERE search_id = $1 AND user_id = $2", search_id, user_id)

                    # Insert new mappings
                    for job_id in job_ids:
                        await conn.execute('''
                        INSERT INTO search_job_mapping (search_id, job_id, user_id)
                        VALUES ($1, $2, $3)
                        ''', search_id, job_id, user_id)

                except Exception as e:
                    logger.debug(f"Could not update search_job_mapping: {e}")

                return True
        except Exception as e:
            logger.error(f"Error saving search history: {e}")
            return False

    async def get_search_history(self, user_id: str, limit: int = 10) -> List[SearchHistory]:
        """
        Get recent search history entries for a user.

        Args:
            user_id: ID of the user
            limit: Maximum number of entries to retrieve

        Returns:
            List of SearchHistory objects
        """
        try:
            async with self.get_connection() as conn:
                rows = await conn.fetch('''
                SELECT * FROM search_history
                WHERE user_id = $1
                ORDER BY date_searched DESC
                LIMIT $2
                ''', user_id, limit)

                history_entries = []
                for row in rows:
                    # Convert row to dict
                    history_dict = dict(row)

                    # Parse filters JSON
                    if history_dict.get("filters"):
                        history_dict["filters"] = json.loads(history_dict["filters"])
                    else:
                        history_dict["filters"] = {}

                    # Parse job_ids JSON if present
                    if history_dict.get("job_ids"):
                        history_dict["job_ids"] = json.loads(history_dict["job_ids"])

                    history_entries.append(SearchHistory.from_dict(history_dict))

                return history_entries
        except Exception as e:
            logger.error(f"Error getting search history for user {user_id}: {e}")
            return []

    # Utility methods

    def _generate_search_key(self, keywords: str, location: str, filters: Dict[str, Any]) -> str:
        """Generate a hash-based search key for consistent lookup."""
        # Normalize the strings
        keywords_norm = ' '.join(keywords.lower().split())
        location_norm = ' '.join(location.lower().split())

        # Create a stable string representation of filters
        filters_str = json.dumps(filters, sort_keys=True)

        # Create the key string
        key_str = f"{keywords_norm}|{location_norm}|{filters_str}"

        # Hash for consistent ID
        import hashlib
        return hashlib.md5(key_str.encode()).hexdigest()

    def generate_id(self) -> str:
        """Generate a unique ID for database entities."""
        return str(uuid.uuid4())