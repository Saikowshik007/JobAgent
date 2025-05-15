import json
import logging
import threading
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
import sqlite3
import uuid

from data_models import Job, Resume, SearchHistory, JobStatus

# Configure logging
logger = logging.getLogger(__name__)

class ThreadLocalConnection:
    """Thread-local storage for SQLite connections."""
    local = threading.local()

    def __init__(self, db_path):
        self.db_path = db_path

    def get_connection(self):
        """Get a connection for the current thread."""
        if not hasattr(self.local, 'connection') or self.local.connection is None:
            self.local.connection = sqlite3.connect(self.db_path)
            # Enable foreign keys support
            self.local.connection.execute("PRAGMA foreign_keys = ON")
            # Return dictionaries instead of tuples for row results
            self.local.connection.row_factory = sqlite3.Row
            logger.debug(f"Created new SQLite connection in thread {threading.get_ident()}")
        return self.local.connection

    def close_connection(self):
        """Close the connection for the current thread."""
        if hasattr(self.local, 'connection') and self.local.connection is not None:
            self.local.connection.close()
            self.local.connection = None
            logger.debug(f"Closed SQLite connection in thread {threading.get_ident()}")

    def close_all_connections(self):
        """Close all connections (best effort)."""
        self.close_connection()  # Close for current thread
        # Note: Cannot access connections from other threads


class Database:
    """Thread-safe database layer for storing job application data using SQLite."""

    def __init__(self, db_path="/app/data/job_tracker.db"):
        """
        Initialize the database connection.

        Args:
            db_path: Path to the SQLite database file
        """
        self.db_path = db_path
        self.connections = ThreadLocalConnection(db_path)
        self.initialize_db()

    def get_connection(self):
        """Get a database connection for the current thread."""
        return self.connections.get_connection()

    def close_connection(self):
        """Close the database connection for the current thread."""
        self.connections.close_connection()

    def initialize_db(self):
        """Initialize the database schema if it doesn't exist."""
        try:
            conn = self.get_connection()
            cursor = conn.cursor()

            # Create jobs table
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS jobs (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                company TEXT NOT NULL,
                location TEXT NOT NULL,
                description TEXT,
                url TEXT NOT NULL,
                status TEXT NOT NULL,
                date_found TEXT,
                applied_date TEXT,
                rejected_date TEXT,
                resume_id TEXT,
                metadata TEXT,
                FOREIGN KEY (resume_id) REFERENCES resumes (id)
            )
            ''')

            # Create resumes table
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS resumes (
                id TEXT PRIMARY KEY,
                job_id TEXT,
                file_path TEXT NOT NULL,
                yaml_content TEXT NOT NULL,
                date_created TEXT,
                uploaded_to_simplify INTEGER DEFAULT 0,
                FOREIGN KEY (job_id) REFERENCES jobs (id)
            )
            ''')

            # Create search_history table
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS search_history (
                id TEXT PRIMARY KEY,
                keywords TEXT NOT NULL,
                location TEXT NOT NULL,
                filters TEXT,
                date_searched TEXT,
                job_count INTEGER DEFAULT 0,
                job_ids TEXT
            )
            ''')

            # Create search_job_mapping table if it doesn't exist
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS search_job_mapping (
                search_id TEXT,
                job_id TEXT,
                PRIMARY KEY (search_id, job_id),
                FOREIGN KEY (search_id) REFERENCES search_history (id),
                FOREIGN KEY (job_id) REFERENCES jobs (id)
            )
            ''')

            # Create index for common queries
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_jobs_company ON jobs(company)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_jobs_date_found ON jobs(date_found)')

            conn.commit()
            logger.info("Database schema initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing database: {e}")
            if conn:
                conn.rollback()
            raise

    # Job methods

    def save_job(self, job: Job) -> bool:
        """
        Save a job to the database.

        Args:
            job: Job object to save

        Returns:
            bool: True if successful, False otherwise
        """
        conn = None
        try:
            conn = self.get_connection()
            cursor = conn.cursor()

            # Convert job object to dict for storage
            job_dict = job.to_dict()

            # Convert metadata to JSON string
            job_dict["metadata"] = json.dumps(job_dict.get("metadata", {}))

            # Check if job already exists (update) or is new (insert)
            cursor.execute("SELECT id FROM jobs WHERE id = ?", (job_dict["id"],))
            if cursor.fetchone():
                # Update existing job
                cursor.execute('''
                UPDATE jobs
                SET title = ?, company = ?, location = ?, description = ?,
                    url = ?, status = ?, date_found = ?, applied_date = ?,
                    rejected_date = ?, resume_id = ?, metadata = ?
                WHERE id = ?
                ''', (
                    job_dict["title"], job_dict["company"], job_dict["location"],
                    job_dict["description"], job_dict["url"], job_dict["status"],
                    job_dict["date_found"], job_dict["applied_date"], job_dict["rejected_date"],
                    job_dict["resume_id"], job_dict["metadata"], job_dict["id"]
                ))
                logger.info(f"Updated job: {job_dict['id']}")
            else:
                # Insert new job
                cursor.execute('''
                INSERT INTO jobs (id, title, company, location, description,
                                 url, status, date_found, applied_date,
                                 rejected_date, resume_id, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    job_dict["id"], job_dict["title"], job_dict["company"],
                    job_dict["location"], job_dict["description"], job_dict["url"],
                    job_dict["status"], job_dict["date_found"], job_dict["applied_date"],
                    job_dict["rejected_date"], job_dict["resume_id"], job_dict["metadata"]
                ))
                logger.info(f"Inserted new job: {job_dict['id']}")

            conn.commit()
            return True
        except Exception as e:
            logger.error(f"Error saving job: {e}")
            if conn:
                conn.rollback()
            return False

    def get_job(self, job_id: str) -> Optional[Job]:
        """
        Get a job by ID.

        Args:
            job_id: ID of the job to retrieve

        Returns:
            Job object if found, None otherwise
        """
        try:
            conn = self.get_connection()
            cursor = conn.cursor()

            cursor.execute("SELECT * FROM jobs WHERE id = ?", (job_id,))
            row = cursor.fetchone()

            if row:
                # Convert row to dict
                job_dict = dict(row)

                # Parse metadata JSON
                if job_dict.get("metadata"):
                    job_dict["metadata"] = json.loads(job_dict["metadata"])
                else:
                    job_dict["metadata"] = {}

                return Job.from_dict(job_dict)
            else:
                return None
        except Exception as e:
            logger.error(f"Error getting job {job_id}: {e}")
            return None

    def get_all_jobs(self, status: Optional[Union[JobStatus, str]] = None) -> List[Job]:
        """
        Get all jobs, optionally filtered by status.

        Args:
            status: Filter jobs by this status

        Returns:
            List of Job objects
        """
        try:
            conn = self.get_connection()
            cursor = conn.cursor()

            if status:
                # Convert enum to string if needed
                status_str = str(status) if isinstance(status, JobStatus) else status
                cursor.execute("SELECT * FROM jobs WHERE status = ?", (status_str,))
            else:
                cursor.execute("SELECT * FROM jobs")

            rows = cursor.fetchall()
            jobs = []

            for row in rows:
                # Convert row to dict
                job_dict = dict(row)

                # Parse metadata JSON
                if job_dict.get("metadata"):
                    job_dict["metadata"] = json.loads(job_dict["metadata"])
                else:
                    job_dict["metadata"] = {}

                jobs.append(Job.from_dict(job_dict))

            return jobs
        except Exception as e:
            logger.error(f"Error getting jobs: {e}")
            return []

    def update_job_status(self, job_id: str, status: Union[JobStatus, str]) -> bool:
        """
        Update the status of a job.

        Args:
            job_id: ID of the job to update
            status: New status for the job

        Returns:
            bool: True if successful, False otherwise
        """
        conn = None
        try:
            conn = self.get_connection()
            cursor = conn.cursor()

            # Convert enum to string if needed
            status_str = str(status) if isinstance(status, JobStatus) else status

            # Update status
            cursor.execute('''
            UPDATE jobs
            SET status = ?
            WHERE id = ?
            ''', (status_str, job_id))

            # Update appropriate date field based on the status
            if status_str == str(JobStatus.APPLIED):
                cursor.execute('''
                UPDATE jobs
                SET applied_date = ?
                WHERE id = ?
                ''', (datetime.now().isoformat(), job_id))
            elif status_str == str(JobStatus.REJECTED):
                cursor.execute('''
                UPDATE jobs
                SET rejected_date = ?
                WHERE id = ?
                ''', (datetime.now().isoformat(), job_id))

            conn.commit()
            logger.info(f"Updated job {job_id} status to {status_str}")
            return True
        except Exception as e:
            logger.error(f"Error updating job status: {e}")
            if conn:
                conn.rollback()
            return False

    def job_exists(self, url: str) -> Optional[str]:
        """
        Check if a job with the given URL already exists in the database.

        Args:
            url: URL of the job posting

        Returns:
            Job ID if it exists, None otherwise
        """
        try:
            conn = self.get_connection()
            cursor = conn.cursor()

            cursor.execute("SELECT id FROM jobs WHERE url = ?", (url,))
            row = cursor.fetchone()

            if row:
                return row["id"]
            else:
                return None
        except Exception as e:
            logger.error(f"Error checking if job exists: {e}")
            return None

    # Resume methods

    def save_resume(self, resume: Resume) -> bool:
        """
        Save a resume to the database.

        Args:
            resume: Resume object to save

        Returns:
            bool: True if successful, False otherwise
        """
        conn = None
        try:
            conn = self.get_connection()
            cursor = conn.cursor()

            # Convert resume object to dict for storage
            resume_dict = resume.to_dict()

            # Convert boolean to integer for SQLite
            resume_dict["uploaded_to_simplify"] = 1 if resume_dict["uploaded_to_simplify"] else 0

            # Check if resume already exists (update) or is new (insert)
            cursor.execute("SELECT id FROM resumes WHERE id = ?", (resume_dict["id"],))
            if cursor.fetchone():
                # Update existing resume
                cursor.execute('''
                UPDATE resumes
                SET job_id = ?, file_path = ?, yaml_content = ?,
                    date_created = ?, uploaded_to_simplify = ?
                WHERE id = ?
                ''', (
                    resume_dict["job_id"], resume_dict["file_path"],
                    resume_dict["yaml_content"], resume_dict["date_created"],
                    resume_dict["uploaded_to_simplify"], resume_dict["id"]
                ))
                logger.info(f"Updated resume: {resume_dict['id']}")
            else:
                # Insert new resume
                cursor.execute('''
                INSERT INTO resumes (id, job_id, file_path, yaml_content,
                                    date_created, uploaded_to_simplify)
                VALUES (?, ?, ?, ?, ?, ?)
                ''', (
                    resume_dict["id"], resume_dict["job_id"], resume_dict["file_path"],
                    resume_dict["yaml_content"], resume_dict["date_created"],
                    resume_dict["uploaded_to_simplify"]
                ))
                logger.info(f"Inserted new resume: {resume_dict['id']}")

            conn.commit()
            return True
        except Exception as e:
            logger.error(f"Error saving resume: {e}")
            if conn:
                conn.rollback()
            return False

    def get_resume(self, resume_id: str) -> Optional[Resume]:
        """
        Get a resume by ID.

        Args:
            resume_id: ID of the resume to retrieve

        Returns:
            Resume object if found, None otherwise
        """
        try:
            conn = self.get_connection()
            cursor = conn.cursor()

            cursor.execute("SELECT * FROM resumes WHERE id = ?", (resume_id,))
            row = cursor.fetchone()

            if row:
                # Convert row to dict
                resume_dict = dict(row)

                # Convert integer to boolean for uploaded_to_simplify
                resume_dict["uploaded_to_simplify"] = bool(resume_dict["uploaded_to_simplify"])

                return Resume.from_dict(resume_dict)
            else:
                return None
        except Exception as e:
            logger.error(f"Error getting resume {resume_id}: {e}")
            return None

    def get_resume_for_job(self, job_id: str) -> Optional[Resume]:
        """
        Get a resume associated with a specific job.

        Args:
            job_id: ID of the job

        Returns:
            Resume object if found, None otherwise
        """
        try:
            conn = self.get_connection()
            cursor = conn.cursor()

            cursor.execute("SELECT * FROM resumes WHERE job_id = ?", (job_id,))
            row = cursor.fetchone()

            if row:
                # Convert row to dict
                resume_dict = dict(row)

                # Convert integer to boolean for uploaded_to_simplify
                resume_dict["uploaded_to_simplify"] = bool(resume_dict["uploaded_to_simplify"])

                return Resume.from_dict(resume_dict)
            else:
                return None
        except Exception as e:
            logger.error(f"Error getting resume for job {job_id}: {e}")
            return None

    def update_simplify_upload_status(self, resume_id: str, uploaded: bool) -> bool:
        """
        Update the Simplify upload status of a resume.

        Args:
            resume_id: ID of the resume to update
            uploaded: Whether the resume has been uploaded to Simplify

        Returns:
            bool: True if successful, False otherwise
        """
        conn = None
        try:
            conn = self.get_connection()
            cursor = conn.cursor()

            # Convert boolean to integer for SQLite
            uploaded_int = 1 if uploaded else 0

            cursor.execute('''
            UPDATE resumes
            SET uploaded_to_simplify = ?
            WHERE id = ?
            ''', (uploaded_int, resume_id))

            conn.commit()
            logger.info(f"Updated resume {resume_id} Simplify upload status to {uploaded}")
            return True
        except Exception as e:
            logger.error(f"Error updating resume Simplify upload status: {e}")
            if conn:
                conn.rollback()
            return False

    # Search history methods

    def get_cached_search_results(self, keywords: str, location: str, filters: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Get cached search results for given search parameters from the database.

        Args:
            keywords: Search keywords
            location: Search location
            filters: Search filters

        Returns:
            List of job dictionaries if found in cache, empty list otherwise
        """
        try:
            conn = self.get_connection()
            cursor = conn.cursor()

            # Generate search ID to find the cached search
            search_id = self._generate_search_key(keywords, location, filters)

            # First check if we have this search in our history
            cursor.execute('''
            SELECT id, job_count, date_searched 
            FROM search_history 
            WHERE id = ?
            ''', (search_id,))

            search_row = cursor.fetchone()
            if not search_row:
                logger.debug(f"No cached search found for '{keywords}' in '{location}'")
                return []

            # Check if the search is recent (less than 24 hours old)
            date_searched = datetime.fromisoformat(search_row["date_searched"])
            cache_age = datetime.now() - date_searched

            # If cache is older than 24 hours, consider it stale
            if cache_age.total_seconds() > 86400:  # 24 hours in seconds
                logger.info(f"Cached search results for '{keywords}' in '{location}' are stale (older than 24 hours)")
                return []

            # Get the job IDs associated with this search
            job_ids = []

            # First try to get job_ids directly from search_history
            cursor.execute('''
            SELECT job_ids FROM search_history WHERE id = ?
            ''', (search_id,))
            row = cursor.fetchone()
            if row and row["job_ids"]:
                try:
                    job_ids = json.loads(row["job_ids"])
                except json.JSONDecodeError:
                    logger.warning(f"Failed to parse job_ids JSON from search_history: {row['job_ids']}")

            # If job_ids not found in search_history, try search_job_mapping
            if not job_ids:
                try:
                    cursor.execute('''
                    SELECT job_id FROM search_job_mapping WHERE search_id = ?
                    ''', (search_id,))
                    mapping_rows = cursor.fetchall()
                    if mapping_rows:
                        job_ids = [row["job_id"] for row in mapping_rows]
                except Exception as e:
                    logger.debug(f"Could not get job_ids from search_job_mapping: {e}")

            # If we couldn't find any job IDs, return empty list
            if not job_ids:
                logger.warning(f"Search ID {search_id} exists but has no associated job IDs")
                return []

            # Now fetch the actual job data for these job IDs
            results = []
            for job_id in job_ids:
                job = self.get_job(job_id)
                if job:
                    results.append(job.to_dict())

            logger.info(f"Retrieved {len(results)} cached jobs for search '{keywords}' in '{location}'")
            return results

        except Exception as e:
            logger.error(f"Error retrieving cached search results: {e}")
            return []

    def save_search_history(self, keywords: str, location: str, filters: Dict[str, Any], job_ids: List[str], search_id: str = None) -> bool:
        """
        Save a search history entry to the database.

        Args:
            keywords: Search keywords
            location: Search location
            filters: Filters used in the search
            job_ids: List of job IDs from the search results
            search_id: Optional custom search ID

        Returns:
            bool: True if successful, False otherwise
        """
        conn = None
        try:
            conn = self.get_connection()
            cursor = conn.cursor()

            # Generate search ID if not provided
            if not search_id:
                search_id = self._generate_search_key(keywords, location, filters)

            # Convert filters to JSON string
            filters_json = json.dumps(filters)

            # Convert job_ids to JSON string
            job_ids_json = json.dumps(job_ids)

            # Get current timestamp
            date_searched = datetime.now().isoformat()

            # Check if this search already exists
            cursor.execute("SELECT id FROM search_history WHERE id = ?", (search_id,))
            existing = cursor.fetchone()

            if existing:
                # Update existing search history
                cursor.execute('''
                UPDATE search_history
                SET keywords = ?, location = ?, filters = ?,
                    date_searched = ?, job_count = ?, job_ids = ?
                WHERE id = ?
                ''', (
                    keywords, location, filters_json,
                    date_searched, len(job_ids), job_ids_json, search_id
                ))
                logger.info(f"Updated search history: {search_id}")
            else:
                # Insert new search history
                cursor.execute('''
                INSERT INTO search_history (id, keywords, location, filters,
                                          date_searched, job_count, job_ids)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (
                    search_id, keywords, location, filters_json,
                    date_searched, len(job_ids), job_ids_json
                ))

                logger.info(f"Inserted new search history: {search_id}")

            # Create mapping between search and jobs in search_job_mapping table
            try:
                # Clear existing mappings
                cursor.execute("DELETE FROM search_job_mapping WHERE search_id = ?", (search_id,))

                # Insert new mappings
                for job_id in job_ids:
                    cursor.execute('''
                    INSERT INTO search_job_mapping (search_id, job_id)
                    VALUES (?, ?)
                    ''', (search_id, job_id))

            except Exception as e:
                logger.debug(f"Could not update search_job_mapping: {e}")

            conn.commit()
            return True
        except Exception as e:
            logger.error(f"Error saving search history: {e}")
            if conn:
                conn.rollback()
            return False

    def get_search_history(self, limit: int = 10) -> List[SearchHistory]:
        """
        Get recent search history entries.

        Args:
            limit: Maximum number of entries to retrieve

        Returns:
            List of SearchHistory objects
        """
        try:
            conn = self.get_connection()
            cursor = conn.cursor()

            cursor.execute('''
            SELECT * FROM search_history
            ORDER BY date_searched DESC
            LIMIT ?
            ''', (limit,))

            rows = cursor.fetchall()
            history_entries = []

            for row in rows:
                # Convert row to dict
                history_dict = dict(row)

                # Parse filters JSON
                if history_dict.get("filters"):
                    history_dict["filters"] = json.loads(history_dict["filters"])
                else:
                    history_dict["filters"] = {}

                history_entries.append(SearchHistory.from_dict(history_dict))

            return history_entries
        except Exception as e:
            logger.error(f"Error getting search history: {e}")
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