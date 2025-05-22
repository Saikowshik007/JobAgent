import json
import logging
import os
from typing import Dict, List, Optional, Any, Union, Tuple
import asyncpg
from contextlib import asynccontextmanager
import asyncio

from dataModels.data_models import Job, Resume, JobStatus

# Configure logging
logger = logging.getLogger(__name__)

class Database:
    """PostgreSQL database layer with connection pooling and batch operations."""

    def __init__(self, db_url=None, min_pool_size=2, max_pool_size=20):
        """
        Initialize the database connection with optimized pool settings.

        Args:
            db_url: PostgreSQL connection string
            min_pool_size: Minimum connections in pool
            max_pool_size: Maximum connections in pool
        """
        self.db_url = db_url or os.environ.get('DATABASE_URL')
        if not self.db_url:
            raise ValueError("No database URL provided")

        self.min_pool_size = min_pool_size
        self.max_pool_size = max_pool_size
        self.pool = None
        self._pool_lock = asyncio.Lock()

        # Prepared statement cache
        self._prepared_statements = {}

    async def initialize_pool(self):
        """Initialize connection pool with optimized settings."""
        async with self._pool_lock:
            if self.pool is None:
                self.pool = await asyncpg.create_pool(
                    dsn=self.db_url,
                    min_size=self.min_pool_size,
                    max_size=self.max_pool_size,
                    command_timeout=30,
                    server_settings={
                        'jit': 'off',  # Disable JIT for better performance on small queries
                        'application_name': 'jobtrak_api'
                    }
                )
                logger.info(f"PostgreSQL connection pool initialized (min={self.min_pool_size}, max={self.max_pool_size})")

    async def close_pool(self):
        """Close connection pool."""
        async with self._pool_lock:
            if self.pool:
                await self.pool.close()
                self.pool = None
                self._prepared_statements.clear()
                logger.info("PostgreSQL connection pool closed")

    @asynccontextmanager
    async def get_connection(self):
        """Get a database connection from the pool."""
        if not self.pool:
            await self.initialize_pool()

        async with self.pool.acquire() as conn:
            yield conn

    async def initialize_db(self):
        """Initialize the database schema with optimized indexes."""
        try:
            async with self.get_connection() as conn:
                # Drop existing schema
                await self._drop_existing_schema(conn)

                # Create optimized tables
                await self._create_optimized_tables(conn)

                # Create optimized indexes
                await self._create_optimized_indexes(conn)

                logger.info("Optimized database schema initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing database: {e}")
            raise

    async def _drop_existing_schema(self, conn):
        """Drop existing schema elements."""
        await conn.execute('''
        DROP TABLE IF EXISTS search_job_mapping CASCADE;
        DROP TABLE IF EXISTS search_history CASCADE;
        DROP TABLE IF EXISTS resumes CASCADE;
        DROP TABLE IF EXISTS jobs CASCADE;
        DROP INDEX IF EXISTS idx_jobs_user_status;
        DROP INDEX IF EXISTS idx_jobs_user_url;
        DROP INDEX IF EXISTS idx_jobs_user_date;
        DROP INDEX IF EXISTS idx_resumes_user_job;
        DROP INDEX IF EXISTS idx_search_user_date;
        ''')

    async def _create_optimized_tables(self, conn):
        """Create tables with optimized structure."""
        # Jobs table with better column types
        await conn.execute('''
        CREATE TABLE jobs (
            id TEXT NOT NULL,
            user_id TEXT NOT NULL,
            job_url TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'NEW',
            date_found TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            applied_date TIMESTAMPTZ,
            rejected_date TIMESTAMPTZ,
            resume_id TEXT,
            metadata JSONB DEFAULT '{}',
            created_at TIMESTAMPTZ DEFAULT NOW(),
            updated_at TIMESTAMPTZ DEFAULT NOW(),
            PRIMARY KEY (id, user_id)
        )
        ''')

        # Resumes table with better structure
        await conn.execute('''
        CREATE TABLE resumes (
            id TEXT NOT NULL,
            user_id TEXT NOT NULL,
            job_id TEXT,
            file_path TEXT NOT NULL,
            yaml_content TEXT NOT NULL,
            date_created TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            uploaded_to_simplify BOOLEAN DEFAULT FALSE,
            created_at TIMESTAMPTZ DEFAULT NOW(),
            updated_at TIMESTAMPTZ DEFAULT NOW(),
            PRIMARY KEY (id, user_id)
        )
        ''')

        # Search history with better performance
        await conn.execute('''
        CREATE TABLE search_history (
            id TEXT NOT NULL,
            user_id TEXT NOT NULL,
            keywords TEXT NOT NULL,
            location TEXT NOT NULL,
            filters JSONB DEFAULT '{}',
            date_searched TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            job_count INTEGER DEFAULT 0,
            job_ids JSONB DEFAULT '[]',
            created_at TIMESTAMPTZ DEFAULT NOW(),
            PRIMARY KEY (id, user_id)
        )
        ''')

    async def _create_optimized_indexes(self, conn):
        """Create optimized indexes for better query performance."""
        # Composite indexes for common query patterns
        await conn.execute('''
        CREATE INDEX CONCURRENTLY idx_jobs_user_status_date ON jobs(user_id, status, date_found DESC);
        CREATE INDEX CONCURRENTLY idx_jobs_user_url_hash ON jobs(user_id, md5(job_url));
        CREATE INDEX CONCURRENTLY idx_jobs_metadata_gin ON jobs USING GIN(metadata);
        CREATE INDEX CONCURRENTLY idx_resumes_user_job ON resumes(user_id, job_id);
        CREATE INDEX CONCURRENTLY idx_search_user_date ON search_history(user_id, date_searched DESC);
        CREATE INDEX CONCURRENTLY idx_search_filters_gin ON search_history USING GIN(filters);
        ''')

        # Unique constraints
        await conn.execute('''
        CREATE UNIQUE INDEX CONCURRENTLY idx_jobs_user_url_unique ON jobs(user_id, job_url);
        ''')

    # Optimized Job Methods

    async def save_job_batch(self, jobs: List[Tuple[Job, str]]) -> bool:
        """
        Save multiple jobs in a single transaction for better performance.

        Args:
            jobs: List of (Job, user_id) tuples

        Returns:
            bool: True if successful
        """
        if not jobs:
            return True

        try:
            async with self.get_connection() as conn:
                async with conn.transaction():
                    # Prepare bulk insert data
                    job_data = []
                    for job, user_id in jobs:
                        job_dict = job.to_dict()
                        job_data.append((
                            job_dict["id"], user_id, job_dict["job_url"],
                            job_dict["status"], job_dict["date_found"],
                            job_dict["applied_date"], job_dict["rejected_date"],
                            job_dict["resume_id"], json.dumps(job_dict.get("metadata", {}))
                        ))

                    # Use COPY for bulk insert (much faster than individual INSERTs)
                    await conn.copy_records_to_table(
                        'jobs',
                        records=job_data,
                        columns=['id', 'user_id', 'job_url', 'status', 'date_found',
                                 'applied_date', 'rejected_date', 'resume_id', 'metadata'],
                        timeout=30
                    )

                    logger.info(f"Bulk inserted {len(jobs)} jobs")
                    return True
        except Exception as e:
            logger.error(f"Error in bulk job save: {e}")
            return False

    async def save_job(self, job: Job, user_id: str) -> bool:
        """Optimized single job save with upsert."""
        try:
            async with self.get_connection() as conn:
                job_dict = job.to_dict()

                # Use UPSERT for better performance
                await conn.execute('''
                INSERT INTO jobs (id, user_id, job_url, status, date_found, applied_date,
                                rejected_date, resume_id, metadata, updated_at)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, NOW())
                ON CONFLICT (id, user_id) DO UPDATE SET
                    job_url = EXCLUDED.job_url,
                    status = EXCLUDED.status,
                    applied_date = EXCLUDED.applied_date,
                    rejected_date = EXCLUDED.rejected_date,
                    resume_id = EXCLUDED.resume_id,
                    metadata = EXCLUDED.metadata,
                    updated_at = NOW()
                ''',
                                   job_dict["id"], user_id, job_dict["job_url"], job_dict["status"],
                                   job_dict["date_found"], job_dict["applied_date"], job_dict["rejected_date"],
                                   job_dict["resume_id"], json.dumps(job_dict.get("metadata", {}))
                                   )
                return True
        except Exception as e:
            logger.error(f"Error saving job: {e}")
            return False

    async def get_job(self, job_id: str, user_id: str) -> Optional[Job]:
        """Optimized job retrieval."""
        try:
            async with self.get_connection() as conn:
                row = await conn.fetchrow(
                    "SELECT * FROM jobs WHERE id = $1 AND user_id = $2",
                    job_id, user_id
                )

                if row:
                    return self._row_to_job(row)
                return None
        except Exception as e:
            logger.error(f"Error getting job {job_id} for user {user_id}: {e}")
            return None

    async def get_jobs_batch(self, job_ids: List[str], user_id: str) -> List[Job]:
        """Get multiple jobs in a single query."""
        if not job_ids:
            return []

        try:
            async with self.get_connection() as conn:
                rows = await conn.fetch(
                    "SELECT * FROM jobs WHERE id = ANY($1) AND user_id = $2",
                    job_ids, user_id
                )

                return [self._row_to_job(row) for row in rows]
        except Exception as e:
            logger.error(f"Error getting batch jobs for user {user_id}: {e}")
            return []

    async def get_all_jobs(self, user_id: str, status: Optional[Union[JobStatus, str]] = None,
                           limit: int = None, offset: int = 0) -> List[Job]:
        """Optimized job listing with pagination."""
        try:
            async with self.get_connection() as conn:
                if status:
                    status_str = str(status) if isinstance(status, JobStatus) else status
                    query = '''
                    SELECT * FROM jobs 
                    WHERE user_id = $1 AND status = $2 
                    ORDER BY date_found DESC
                    '''
                    params = [user_id, status_str]
                else:
                    query = '''
                    SELECT * FROM jobs 
                    WHERE user_id = $1 
                    ORDER BY date_found DESC
                    '''
                    params = [user_id]

                if limit:
                    query += f" LIMIT ${len(params) + 1} OFFSET ${len(params) + 2}"
                    params.extend([limit, offset])

                rows = await conn.fetch(query, *params)
                return [self._row_to_job(row) for row in rows]
        except Exception as e:
            logger.error(f"Error getting jobs for user {user_id}: {e}")
            return []

    async def get_job_stats(self, user_id: str) -> Dict[str, int]:
        """Get job statistics in a single optimized query."""
        try:
            async with self.get_connection() as conn:
                # Single query to get all stats
                rows = await conn.fetch('''
                SELECT status, COUNT(*) as count
                FROM jobs 
                WHERE user_id = $1 
                GROUP BY status
                ''', user_id)

                stats = {status.value: 0 for status in JobStatus}
                total = 0

                for row in rows:
                    stats[row['status']] = row['count']
                    total += row['count']

                stats['total'] = total
                return stats
        except Exception as e:
            logger.error(f"Error getting job stats for user {user_id}: {e}")
            return {'total': 0}

    async def job_exists(self, url: str, user_id: str) -> Optional[str]:
        """Optimized job existence check using index."""
        try:
            async with self.get_connection() as conn:
                job_id = await conn.fetchval(
                    "SELECT id FROM jobs WHERE user_id = $1 AND job_url = $2",
                    user_id, url
                )
                return job_id
        except Exception as e:
            logger.error(f"Error checking if job exists for user {user_id}: {e}")
            return None

    async def update_job_status_batch(self, updates: List[Tuple[str, str, str]]) -> bool:
        """
        Update multiple job statuses in a single transaction.

        Args:
            updates: List of (job_id, user_id, status) tuples
        """
        if not updates:
            return True

        try:
            async with self.get_connection() as conn:
                async with conn.transaction():
                    for job_id, user_id, status in updates:
                        await conn.execute('''
                        UPDATE jobs SET status = $3, updated_at = NOW()
                        WHERE id = $1 AND user_id = $2
                        ''', job_id, user_id, status)

                        # Update date fields based on status
                        if status == str(JobStatus.APPLIED):
                            await conn.execute('''
                            UPDATE jobs SET applied_date = NOW()
                            WHERE id = $1 AND user_id = $2
                            ''', job_id, user_id)
                        elif status == str(JobStatus.REJECTED):
                            await conn.execute('''
                            UPDATE jobs SET rejected_date = NOW()
                            WHERE id = $1 AND user_id = $2
                            ''', job_id, user_id)

                    logger.info(f"Updated {len(updates)} job statuses")
                    return True
        except Exception as e:
            logger.error(f"Error in batch status update: {e}")
            return False

    # Optimized Resume Methods

    async def save_resume(self, resume: Resume, user_id: str) -> bool:
        """Optimized resume save with upsert."""
        try:
            async with self.get_connection() as conn:
                resume_dict = resume.to_dict()

                await conn.execute('''
                INSERT INTO resumes (id, user_id, job_id, file_path, yaml_content,
                                   date_created, uploaded_to_simplify, updated_at)
                VALUES ($1, $2, $3, $4, $5, $6, $7, NOW())
                ON CONFLICT (id, user_id) DO UPDATE SET
                    job_id = EXCLUDED.job_id,
                    file_path = EXCLUDED.file_path,
                    yaml_content = EXCLUDED.yaml_content,
                    uploaded_to_simplify = EXCLUDED.uploaded_to_simplify,
                    updated_at = NOW()
                ''',
                                   resume_dict["id"], user_id, resume_dict["job_id"],
                                   resume_dict["file_path"], resume_dict["yaml_content"],
                                   resume_dict["date_created"], resume_dict["uploaded_to_simplify"]
                                   )
                return True
        except Exception as e:
            logger.error(f"Error saving resume: {e}")
            return False

    async def get_resume(self, resume_id: str, user_id: str) -> Optional[Resume]:
        """Get resume by ID."""
        try:
            async with self.get_connection() as conn:
                row = await conn.fetchrow(
                    "SELECT * FROM resumes WHERE id = $1 AND user_id = $2",
                    resume_id, user_id
                )

                if row:
                    return self._row_to_resume(row)
                return None
        except Exception as e:
            logger.error(f"Error getting resume {resume_id} for user {user_id}: {e}")
            return None

    # Optimized Search Methods

    async def save_search_history(self, keywords: str, location: str, filters: Dict[str, Any],
                                  job_ids: List[str], user_id: str, search_id: str = None) -> bool:
        """Optimized search history save."""
        try:
            async with self.get_connection() as conn:
                if not search_id:
                    search_id = self._generate_search_key(keywords, location, filters)

                await conn.execute('''
                INSERT INTO search_history (id, user_id, keywords, location, filters,
                                          job_count, job_ids, updated_at)
                VALUES ($1, $2, $3, $4, $5, $6, $7, NOW())
                ON CONFLICT (id, user_id) DO UPDATE SET
                    keywords = EXCLUDED.keywords,
                    location = EXCLUDED.location,
                    filters = EXCLUDED.filters,
                    job_count = EXCLUDED.job_count,
                    job_ids = EXCLUDED.job_ids,
                    date_searched = NOW(),
                    updated_at = NOW()
                ''',
                                   search_id, user_id, keywords, location, json.dumps(filters),
                                   len(job_ids), json.dumps(job_ids)
                                   )
                return True
        except Exception as e:
            logger.error(f"Error saving search history: {e}")
            return False

    async def get_cached_search_results(self, keywords: str, location: str,
                                        filters: Dict[str, Any], user_id: str) -> List[Dict[str, Any]]:
        """Optimized cached search retrieval."""
        try:
            async with self.get_connection() as conn:
                search_id = self._generate_search_key(keywords, location, filters)

                # Get search with job data in single query
                row = await conn.fetchrow('''
                SELECT sh.job_ids, sh.date_searched
                FROM search_history sh
                WHERE sh.id = $1 AND sh.user_id = $2
                AND sh.date_searched > NOW() - INTERVAL '24 hours'
                ''', search_id, user_id)

                if not row or not row['job_ids']:
                    return []

                job_ids = json.loads(row['job_ids'])
                if not job_ids:
                    return []

                # Get all jobs in single query
                jobs = await self.get_jobs_batch(job_ids, user_id)
                return [job.to_dict() for job in jobs]

        except Exception as e:
            logger.error(f"Error retrieving cached search results: {e}")
            return []

    # Helper Methods

    def _row_to_job(self, row) -> Job:
        """Convert database row to Job object."""
        job_dict = dict(row)

        # Parse metadata
        if job_dict.get("metadata"):
            if isinstance(job_dict["metadata"], str):
                job_dict["metadata"] = json.loads(job_dict["metadata"])
        else:
            job_dict["metadata"] = {}
        return Job.from_dict(job_dict)

    def _row_to_resume(self, row) -> Resume:
        """Convert database row to Resume object."""
        resume_dict = dict(row)

        return Resume.from_dict(resume_dict)

    def _generate_search_key(self, keywords: str, location: str, filters: Dict[str, Any]) -> str:
        """Generate search key with better normalization."""
        import hashlib

        # Normalize strings
        keywords_norm = ' '.join(keywords.lower().split())
        location_norm = ' '.join(location.lower().split())
        filters_str = json.dumps(filters, sort_keys=True)

        # Create key
        key_str = f"{keywords_norm}|{location_norm}|{filters_str}"
        return hashlib.md5(key_str.encode()).hexdigest()

    async def health_check(self) -> Dict[str, Any]:
        """Database health check."""
        try:
            async with self.get_connection() as conn:
                result = await conn.fetchval("SELECT 1")
                pool_stats = {
                    "size": self.pool.get_size(),
                    "idle": self.pool.get_idle_size(),
                    "min_size": self.pool.get_min_size(),
                    "max_size": self.pool.get_max_size()
                } if self.pool else {}

                return {
                    "status": "healthy" if result == 1 else "unhealthy",
                    "pool": pool_stats
                }
        except Exception as e:
            return {"status": "unhealthy", "error": str(e)}