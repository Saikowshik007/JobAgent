-- Create optimized indexes for JobTrak database
-- Based on your Database class optimization patterns
\c jobtrak;

-- Set application name for connection identification
SET application_name = 'jobtrak_indexes';

-- Drop existing indexes if they exist (for clean re-creation)
DROP INDEX IF EXISTS idx_jobs_user_status;
DROP INDEX IF EXISTS idx_jobs_user_url;
DROP INDEX IF EXISTS idx_jobs_user_date;
DROP INDEX IF EXISTS idx_resumes_user_job;
DROP INDEX IF EXISTS idx_search_user_date;
DROP INDEX IF EXISTS idx_jobs_user_status_date;
DROP INDEX IF EXISTS idx_jobs_user_url_hash;
DROP INDEX IF EXISTS idx_jobs_metadata_gin;
DROP INDEX IF EXISTS idx_search_filters_gin;
DROP INDEX IF EXISTS idx_jobs_user_url_unique;

-- Create optimized indexes for jobs table
-- Primary query patterns: user_id + status + date_found (for job listings)
CREATE INDEX idx_jobs_user_status_date ON jobs(user_id, status, date_found DESC);

-- For job existence checks: user_id + job_url
CREATE UNIQUE INDEX idx_jobs_user_url_unique ON jobs(user_id, job_url);

-- For URL hash-based lookups (performance optimization)
CREATE INDEX idx_jobs_user_url_hash ON jobs(user_id, md5(job_url));

-- For metadata searches (JSONB GIN index)
CREATE INDEX idx_jobs_metadata_gin ON jobs USING GIN(metadata);

-- For basic user queries
CREATE INDEX idx_jobs_user_id ON jobs(user_id);

-- For status-based queries
CREATE INDEX idx_jobs_status ON jobs(status);

-- For date-based sorting
CREATE INDEX idx_jobs_date_found ON jobs(date_found DESC);

-- Create optimized indexes for resumes table
-- Primary query pattern: user_id + job_id
CREATE INDEX idx_resumes_user_job ON resumes(user_id, job_id);

-- For user's all resumes
CREATE INDEX idx_resumes_user_date ON resumes(user_id, date_created DESC);

-- For job-based resume lookups
CREATE INDEX idx_resumes_job_id ON resumes(job_id);

-- Create optimized indexes for search_history table
-- Primary query pattern: user_id + date_searched (for recent searches)
CREATE INDEX idx_search_user_date ON search_history(user_id, date_searched DESC);

-- For filter-based searches (JSONB GIN index)
CREATE INDEX idx_search_filters_gin ON search_history USING GIN(filters);

-- For search cache lookups by search_id
CREATE INDEX idx_search_id ON search_history(id);

-- For job_ids array searches
CREATE INDEX idx_search_job_ids_gin ON search_history USING GIN(job_ids);

-- Performance optimization indexes for common query patterns

-- Jobs by user and multiple statuses
CREATE INDEX idx_jobs_user_status_in ON jobs(user_id, status) WHERE status IN ('NEW', 'APPLIED', 'INTERVIEW');

-- Recently found jobs (last 30 days)
CREATE INDEX idx_jobs_recent ON jobs(user_id, date_found) WHERE date_found > NOW() - INTERVAL '30 days';

-- Applied jobs with dates
CREATE INDEX idx_jobs_applied_dates ON jobs(user_id, applied_date) WHERE applied_date IS NOT NULL;

-- Rejected jobs with dates
CREATE INDEX idx_jobs_rejected_dates ON jobs(user_id, rejected_date) WHERE rejected_date IS NOT NULL;

-- Resumes uploaded to Simplify
CREATE INDEX idx_resumes_uploaded ON resumes(user_id, uploaded_to_simplify) WHERE uploaded_to_simplify = true;

-- Recent search history (last 7 days for cache)
CREATE INDEX idx_search_recent ON search_history(user_id, id, date_searched)
    WHERE date_searched > NOW() - INTERVAL '7 days';

-- Analyze tables for better query planning
ANALYZE jobs;
ANALYZE resumes;
ANALYZE search_history;

-- Create index usage monitoring view (for performance tuning)
CREATE OR REPLACE VIEW index_usage_stats AS
SELECT
    schemaname,
    tablename,
    indexname,
    idx_scan as times_used,
    idx_tup_read as tuples_read,
    idx_tup_fetch as tuples_fetched
FROM pg_stat_user_indexes
WHERE schemaname = 'public'
  AND tablename IN ('jobs', 'resumes', 'search_history')
ORDER BY idx_scan DESC;

-- Create table size monitoring view
CREATE OR REPLACE VIEW table_sizes AS
SELECT
    tablename,
    pg_size_pretty(pg_total_relation_size(tablename::regclass)) as size,
    pg_total_relation_size(tablename::regclass) as size_bytes
FROM (VALUES ('jobs'), ('resumes'), ('search_history')) AS t(tablename)
ORDER BY size_bytes DESC;