-- Create utility functions for JobTrak database
\c jobtrak;

-- Set application name for connection identification
SET application_name = 'jobtrak_functions';

-- Function to get job statistics for a user (optimized single query)
CREATE OR REPLACE FUNCTION get_job_stats(p_user_id TEXT)
RETURNS TABLE(
    status TEXT,
    count BIGINT,
    total BIGINT
) AS $$
BEGIN
RETURN QUERY
    WITH status_counts AS (
        SELECT
            j.status,
            COUNT(*) as status_count
        FROM jobs j
        WHERE j.user_id = p_user_id
        GROUP BY j.status
    ),
    total_count AS (
        SELECT COUNT(*) as total_jobs
        FROM jobs j
        WHERE j.user_id = p_user_id
    )
SELECT
    sc.status,
    sc.status_count,
    tc.total_jobs
FROM status_counts sc
         CROSS JOIN total_count tc

UNION ALL

SELECT
    'total'::TEXT,
        tc.total_jobs,
    tc.total_jobs
FROM total_count tc;
END;
$$ LANGUAGE plpgsql;

-- Function to clean old search history (keep last 30 days)
CREATE OR REPLACE FUNCTION cleanup_old_searches()
RETURNS INTEGER AS $$
DECLARE
deleted_count INTEGER;
BEGIN
DELETE FROM search_history
WHERE date_searched < NOW() - INTERVAL '30 days';

GET DIAGNOSTICS deleted_count = ROW_COUNT;

RETURN deleted_count;
END;
$$ LANGUAGE plpgsql;

-- Function to update job status with automatic date setting
CREATE OR REPLACE FUNCTION update_job_status(
    p_job_id TEXT,
    p_user_id TEXT,
    p_status TEXT
)
RETURNS BOOLEAN AS $$
BEGIN
UPDATE jobs
SET
    status = p_status,
    applied_date = CASE
                       WHEN p_status = 'APPLIED' THEN COALESCE(applied_date, NOW())
                       ELSE applied_date
        END,
    rejected_date = CASE
                        WHEN p_status = 'REJECTED' THEN COALESCE(rejected_date, NOW())
                        ELSE rejected_date
        END,
    updated_at = NOW()
WHERE id = p_job_id AND user_id = p_user_id;

RETURN FOUND;
END;
$$ LANGUAGE plpgsql;

-- Function to get recent job activity for a user
CREATE OR REPLACE FUNCTION get_recent_activity(
    p_user_id TEXT,
    p_days INTEGER DEFAULT 7
)
RETURNS TABLE(
    activity_date DATE,
    jobs_found INTEGER,
    jobs_applied INTEGER,
    jobs_rejected INTEGER
) AS $$
BEGIN
RETURN QUERY
    WITH date_series AS (
        SELECT generate_series(
            CURRENT_DATE - INTERVAL '1 day' * p_days,
            CURRENT_DATE,
            INTERVAL '1 day'
        )::DATE as activity_date
    )
SELECT
    ds.activity_date,
    COALESCE(found.count, 0)::INTEGER as jobs_found,
        COALESCE(applied.count, 0)::INTEGER as jobs_applied,
        COALESCE(rejected.count, 0)::INTEGER as jobs_rejected
FROM date_series ds
         LEFT JOIN (
    SELECT
        date_found::DATE as activity_date,
            COUNT(*) as count
    FROM jobs
    WHERE user_id = p_user_id
      AND date_found >= CURRENT_DATE - INTERVAL '1 day' * p_days
    GROUP BY date_found::DATE
) found ON ds.activity_date = found.activity_date
    LEFT JOIN (
    SELECT
    applied_date::DATE as activity_date,
    COUNT(*) as count
    FROM jobs
    WHERE user_id = p_user_id
    AND applied_date >= CURRENT_DATE - INTERVAL '1 day' * p_days
    AND applied_date IS NOT NULL
    GROUP BY applied_date::DATE
    ) applied ON ds.activity_date = applied.activity_date
    LEFT JOIN (
    SELECT
    rejected_date::DATE as activity_date,
    COUNT(*) as count
    FROM jobs
    WHERE user_id = p_user_id
    AND rejected_date >= CURRENT_DATE - INTERVAL '1 day' * p_days
    AND rejected_date IS NOT NULL
    GROUP BY rejected_date::DATE
    ) rejected ON ds.activity_date = rejected.activity_date
ORDER BY ds.activity_date DESC;
END;
$$ LANGUAGE plpgsql;

-- Function to find duplicate jobs by URL
CREATE OR REPLACE FUNCTION find_duplicate_jobs(p_user_id TEXT)
RETURNS TABLE(
    job_url TEXT,
    job_ids TEXT[],
    count BIGINT
) AS $$
BEGIN
RETURN QUERY
SELECT
    j.job_url,
    array_agg(j.id ORDER BY j.date_found DESC) as job_ids,
    COUNT(*) as count
FROM jobs j
WHERE j.user_id = p_user_id
GROUP BY j.job_url
HAVING COUNT(*) > 1
ORDER BY COUNT(*) DESC;
END;
$$ LANGUAGE plpgsql;

-- Function to get job search suggestions based on history
CREATE OR REPLACE FUNCTION get_search_suggestions(p_user_id TEXT, p_limit INTEGER DEFAULT 10)
RETURNS TABLE(
    keywords TEXT,
    location TEXT,
    search_count BIGINT,
    last_searched TIMESTAMPTZ,
    avg_results INTEGER
) AS $$
BEGIN
RETURN QUERY
SELECT
    sh.keywords,
    sh.location,
    COUNT(*) as search_count,
    MAX(sh.date_searched) as last_searched,
    AVG(sh.job_count)::INTEGER as avg_results
FROM search_history sh
WHERE sh.user_id = p_user_id
  AND sh.date_searched >= NOW() - INTERVAL '30 days'
GROUP BY sh.keywords, sh.location
HAVING COUNT(*) > 1
ORDER BY COUNT(*) DESC, MAX(sh.date_searched) DESC
    LIMIT p_limit;
END;
$$ LANGUAGE plpgsql;

-- Function to calculate job application rate
CREATE OR REPLACE FUNCTION get_application_rate(p_user_id TEXT)
RETURNS TABLE(
    period TEXT,
    jobs_found INTEGER,
    jobs_applied INTEGER,
    application_rate NUMERIC
) AS $$
BEGIN
RETURN QUERY
    WITH periods AS (
        SELECT 'Last 7 days' as period, 7 as days
        UNION ALL
        SELECT 'Last 30 days', 30
        UNION ALL
        SELECT 'Last 90 days', 90
    )
SELECT
    p.period,
    COUNT(j.id)::INTEGER as jobs_found,
        COUNT(j.id) FILTER (WHERE j.status = 'APPLIED')::INTEGER as jobs_applied,
        CASE
            WHEN COUNT(j.id) > 0 THEN
                ROUND((COUNT(j.id) FILTER (WHERE j.status = 'APPLIED')::NUMERIC / COUNT(j.id)::NUMERIC) * 100, 2)
            ELSE 0
            END as application_rate
FROM periods p
         LEFT JOIN jobs j ON j.user_id = p_user_id
    AND j.date_found >= NOW() - INTERVAL '1 day' * p.days
GROUP BY p.period, p.days
ORDER BY p.days;
END;
$$ LANGUAGE plpgsql;

-- Function to cleanup orphaned data
CREATE OR REPLACE FUNCTION cleanup_orphaned_data()
RETURNS TABLE(
    table_name TEXT,
    deleted_count INTEGER
) AS $$
DECLARE
resumes_deleted INTEGER;
BEGIN
    -- Clean up resumes that reference non-existent jobs
DELETE FROM resumes r
WHERE r.job_id IS NOT NULL
  AND NOT EXISTS (
    SELECT 1 FROM jobs j
    WHERE j.id = r.job_id AND j.user_id = r.user_id
);

GET DIAGNOSTICS resumes_deleted = ROW_COUNT;

RETURN QUERY VALUES ('resumes', resumes_deleted);
END;
$$ LANGUAGE plpgsql;

-- Create a maintenance function to be run periodically
CREATE OR REPLACE FUNCTION run_maintenance()
RETURNS TABLE(
    task TEXT,
    result TEXT
) AS $$
DECLARE
searches_deleted INTEGER;
    orphans_cleaned INTEGER;
BEGIN
    -- Clean old search history
SELECT cleanup_old_searches() INTO searches_deleted;

-- Clean orphaned data
SELECT deleted_count FROM cleanup_orphaned_data() INTO orphans_cleaned;

-- Update table statistics
ANALYZE jobs;
    ANALYZE resumes;
    ANALYZE search_history;

RETURN QUERY VALUES
        ('old_searches_deleted', searches_deleted::TEXT),
        ('orphaned_resumes_cleaned', orphans_cleaned::TEXT),
        ('statistics_updated', 'completed');
END;
$$ LANGUAGE plpgsql;

-- Grant execute permissions to application user
GRANT EXECUTE ON ALL FUNCTIONS IN SCHEMA public TO jobtrak_user;