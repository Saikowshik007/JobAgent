-- JobTrak Database Schema Initialization (Matching database.py schema)
-- This file should be placed in ./init-scripts/01-init-database.sql

-- Enable UUID extension (though your code uses TEXT for IDs)
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Drop existing tables and indexes if they exist
DROP TABLE IF EXISTS search_job_mapping CASCADE;
DROP TABLE IF EXISTS search_history CASCADE;
DROP TABLE IF EXISTS resumes CASCADE;
DROP TABLE IF EXISTS jobs CASCADE;

-- Drop indexes if they exist
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

-- Create jobs table (matching your database.py schema exactly)
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
);

-- Create resumes table (matching your database.py schema exactly)
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
);

-- Create search_history table (matching your database.py schema exactly)
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
);

-- Create indexes for better performance (matching what your code expects)
CREATE INDEX IF NOT EXISTS idx_jobs_user_status_date ON jobs(user_id, status, date_found DESC);
CREATE INDEX IF NOT EXISTS idx_jobs_user_url_hash ON jobs(user_id, md5(job_url));
CREATE INDEX IF NOT EXISTS idx_jobs_metadata_gin ON jobs USING GIN(metadata);
CREATE INDEX IF NOT EXISTS idx_resumes_user_job ON resumes(user_id, job_id);
CREATE INDEX IF NOT EXISTS idx_search_user_date ON search_history(user_id, date_searched DESC);
CREATE INDEX IF NOT EXISTS idx_search_filters_gin ON search_history USING GIN(filters);

-- Create unique constraint for job URLs per user
CREATE UNIQUE INDEX IF NOT EXISTS idx_jobs_user_url_unique ON jobs(user_id, job_url);

-- Grant necessary permissions to your database user
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO jobtrak_user;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO jobtrak_user;
GRANT EXECUTE ON ALL FUNCTIONS IN SCHEMA public TO jobtrak_user;

-- Insert some sample data for testing (matching your schema)
INSERT INTO jobs (id, user_id, job_url, status, date_found, metadata) VALUES
                                                                          ('job_001', 'sample_user_123', 'https://example.com/job1', 'NEW', NOW(), '{"title": "Senior Software Engineer", "company": "Tech Corp", "location": "San Francisco, CA"}'),
                                                                          ('job_002', 'sample_user_123', 'https://example.com/job2', 'APPLIED', NOW() - INTERVAL '2 days', '{"title": "Product Manager", "company": "StartupXYZ", "location": "Remote"}'),
                                                                          ('job_003', 'sample_user_123', 'https://example.com/job3', 'INTERVIEW', NOW() - INTERVAL '5 days', '{"title": "DevOps Engineer", "company": "CloudTech", "location": "Seattle, WA"}')
    ON CONFLICT (id, user_id) DO NOTHING;

-- Print success message
DO $$
BEGIN
    RAISE NOTICE 'JobTrak database schema created successfully!';
    RAISE NOTICE 'Tables created: jobs, resumes, search_history';
    RAISE NOTICE 'Schema matches database.py implementation';
    RAISE NOTICE 'Indexes and constraints applied for optimal performance';
END $$;