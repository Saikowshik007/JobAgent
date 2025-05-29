-- Initialize JobTrak database based on your application schema
\c jobtrak;

-- Enable useful extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_stat_statements";

-- Set application name for connection identification
SET application_name = 'jobtrak_init';

-- Drop existing tables if they exist (for clean initialization)
DROP TABLE IF EXISTS search_history CASCADE;
DROP TABLE IF EXISTS resumes CASCADE;
DROP TABLE IF EXISTS jobs CASCADE;

-- Create optimized jobs table (based on your Database class)
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

-- Create optimized resumes table
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

-- Create optimized search history table
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

-- Create updated_at trigger function
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $
BEGIN
    NEW.updated_at = NOW();
RETURN NEW;
END;
$ language 'plpgsql';

-- Create triggers for automatic updated_at
CREATE TRIGGER update_jobs_updated_at
    BEFORE UPDATE ON jobs
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_resumes_updated_at
    BEFORE UPDATE ON resumes
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_search_history_updated_at
    BEFORE UPDATE ON search_history
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Job status enum values (for reference)
-- NEW, APPLIED, REJECTED, INTERVIEW, OFFER, ACCEPTED, DECLINED

-- Note: Indexes are created separately in 02-create-indexes.sql
-- to avoid blocking operations during initialization