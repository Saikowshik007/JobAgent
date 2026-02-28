#!/bin/bash

# Database Initialization Fix Script for JobTrak
# This script will help you fix the database initialization issue

echo "🔧 JobTrak Database Initialization Fix"
echo "======================================"

# Function to check if PostgreSQL is ready
check_postgres() {
    echo "🔍 Checking PostgreSQL connection..."
    docker exec infra-postgres pg_isready -U postgres -d jobtrak
    return $?
}

# Function to run SQL initialization
init_database() {
    echo "🚀 Initializing database schema..."

    # Copy the init script to the container if it doesn't exist
    if [ -f "./init-scripts/01.sql" ]; then
        echo "📄 Found init script, executing..."
        docker exec -i infra-postgres psql -U postgres -d jobtrak < ./init-scripts/01.sql
    else
        echo "❌ Init script not found at ./init-scripts/01.sql"
        echo "Creating the script now..."

        # Create the init-scripts directory if it doesn't exist
        mkdir -p ./init-scripts

        # Create the initialization script
        cat > ./init-scripts/01.sql << 'EOF'
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
EOF

        echo "✅ Created init script"

        # Now execute it
        echo "🚀 Executing initialization script..."
        docker exec -i jobtrak-postgres psql -U jobtrak_user -d jobtrak < ./init-scripts/01.sql
    fi
}

# Function to verify tables were created
verify_tables() {
    echo "🔍 Verifying tables were created..."

    # Check if tables exist
    TABLES=$(docker exec jobtrak-postgres psql -U jobtrak_user -d jobtrak -t -c "SELECT tablename FROM pg_tables WHERE schemaname = 'public';")

    if echo "$TABLES" | grep -q "jobs"; then
        echo "✅ jobs table created"
    else
        echo "❌ jobs table missing"
        return 1
    fi

    if echo "$TABLES" | grep -q "resumes"; then
        echo "✅ resumes table created"
    else
        echo "❌ resumes table missing"
        return 1
    fi

    if echo "$TABLES" | grep -q "search_history"; then
        echo "✅ search_history table created"
    else
        echo "❌ search_history table missing"
        return 1
    fi

    # Check row counts
    echo "📊 Table statistics:"
    docker exec jobtrak-postgres psql -U jobtrak_user -d jobtrak -c "
        SELECT
            'jobs' as table_name,
            count(*) as row_count
        FROM jobs
        UNION ALL
        SELECT
            'resumes' as table_name,
            count(*) as row_count
        FROM resumes
        UNION ALL
        SELECT
            'search_history' as table_name,
            count(*) as row_count
        FROM search_history;
    "

    return 0
}

# Function to restart the API service
restart_api() {
    echo "🔄 Restarting JobTrak API to clear any cached errors..."
    docker restart jobtrak-api
    echo "✅ API restarted"
}

# Main execution
main() {
    echo "Starting database initialization fix..."

    # Check if PostgreSQL container is running
    if ! docker ps | grep -q "jobtrak-postgres"; then
        echo "❌ PostgreSQL container is not running!"
        echo "Please start your Docker Compose stack first:"
        echo "docker-compose up -d"
        exit 1
    fi

    # Wait for PostgreSQL to be ready
    echo "⏳ Waiting for PostgreSQL to be ready..."
    RETRY_COUNT=0
    MAX_RETRIES=30

    while ! check_postgres; do
        RETRY_COUNT=$((RETRY_COUNT + 1))
        if [ $RETRY_COUNT -ge $MAX_RETRIES ]; then
            echo "❌ PostgreSQL failed to become ready after $MAX_RETRIES attempts"
            exit 1
        fi
        echo "⏳ Waiting for PostgreSQL... (attempt $RETRY_COUNT/$MAX_RETRIES)"
        sleep 2
    done

    echo "✅ PostgreSQL is ready!"

    # Initialize the database
    if init_database; then
        echo "✅ Database initialization completed"
    else
        echo "❌ Database initialization failed"
        exit 1
    fi

    # Verify tables were created
    if verify_tables; then
        echo "✅ Database verification passed"
    else
        echo "❌ Database verification failed"
        exit 1
    fi

    # Restart API service
    restart_api

    echo ""
    echo "🎉 Database initialization completed successfully!"
    echo ""
    echo "Next steps:"
    echo "1. Check your application logs: docker logs jobtrak-api"
    echo "2. Test your API endpoints"
    echo "3. Access pgAdmin at http://localhost:8082 (admin@jobtrak.local / admin123)"
    echo ""
    echo "If you still see issues, run this command to check the database:"
    echo "docker exec -it jobtrak-postgres psql -U jobtrak_user -d jobtrak -c '\\dt'"
}

# Run the main function
main "$@"