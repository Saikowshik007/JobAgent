# Multi-stage Dockerfile for JobTrak API Service
FROM python:3.10-slim as base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Create app user for security
RUN groupadd -r appuser && useradd -r -g appuser appuser

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    gnupg \
    unzip \
    wget \
    git \
    build-essential \
    redis-tools \
    postgresql-client \
    netcat-openbsd \
    procps \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Create necessary directories with proper permissions
RUN mkdir -p /app/database /app/output /app/logs /app/tmp \
    && chown -R appuser:appuser /app

# Copy source code
COPY --chown=appuser:appuser . .

# Create health check script
COPY --chown=appuser:appuser --chmod=755 <<'EOF' /usr/local/bin/healthcheck.sh
#!/bin/bash
set -e

# Function to check service health
check_service() {
    local service_name="$1"
    local check_command="$2"
    local timeout="${3:-30}"
    local counter=0

    echo "Checking $service_name..."

    while [ $counter -lt $timeout ]; do
        if eval "$check_command" >/dev/null 2>&1; then
            echo "✓ $service_name: OK"
            return 0
        fi
        counter=$((counter + 1))
        sleep 1
    done

    echo "✗ $service_name: FAILED (timeout after ${timeout}s)"
    return 1
}

# Check Redis connectivity
check_service "Redis" "redis-cli -h \${REDIS_HOST:-redis} -p \${REDIS_PORT:-6379} ping" 10

# Check PostgreSQL connectivity
check_service "PostgreSQL" "pg_isready -h \${POSTGRES_HOST:-postgres} -p \${POSTGRES_PORT:-5432} -U \${POSTGRES_USER:-jobtrak_user} -d \${POSTGRES_DB:-jobtrak}" 10

# Check FastAPI application
check_service "FastAPI" "curl -f http://localhost:8000/health" 5

echo "All health checks passed!"
exit 0
EOF

# Create startup script
COPY --chown=appuser:appuser --chmod=755 <<'EOF' /usr/local/bin/start.sh
#!/bin/bash
set -e

echo "🚀 Starting JobTrak API Service"
echo "==============================="

# Wait for dependencies
echo "⏳ Waiting for dependencies..."

# Wait for PostgreSQL
echo "Waiting for PostgreSQL..."
timeout=60
counter=0
while ! pg_isready -h ${POSTGRES_HOST:-postgres} -p ${POSTGRES_PORT:-5432} -U ${POSTGRES_USER:-jobtrak_user} -d ${POSTGRES_DB:-jobtrak} >/dev/null 2>&1; do
    counter=$((counter + 1))
    if [ $counter -ge $timeout ]; then
        echo "❌ PostgreSQL connection timeout after ${timeout}s"
        exit 1
    fi
    sleep 1
done
echo "✅ PostgreSQL is ready"

# Wait for Redis
echo "Waiting for Redis..."
counter=0
while ! redis-cli -h ${REDIS_HOST:-redis} -p ${REDIS_PORT:-6379} ping >/dev/null 2>&1; do
    counter=$((counter + 1))
    if [ $counter -ge 30 ]; then
        echo "❌ Redis connection timeout after 30s"
        exit 1
    fi
    sleep 1
done
echo "✅ Redis is ready"

# Run database migrations if needed
if [ -f "/app/migrations/run_migrations.py" ]; then
    echo "🔄 Running database migrations..."
    python /app/migrations/run_migrations.py || echo "⚠️ Migration warnings (continuing...)"
fi

# Set up environment
export PYTHONPATH=/app
cd /app

# Start the application
echo "🌟 Starting FastAPI application..."
if [ "${API_DEBUG:-0}" = "1" ]; then
    echo "🐛 Debug mode enabled"
    exec python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload --log-level debug
else
    exec python -m uvicorn main:app --host 0.0.0.0 --port 8000 --workers 2 --log-level info
fi
EOF

# Create a simple migration runner (placeholder)
COPY --chown=appuser:appuser <<'EOF' /app/migrations/run_migrations.py
#!/usr/bin/env python3
"""
Simple migration runner for JobTrak
This is a placeholder - replace with your actual migration logic
"""
import os
import asyncio
import asyncpg
from pathlib import Path

async def run_migrations():
    """Run database migrations"""
    try:
        # Database connection
        db_url = os.getenv('DATABASE_URL', 'postgresql://jobtrak_user:jobtrak_secure_password_2024@postgres:5432/jobtrak')

        print("🔄 Connecting to database...")
        conn = await asyncpg.connect(db_url)

        # Check if migrations table exists
        result = await conn.fetchval("""
            SELECT EXISTS (
                SELECT FROM information_schema.tables
                WHERE table_schema = 'public'
                AND table_name = 'schema_migrations'
            );
        """)

        if not result:
            print("📋 Creating migrations table...")
            await conn.execute("""
                CREATE TABLE schema_migrations (
                    version VARCHAR(255) PRIMARY KEY,
                    applied_at TIMESTAMP DEFAULT NOW()
                );
            """)

        # Check current schema version
        try:
            version = await conn.fetchval("SELECT version FROM schema_migrations ORDER BY applied_at DESC LIMIT 1")
            print(f"📊 Current schema version: {version or 'none'}")
        except:
            print("📊 No previous migrations found")

        await conn.close()
        print("✅ Migration check completed")

    except Exception as e:
        print(f"⚠️ Migration check failed: {e}")
        # Don't fail the startup for migration issues
        return

if __name__ == "__main__":
    asyncio.run(run_migrations())
EOF

# Set proper ownership and permissions
RUN chown -R appuser:appuser /app \
    && chmod +x /usr/local/bin/healthcheck.sh \
    && chmod +x /usr/local/bin/start.sh \
    && chmod +x /app/migrations/run_migrations.py

# Switch to non-root user
USER appuser

# Set default environment variables
ENV PYTHONPATH=/app \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    SELENIUM_REMOTE_URL=http://selenium-chrome:4444/wd/hub \
    REDIS_URL=redis://redis:6379/0 \
    REDIS_HOST=redis \
    REDIS_PORT=6379 \
    REDIS_DB=0 \
    CACHE_TTL=3600 \
    POSTGRES_HOST=postgres \
    POSTGRES_PORT=5432 \
    POSTGRES_DB=jobtrak \
    POSTGRES_USER=jobtrak_user \
    DATABASE_URL=postgresql://jobtrak_user:jobtrak_secure_password_2024@postgres:5432/jobtrak

# Add comprehensive health check
HEALTHCHECK --interval=30s --timeout=15s --start-period=60s --retries=3 \
    CMD /usr/local/bin/healthcheck.sh

# Expose ports
EXPOSE 8000 5678

# Use the startup script
CMD ["/usr/local/bin/start.sh"]