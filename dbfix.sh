#!/bin/bash

# Database Runner for JobTrak (Shared Infra)
echo "🔧 JobTrak Database Runner"
echo "======================================"

# --- CONFIGURATION ---
DB_CONTAINER="infra-postgres"
DB_USER="postgres"
DB_NAME="postgres"
API_CONTAINER="jobtrak-api"
SQL_FILE="./init-scripts/01.sql"

# Function to check if PostgreSQL is ready
check_postgres() {
    docker exec $DB_CONTAINER pg_isready -U $DB_USER -d $DB_NAME &> /dev/null
    return $?
}

main() {
    # 1. Check for infrastructure
    if ! docker ps | grep -q "$DB_CONTAINER"; then
        echo "❌ Error: $DB_CONTAINER is not running."
        exit 1
    fi

    # 2. Check for the SQL file
    if [ ! -f "$SQL_FILE" ]; then
        echo "❌ Error: SQL file not found at $SQL_FILE"
        exit 1
    fi

    # 3. Wait for PostgreSQL to be ready
    echo "⏳ Waiting for PostgreSQL to be ready..."
    RETRY_COUNT=0
    until check_postgres || [ $RETRY_COUNT -eq 15 ]; do
        RETRY_COUNT=$((RETRY_COUNT + 1))
        echo "⏳ Waiting... ($RETRY_COUNT/15)"
        sleep 2
    done

    if [ $RETRY_COUNT -eq 15 ]; then
        echo "❌ Timeout: PostgreSQL is not responding."
        exit 1
    fi

    # 4. Execute your existing SQL file
    echo "🚀 Executing schema from $SQL_FILE..."
    docker exec -i $DB_CONTAINER psql -U $DB_USER -d $DB_NAME < "$SQL_FILE"

    # 5. Quick Verification
    echo "🔍 Verifying tables..."
    TABLES=$(docker exec $DB_CONTAINER psql -U $DB_USER -d $DB_NAME -t -c "SELECT tablename FROM pg_tables WHERE schemaname = 'public';")

    if echo "$TABLES" | grep -q "jobs"; then
        echo "✅ Schema applied successfully."
    else
        echo "❌ Schema check failed. Tables not found."
        exit 1
    fi

    # 6. Restart API
    echo "🔄 Restarting $API_CONTAINER..."
    docker restart $API_CONTAINER
    echo "🎉 JobTrak is ready!"
}

main "$@"