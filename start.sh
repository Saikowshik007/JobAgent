#!/bin/bash
# JobTrak Launcher - Integrated with Shared Infra

set -e

# Commands
DOCKER_CMD="docker"
COMPOSE_CMD="docker compose"

print_status() { echo -e "\033[0;34m[INFO]\033[0m $1"; }
print_success() { echo -e "\033[0;32m[SUCCESS]\033[0m $1"; }

# 1. Check if Infra is actually running
if ! $DOCKER_CMD ps | grep -q "infra-postgres"; then
    echo "Error: Shared infrastructure (infra-postgres) is not running!"
    echo "Please run 'docker compose up -d' in your infra folder first."
    exit 1
fi

# 2. Cleanup local orphans
print_status "Cleaning up JobTrak app..."
$COMPOSE_CMD down --remove-orphans 2>/dev/null || true

# 3. Start Application
print_status "Starting JobTrak API..."
$COMPOSE_CMD up --build -d

# 4. Health Check (Wait for API)
print_status "Waiting for API to respond..."
timeout=30
counter=0
while [ $counter -lt $timeout ]; do
    if curl -s http://localhost:8000/health > /dev/null; then
        print_success "🚀 JobTrak API is online!"
        break
    fi
    sleep 2
    ((counter++))
done

print_status "Access via: https://jobtrackai.duckdns.org (Ensure Caddy is reloaded)"