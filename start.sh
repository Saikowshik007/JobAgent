#!/bin/bash
# JobTrak Launcher - POSIX Compliant & Integrated with Shared Infra

set -e

# Commands
DOCKER_CMD="docker"
COMPOSE_CMD="docker compose"

print_status() { echo -e "\033[0;34m[INFO]\033[0m $1"; }
print_success() { echo -e "\033[0;32m[SUCCESS]\033[0m $1"; }
print_error() { echo -e "\033[0;31m[ERROR]\033[0m $1"; }

# 1. Check if Shared Infra is actually running
if ! $DOCKER_CMD ps | grep -q "infra-postgres"; then
    print_error "Shared infrastructure (infra-postgres) is not running!"
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
    # POSIX compliant increment
    counter=$((counter + 1))
done

if [ $counter -eq $timeout ]; then
    print_error "API failed to start within $timeout seconds."
    exit 1
fi

print_status "Access via: https://jobtrackai.duckdns.org (Ensure Caddy is reloaded)"