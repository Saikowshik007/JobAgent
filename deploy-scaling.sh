#!/bin/bash

# Simple JobTrak API Scaling Deployment Script
# This script scales your API from 1 to 3 instances with load balancing

set -e

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if Docker is running
check_docker() {
    if ! docker info > /dev/null 2>&1; then
        print_error "Docker is not running. Please start Docker first."
        exit 1
    fi
    print_status "Docker is running ✓"
}

# Backup current setup
backup_current() {
    print_status "Creating backup of current setup..."

    BACKUP_DIR="./backup_$(date +%Y%m%d_%H%M%S)"
    mkdir -p "$BACKUP_DIR"

    # Backup current docker-compose file
    if [ -f "docker-compose.yml" ]; then
        cp docker-compose.yml "$BACKUP_DIR/"
        print_status "Backed up docker-compose.yml"
    fi

    # Backup database if running
    if docker ps | grep -q postgres; then
        print_status "Creating database backup..."
        docker exec jobtrak-postgres pg_dump -U jobtrak_user jobtrak > "$BACKUP_DIR/database_backup.sql" 2>/dev/null || print_warning "Database backup failed"
    fi

    echo "$BACKUP_DIR" > .last_backup
    print_status "Backup completed in $BACKUP_DIR"
}

# Deploy the scaled setup
deploy_scaled() {
    print_status "Stopping current services..."
    docker-compose down --timeout 30 2>/dev/null || true

    print_status "Starting scaled services..."

    # Start database and Redis first
    docker-compose up -d postgres redis selenium-chrome

    print_status "Waiting for database to be ready..."
    sleep 15

    # Wait for PostgreSQL to be ready
    for i in {1..12}; do
        if docker exec jobtrak-postgres pg_isready -U jobtrak_user -d jobtrak > /dev/null 2>&1; then
            print_status "PostgreSQL is ready ✓"
            break
        fi
        if [ $i -eq 12 ]; then
            print_error "PostgreSQL failed to start after 2 minutes"
            exit 1
        fi
        sleep 10
    done

    # Start all API instances and Traefik
    print_status "Starting API instances and load balancer..."
    docker-compose up -d

    print_status "Waiting for services to be ready..."
    sleep 20
}

# Verify the deployment
verify_deployment() {
    print_status "Verifying deployment..."

    # Check if all containers are running
    services=("jobtrak-api-1" "jobtrak-api-2" "jobtrak-api-3" "jobtrak-postgres" "jobtrak-redis" "jobtrak-traefik")

    for service in "${services[@]}"; do
        if docker ps --format "{{.Names}}" | grep -q "^${service}$"; then
            print_status "$service is running ✓"
        else
            print_error "$service is not running ✗"
        fi
    done

    # Test load balancer
    print_status "Testing load balancer..."
    sleep 10

    # Check if Traefik dashboard is accessible
    if curl -s -f http://localhost:8080 > /dev/null; then
        print_status "Traefik dashboard accessible ✓"
    else
        print_warning "Traefik dashboard not accessible"
    fi

    print_status "Deployment verification completed!"
}

# Show status
show_status() {
    echo ""
    echo "=== JobTrak Scaling Status ==="
    echo ""

    # Show running containers
    echo "Running Containers:"
    docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}" | grep jobtrak
    echo ""

    # Show service URLs
    echo "Service URLs:"
    echo "• Application: https://jobtrackai.duckdns.org"
    echo "• Traefik Dashboard: http://localhost:8080"
    echo ""

    # Show quick resource usage
    echo "Resource Usage:"
    docker stats --no-stream --format "table {{.Container}}\t{{.CPUPerc}}\t{{.MemUsage}}" | grep jobtrak
    echo ""

    print_status "Your application is now running with 3 API instances!"
    print_status "Monitor with: docker-compose logs -f"
}

# Rollback function
rollback() {
    if [ -f ".last_backup" ]; then
        BACKUP_DIR=$(cat .last_backup)
        if [ -f "$BACKUP_DIR/docker-compose.yml" ]; then
            print_warning "Rolling back to previous setup..."
            docker-compose down --timeout 30
            cp "$BACKUP_DIR/docker-compose.yml" .
            docker-compose up -d
            print_status "Rollback completed"
        else
            print_error "Backup file not found"
        fi
    else
        print_error "No backup reference found"
    fi
}

# Main function
main() {
    echo -e "${GREEN}"
    echo "╔══════════════════════════════════════╗"
    echo "║        JobTrak API Scaling           ║"
    echo "║    1 → 3 Instances + Load Balancer  ║"
    echo "╚══════════════════════════════════════╝"
    echo -e "${NC}"

    case "${1:-deploy}" in
        "deploy")
            check_docker
            backup_current
            deploy_scaled
            verify_deployment
            show_status
            ;;
        "status")
            show_status
            ;;
        "rollback")
            rollback
            ;;
        "logs")
            docker-compose logs -f "${2:-}"
            ;;
        *)
            echo "Usage: $0 [deploy|status|rollback|logs]"
            echo ""
            echo "Commands:"
            echo "  deploy   - Scale API to 3 instances (default)"
            echo "  status   - Show current status"
            echo "  rollback - Rollback to previous setup"
            echo "  logs     - Show logs (optional: specify service)"
            exit 1
            ;;
    esac
}

# Run main function
main "$@"