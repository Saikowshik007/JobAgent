#!/bin/bash
# JobTrak Docker Management Script

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check if Docker is running
check_docker() {
    if ! command -v docker &> /dev/null; then
        print_error "Docker is not installed or not in PATH"
        exit 1
    fi

    if ! docker info &> /dev/null; then
        print_error "Docker daemon is not running"
        exit 1
    fi

    print_success "Docker is running"
}

# Function to check if docker-compose is available
check_compose() {
    if command -v docker-compose &> /dev/null; then
        COMPOSE_CMD="docker-compose"
    elif docker compose version &> /dev/null; then
        COMPOSE_CMD="docker compose"
    else
        print_error "Docker Compose is not available"
        exit 1
    fi

    print_success "Using: $COMPOSE_CMD"
}

# Function to clean up old containers and networks
cleanup() {
    print_status "Cleaning up old containers and networks..."

    # Stop and remove containers
    $COMPOSE_CMD down --remove-orphans 2>/dev/null || true

    # Remove the old data services if they exist
    if [ -f "docker-compose-data.yaml" ]; then
        $COMPOSE_CMD -f docker-compose-data.yaml down --remove-orphans 2>/dev/null || true
    fi

    # Clean up unused networks
    docker network prune -f 2>/dev/null || true

    print_success "Cleanup completed"
}

# Function to start services
start_services() {
    print_status "Starting JobTrak services..."

    # Build and start services
    $COMPOSE_CMD up --build -d

    print_success "Services started"

    # Wait for database to be ready
    print_status "Waiting for database to be ready..."
    timeout=60
    counter=0

    while [ $counter -lt $timeout ]; do
        if docker exec jobtrak-postgres pg_isready -U jobtrak_user -d jobtrak &> /dev/null; then
            print_success "Database is ready"
            break
        fi

        counter=$((counter + 1))
        sleep 1

        if [ $counter -eq $timeout ]; then
            print_error "Database failed to start within $timeout seconds"
            show_logs
            exit 1
        fi
    done

    # Wait for Redis to be ready
    print_status "Waiting for Redis to be ready..."
    counter=0

    while [ $counter -lt $timeout ]; do
        if docker exec jobtrak-redis redis-cli ping &> /dev/null; then
            print_success "Redis is ready"
            break
        fi

        counter=$((counter + 1))
        sleep 1

        if [ $counter -eq $timeout ]; then
            print_error "Redis failed to start within $timeout seconds"
            show_logs
            exit 1
        fi
    done

    # Wait for API to be ready
    print_status "Waiting for API to be ready..."
    counter=0

    while [ $counter -lt $timeout ]; do
        if curl -f http://localhost:8000/health &> /dev/null; then
            print_success "API is ready"
            break
        fi

        counter=$((counter + 1))
        sleep 2

        if [ $counter -eq 30 ]; then  # 30 * 2 = 60 seconds
            print_warning "API is taking longer than expected to start"
            print_status "Checking API logs..."
            show_api_logs
            break
        fi
    done
}

# Function to show logs
show_logs() {
    print_status "Showing recent logs..."
    $COMPOSE_CMD logs --tail=50
}

# Function to show API logs specifically
show_api_logs() {
    print_status "Showing API logs..."
    $COMPOSE_CMD logs --tail=30 jobtrak-api
}

# Function to show service status
show_status() {
    print_status "Service Status:"
    $COMPOSE_CMD ps

    echo ""
    print_status "Health Checks:"

    # Check PostgreSQL
    if docker exec jobtrak-postgres pg_isready -U jobtrak_user -d jobtrak &> /dev/null; then
        print_success "PostgreSQL: Healthy"
    else
        print_error "PostgreSQL: Unhealthy"
    fi

    # Check Redis
    if docker exec jobtrak-redis redis-cli ping &> /dev/null; then
        print_success "Redis: Healthy"
    else
        print_error "Redis: Unhealthy"
    fi

    # Check API
    if curl -f http://localhost:8000/health &> /dev/null; then
        print_success "API: Healthy"
    else
        print_error "API: Unhealthy"
    fi
}

# Function to stop services
stop_services() {
    print_status "Stopping JobTrak services..."
    $COMPOSE_CMD down
    print_success "Services stopped"
}

# Function to restart services
restart_services() {
    print_status "Restarting JobTrak services..."
    stop_services
    start_services
}

# Function to show usage
show_usage() {
    echo "JobTrak Docker Management Script"
    echo ""
    echo "Usage: $0 [COMMAND]"
    echo ""
    echo "Commands:"
    echo "  start     - Start all services"
    echo "  stop      - Stop all services"
    echo "  restart   - Restart all services"
    echo "  status    - Show service status"
    echo "  logs      - Show recent logs"
    echo "  api-logs  - Show API logs"
    echo "  cleanup   - Clean up old containers and networks"
    echo "  help      - Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 start"
    echo "  $0 logs"
    echo "  $0 status"
}

# Main script logic
main() {
    case "${1:-start}" in
        "start")
            check_docker
            check_compose
            cleanup
            start_services
            show_status
            print_success "JobTrak is now running!"
            print_status "API available at: http://localhost:8000"
            print_status "Traefik dashboard at: http://localhost:8080"
            print_status "pgAdmin at: http://localhost:8082 (with --profile tools)"
            ;;
        "stop")
            check_docker
            check_compose
            stop_services
            ;;
        "restart")
            check_docker
            check_compose
            restart_services
            show_status
            ;;
        "status")
            check_docker
            check_compose
            show_status
            ;;
        "logs")
            check_docker
            check_compose
            show_logs
            ;;
        "api-logs")
            check_docker
            check_compose
            show_api_logs
            ;;
        "cleanup")
            check_docker
            check_compose
            cleanup
            ;;
        "help"|"-h"|"--help")
            show_usage
            ;;
        *)
            print_error "Unknown command: $1"
            echo ""
            show_usage
            exit 1
            ;;
    esac
}

# Run main function with all arguments
main "$@"