#!/bin/bash
# Unified JobTrak Docker Management Script
# Handles everything: database initialization, Traefik setup, and service management

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Configuration
COMPOSE_FILE="docker-compose.yml"
PROJECT_NAME="jobtrak"
DOMAIN="jobtrackai.duckdns.org"

# Docker command detection
DOCKER_CMD=""
COMPOSE_CMD=""

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

print_debug() {
    echo -e "${PURPLE}[DEBUG]${NC} $1"
}

print_header() {
    echo -e "${CYAN}"
    echo "=================================================="
    echo "$1"
    echo "=================================================="
    echo -e "${NC}"
}

# Function to find Docker and Docker Compose
setup_docker_commands() {
    # Find Docker
    for cmd in docker /usr/bin/docker /usr/local/bin/docker; do
        if command -v "$cmd" >/dev/null 2>&1; then
            DOCKER_CMD="$cmd"
            break
        fi
    done

    if [ -z "$DOCKER_CMD" ]; then
        print_error "Docker not found. Please install Docker first."
        exit 1
    fi

    # Find Docker Compose
    if $DOCKER_CMD compose version >/dev/null 2>&1; then
        COMPOSE_CMD="$DOCKER_CMD compose"
    elif command -v docker-compose >/dev/null 2>&1; then
        COMPOSE_CMD="docker-compose"
    else
        print_error "Docker Compose not found. Please install Docker Compose."
        exit 1
    fi

    print_success "Docker: $DOCKER_CMD"
    print_success "Docker Compose: $COMPOSE_CMD"
}

# Function to check if Docker daemon is running
check_docker_daemon() {
    if ! $DOCKER_CMD info >/dev/null 2>&1; then
        print_error "Docker daemon is not running"
        print_status "Try: sudo systemctl start docker"
        exit 1
    fi
    print_success "Docker daemon is running"
}

# Function to create necessary directories and files
setup_environment() {
    print_header "Setting up environment"

    # Create directories
    mkdir -p init-scripts letsencrypt logs output database

    # Create .env file if it doesn't exist
    if [ ! -f .env ]; then
        print_status "Creating .env file..."
        cat > .env << 'EOF'
# JobTrak Environment Configuration
COMPOSE_PROJECT_NAME=jobtrak
OPENAI_API_KEY=your_openai_api_key
HOST_IP=auto-detect
EOF
        print_success "Created .env file - please update with your actual API keys"
    fi

    # Create database initialization script
    if [ ! -f "init-scripts/01-init-database.sql" ]; then
        print_status "Creating database initialization script..."
        cat > init-scripts/01-init-database.sql << 'EOF'
-- JobTrak Database Schema Initialization
-- Enable UUID extension
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Drop existing tables if they exist
DROP TABLE IF EXISTS search_job_mapping CASCADE;
DROP TABLE IF EXISTS search_history CASCADE;
DROP TABLE IF EXISTS resumes CASCADE;
DROP TABLE IF EXISTS jobs CASCADE;

-- Create jobs table
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

-- Create resumes table
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

-- Create search_history table
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

-- Create indexes for performance
CREATE INDEX idx_jobs_user_status_date ON jobs(user_id, status, date_found DESC);
CREATE INDEX idx_jobs_user_url_hash ON jobs(user_id, md5(job_url));
CREATE INDEX idx_jobs_metadata_gin ON jobs USING GIN(metadata);
CREATE INDEX idx_resumes_user_job ON resumes(user_id, job_id);
CREATE INDEX idx_search_user_date ON search_history(user_id, date_searched DESC);
CREATE INDEX idx_search_filters_gin ON search_history USING GIN(filters);

-- Create unique constraint for job URLs per user
CREATE UNIQUE INDEX idx_jobs_user_url_unique ON jobs(user_id, job_url);

-- Grant permissions
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO jobtrak_user;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO jobtrak_user;

-- Insert sample data
INSERT INTO jobs (id, user_id, job_url, status, date_found, metadata) VALUES
    ('job_001', 'sample_user_123', 'https://example.com/job1', 'NEW', NOW(), '{"title": "Senior Software Engineer", "company": "Tech Corp", "location": "San Francisco, CA"}'),
    ('job_002', 'sample_user_123', 'https://example.com/job2', 'APPLIED', NOW() - INTERVAL '2 days', '{"title": "Product Manager", "company": "StartupXYZ", "location": "Remote"}'),
    ('job_003', 'sample_user_123', 'https://example.com/job3', 'INTERVIEW', NOW() - INTERVAL '5 days', '{"title": "DevOps Engineer", "company": "CloudTech", "location": "Seattle, WA"}')
ON CONFLICT (id, user_id) DO NOTHING;

-- Success message
SELECT 'JobTrak database initialized successfully!' as status;
EOF
        print_success "Created database initialization script"
    fi

    # Set proper permissions for Let's Encrypt directory
    chmod 755 letsencrypt

    print_success "Environment setup completed"
}

# Function to clean up old containers and networks
cleanup() {
    print_header "Cleaning up old containers and networks"

    # Stop and remove all services
    $COMPOSE_CMD -f $COMPOSE_FILE down --remove-orphans 2>/dev/null || true

    # Clean up old networks
    $DOCKER_CMD network prune -f 2>/dev/null || true

    # Clean up unused volumes (be careful with this)
    # $DOCKER_CMD volume prune -f 2>/dev/null || true

    print_success "Cleanup completed"
}

# Function to start all services
start_services() {
    print_header "Starting JobTrak Services"

    # Start all services
    print_status "Starting all services..."
    $COMPOSE_CMD -f $COMPOSE_FILE up --build -d

    # Wait for services to be ready
    wait_for_services

    print_success "All services started successfully"
}

# Function to wait for services to be ready
wait_for_services() {
    print_status "Waiting for services to be ready..."

    # Wait for PostgreSQL
    print_status "Waiting for PostgreSQL..."
    wait_for_service "postgres" "pg_isready -U jobtrak_user -d jobtrak" 60

    # Wait for Redis
    print_status "Waiting for Redis..."
    wait_for_service "redis" "redis-cli ping" 30

    # Wait for Selenium
    print_status "Waiting for Selenium..."
    wait_for_url "http://localhost:4444/wd/hub/status" 60

    # Wait for API
    print_status "Waiting for API..."
    wait_for_url "http://localhost:8000/health" 60

    print_success "All services are ready!"
}

# Generic function to wait for a service command
wait_for_service() {
    local container_name="jobtrak-$1"
    local command="$2"
    local timeout="$3"
    local counter=0

    while [ $counter -lt $timeout ]; do
        if $DOCKER_CMD exec $container_name $command >/dev/null 2>&1; then
            print_success "$1 is ready"
            return 0
        fi

        counter=$((counter + 1))
        sleep 1

        if [ $((counter % 10)) -eq 0 ]; then
            print_status "Still waiting for $1... (${counter}s/${timeout}s)"
        fi
    done

    print_error "$1 failed to start within $timeout seconds"
    show_service_logs $1
    return 1
}

# Generic function to wait for a URL to be available
wait_for_url() {
    local url="$1"
    local timeout="$2"
    local counter=0

    while [ $counter -lt $timeout ]; do
        if curl -f "$url" >/dev/null 2>&1; then
            return 0
        fi

        counter=$((counter + 1))
        sleep 1

        if [ $((counter % 10)) -eq 0 ]; then
            print_status "Still waiting for $url... (${counter}s/${timeout}s)"
        fi
    done

    print_error "URL $url not available within $timeout seconds"
    return 1
}

# Function to show service logs
show_service_logs() {
    local service="$1"
    print_status "Recent logs for $service:"
    $COMPOSE_CMD -f $COMPOSE_FILE logs --tail=20 "jobtrak-$service" || true
}

# Function to show all logs
show_logs() {
    print_header "Service Logs"
    $COMPOSE_CMD -f $COMPOSE_FILE logs --tail=50
}

# Function to show service status
show_status() {
    print_header "Service Status"

    # Show container status
    $COMPOSE_CMD -f $COMPOSE_FILE ps

    echo ""
    print_status "Health Checks:"

    # Check each service
    check_service_health "postgres" "pg_isready -U jobtrak_user -d jobtrak"
    check_service_health "redis" "redis-cli ping"
    check_url_health "API" "http://localhost:8000/health"
    check_url_health "Selenium" "http://localhost:4444/wd/hub/status"
    check_url_health "Traefik Dashboard" "http://localhost:8080/api/rawdata"

    # Show access URLs
    echo ""
    print_header "Access URLs"
    echo "🌐 API (Local): http://localhost:8000"
    echo "🌐 API (Domain): https://$DOMAIN"
    echo "🚦 Traefik Dashboard: http://localhost:8080"
    echo "🗄️ pgAdmin: http://localhost:8082 (start with --profile tools)"
    echo "🖥️ Selenium VNC: http://localhost:7900 (password: secret)"
    echo ""
    echo "📊 Database Connection:"
    echo "   Host: localhost:5432"
    echo "   Database: jobtrak"
    echo "   User: jobtrak_user"
    echo "   Password: jobtrak_secure_password_2024"
}

# Function to check individual service health
check_service_health() {
    local service_name="$1"
    local command="$2"
    local container_name="jobtrak-$service_name"

    if $DOCKER_CMD exec $container_name $command >/dev/null 2>&1; then
        print_success "$service_name: Healthy"
    else
        print_error "$service_name: Unhealthy"
    fi
}

# Function to check URL health
check_url_health() {
    local service_name="$1"
    local url="$2"

    if curl -f "$url" >/dev/null 2>&1; then
        print_success "$service_name: Healthy"
    else
        print_error "$service_name: Unhealthy"
    fi
}

# Function to stop services
stop_services() {
    print_header "Stopping JobTrak Services"
    $COMPOSE_CMD -f $COMPOSE_FILE down
    print_success "Services stopped"
}

# Function to restart services
restart_services() {
    print_header "Restarting JobTrak Services"
    stop_services
    start_services
}

# Function to update services
update_services() {
    print_header "Updating JobTrak Services"

    # Pull latest images
    print_status "Pulling latest images..."
    $COMPOSE_CMD -f $COMPOSE_FILE pull

    # Rebuild and restart
    print_status "Rebuilding and restarting services..."
    $COMPOSE_CMD -f $COMPOSE_FILE up --build -d

    wait_for_services
    print_success "Services updated successfully"
}

# Function to run database operations
database_operations() {
    print_header "Database Operations"

    case "${2:-status}" in
        "init"|"initialize")
            print_status "Initializing database..."
            if $DOCKER_CMD exec -i jobtrak-postgres psql -U jobtrak_user -d jobtrak < ./init-scripts/01-init-database.sql; then
                print_success "Database initialized successfully"
            else
                print_error "Database initialization failed"
                return 1
            fi
            ;;
        "reset")
            print_warning "This will delete all data. Are you sure? (y/N)"
            read -r confirmation
            if [ "$confirmation" = "y" ] || [ "$confirmation" = "Y" ]; then
                print_status "Resetting database..."
                $DOCKER_CMD exec -i jobtrak-postgres psql -U jobtrak_user -d jobtrak < ./init-scripts/01-init-database.sql
                print_success "Database reset completed"
            else
                print_status "Database reset cancelled"
            fi
            ;;
        "backup")
            local backup_file="backup_$(date +%Y%m%d_%H%M%S).sql"
            print_status "Creating database backup: $backup_file"
            $DOCKER_CMD exec jobtrak-postgres pg_dump -U jobtrak_user jobtrak > "$backup_file"
            print_success "Database backup created: $backup_file"
            ;;
        "status"|*)
            print_status "Database Status:"
            $DOCKER_CMD exec jobtrak-postgres psql -U jobtrak_user -d jobtrak -c "
                SELECT
                    schemaname,
                    tablename,
                    attname,
                    n_distinct,
                    n_tup_ins,
                    n_tup_upd,
                    n_tup_del
                FROM pg_stat_user_tables
                JOIN pg_stats ON pg_stat_user_tables.relname = pg_stats.tablename;
            "
            ;;
    esac
}

# Function to show usage
show_usage() {
    cat << 'EOF'
🚀 JobTrak Unified Docker Management Script
==========================================

Usage: ./start.sh [COMMAND] [OPTIONS]

Commands:
  start          - Start all services (default)
  stop           - Stop all services
  restart        - Restart all services
  status         - Show service status and health
  logs           - Show recent logs from all services
  update         - Pull latest images and restart services
  cleanup        - Clean up old containers and networks

Database Commands:
  db init        - Initialize database schema
  db reset       - Reset database (WARNING: deletes all data)
  db backup      - Create database backup
  db status      - Show database status

Service-specific Commands:
  logs [service] - Show logs for specific service (api, postgres, redis, etc.)
  shell [service]- Open shell in service container

Development Commands:
  dev            - Start with development settings
  tools          - Start with additional tools (pgAdmin)

Options:
  --profile [profile] - Start with specific profile (tools, dev)
  --build            - Force rebuild of images
  --no-cache         - Build without using cache

Examples:
  ./start.sh                    # Start all services
  ./start.sh start --profile tools  # Start with pgAdmin
  ./start.sh logs api           # Show API logs
  ./start.sh db init            # Initialize database
  ./start.sh status             # Show service status

Access Points:
  🌐 API: http://localhost:8000 or https://jobtrackai.duckdns.org
  🚦 Traefik: http://localhost:8080
  🗄️ pgAdmin: http://localhost:8082 (with --profile tools)
  🖥️ Selenium VNC: http://localhost:7900

EOF
}

# Function to handle service-specific operations
service_operations() {
    local operation="$1"
    local service="$2"

    case "$operation" in
        "logs")
            if [ -n "$service" ]; then
                print_status "Showing logs for $service..."
                $COMPOSE_CMD -f $COMPOSE_FILE logs --tail=50 -f "jobtrak-$service"
            else
                show_logs
            fi
            ;;
        "shell")
            if [ -z "$service" ]; then
                print_error "Please specify a service for shell access"
                print_status "Available services: api, postgres, redis, traefik, selenium"
                return 1
            fi

            local container_name="jobtrak-$service"
            print_status "Opening shell in $container_name..."

            case "$service" in
                "postgres")
                    $DOCKER_CMD exec -it $container_name psql -U jobtrak_user -d jobtrak
                    ;;
                "redis")
                    $DOCKER_CMD exec -it $container_name redis-cli
                    ;;
                *)
                    $DOCKER_CMD exec -it $container_name /bin/sh
                    ;;
            esac
            ;;
        "restart")
            if [ -n "$service" ]; then
                print_status "Restarting $service..."
                $COMPOSE_CMD -f $COMPOSE_FILE restart "jobtrak-$service"
                print_success "$service restarted"
            else
                restart_services
            fi
            ;;
    esac
}

# Function to handle development mode
dev_mode() {
    print_header "Starting JobTrak in Development Mode"

    # Set development environment variables
    export API_DEBUG=1
    export LOG_LEVEL=DEBUG

    # Start services with development overrides
    $COMPOSE_CMD -f $COMPOSE_FILE -f docker-compose.dev.yml up --build -d 2>/dev/null || {
        print_warning "Development compose file not found, using standard configuration"
        $COMPOSE_CMD -f $COMPOSE_FILE up --build -d
    }

    wait_for_services
    print_success "Development environment ready"

    # Show development-specific information
    echo ""
    print_status "Development Mode Active:"
    print_status "- API Debug mode enabled"
    print_status "- Hot reload enabled (if configured)"
    print_status "- Extended logging enabled"
}

# Function to check system requirements
check_requirements() {
    print_header "Checking System Requirements"

    local requirements_met=true

    # Check available memory
    local mem_gb=$(free -g | awk '/^Mem:/{print $2}')
    if [ "$mem_gb" -lt 4 ]; then
        print_warning "System has less than 4GB RAM ($mem_gb GB). Performance may be affected."
    else
        print_success "Memory: ${mem_gb}GB available"
    fi

    # Check available disk space
    local disk_gb=$(df -BG . | awk 'NR==2{print $4}' | sed 's/G//')
    if [ "$disk_gb" -lt 10 ]; then
        print_warning "Less than 10GB disk space available ($disk_gb GB). Consider freeing up space."
    else
        print_success "Disk space: ${disk_gb}GB available"
    fi

    # Check if ports are available
    local ports=(80 443 5432 6379 8000 8080 8082 4444)
    for port in "${ports[@]}"; do
        if netstat -tuln 2>/dev/null | grep -q ":$port "; then
            print_warning "Port $port is already in use"
            requirements_met=false
        fi
    done

    if [ "$requirements_met" = true ]; then
        print_success "All port checks passed"
    fi

    # Check Docker version
    local docker_version=$($DOCKER_CMD --version | grep -oE '[0-9]+\.[0-9]+\.[0-9]+' | head -1)
    print_success "Docker version: $docker_version"

    # Check Docker Compose version
    local compose_version=$($COMPOSE_CMD version --short 2>/dev/null || echo "unknown")
    print_success "Docker Compose version: $compose_version"

    return 0
}

# Function to generate SSL certificates manually (fallback)
generate_ssl_certs() {
    print_header "Generating SSL Certificates"

    if [ ! -d "letsencrypt/live/$DOMAIN" ]; then
        print_status "Generating self-signed certificates for development..."

        mkdir -p letsencrypt/live/$DOMAIN

        # Generate self-signed certificate
        openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
            -keyout letsencrypt/live/$DOMAIN/privkey.pem \
            -out letsencrypt/live/$DOMAIN/fullchain.pem \
            -subj "/C=US/ST=State/L=City/O=Organization/CN=$DOMAIN" \
            2>/dev/null || {
            print_warning "OpenSSL not available, using HTTP only"
            return 1
        }

        print_success "Self-signed certificates generated"
        print_warning "These are self-signed certificates. Browsers will show security warnings."
    else
        print_success "SSL certificates already exist"
    fi
}

# Function to test the API
test_api() {
    print_header "Testing API Endpoints"

    local base_url="http://localhost:8000"
    local endpoints=("/health" "/docs" "/api/v1/jobs" "/api/v1/status")

    for endpoint in "${endpoints[@]}"; do
        local url="$base_url$endpoint"
        if curl -f -s "$url" >/dev/null; then
            print_success "✓ $endpoint"
        else
            print_error "✗ $endpoint"
        fi
    done

    # Test domain if available
    if curl -f -s "https://$DOMAIN/health" >/dev/null 2>&1; then
        print_success "✓ Domain access: https://$DOMAIN"
    else
        print_warning "✗ Domain access not available: https://$DOMAIN"
    fi
}

# Function to backup entire system
backup_system() {
    print_header "Creating System Backup"

    local backup_dir="backup_$(date +%Y%m%d_%H%M%S)"
    mkdir -p "$backup_dir"

    # Backup database
    print_status "Backing up database..."
    $DOCKER_CMD exec jobtrak-postgres pg_dump -U jobtrak_user jobtrak > "$backup_dir/database.sql"

    # Backup volumes
    print_status "Backing up Redis data..."
    $DOCKER_CMD run --rm -v jobtrak-redis-data:/data -v "$(pwd)/$backup_dir":/backup alpine tar czf /backup/redis-data.tar.gz -C /data .

    # Backup configuration files
    print_status "Backing up configuration..."
    cp docker-compose.yml "$backup_dir/"
    cp -r init-scripts "$backup_dir/" 2>/dev/null || true
    cp .env "$backup_dir/" 2>/dev/null || true

    print_success "System backup created in: $backup_dir"
}

# Main script logic
main() {
    local command="${1:-start}"
    local arg2="$2"
    local arg3="$3"

    # Setup Docker commands first
    setup_docker_commands
    check_docker_daemon

    case "$command" in
        "start")
            setup_environment
            cleanup

            # Handle profiles
            if [ "$arg2" = "--profile" ] && [ -n "$arg3" ]; then
                print_status "Starting with profile: $arg3"
                $COMPOSE_CMD -f $COMPOSE_FILE --profile "$arg3" up --build -d
            elif [[ "$*" == *"--profile tools"* ]]; then
                print_status "Starting with tools profile (pgAdmin included)"
                $COMPOSE_CMD -f $COMPOSE_FILE --profile tools up --build -d
            else
                start_services
            fi

            show_status
            test_api

            echo ""
            print_success "🚀 JobTrak is now running!"
            print_status "📖 Check the status with: ./start.sh status"
            print_status "📋 View logs with: ./start.sh logs"
            ;;

        "stop")
            stop_services
            ;;

        "restart")
            restart_services
            show_status
            ;;

        "status")
            show_status
            ;;

        "logs")
            service_operations "logs" "$arg2"
            ;;

        "shell")
            service_operations "shell" "$arg2"
            ;;

        "update")
            update_services
            show_status
            ;;

        "cleanup")
            cleanup
            print_success "Cleanup completed"
            ;;

        "dev")
            setup_environment
            cleanup
            dev_mode
            show_status
            ;;

        "tools")
            setup_environment
            cleanup
            print_status "Starting with tools profile..."
            $COMPOSE_CMD -f $COMPOSE_FILE --profile tools up --build -d
            wait_for_services
            show_status
            ;;

        "db")
            database_operations "$command" "$arg2"
            ;;

        "test")
            test_api
            ;;

        "backup")
            backup_system
            ;;

        "requirements"|"req")
            check_requirements
            ;;

        "ssl")
            generate_ssl_certs
            ;;

        "help"|"-h"|"--help")
            show_usage
            ;;

        *)
            print_error "Unknown command: $command"
            echo ""
            show_usage
            exit 1
            ;;
    esac
}

# Trap signals for clean shutdown
trap 'print_warning "Interrupted by user"; exit 130' INT TERM

# Run main function with all arguments
main "$@"