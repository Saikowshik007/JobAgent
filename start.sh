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
    if [ -n "$1" ]; then
        $COMPOSE_CMD -f $COMPOSE_FILE logs --tail=50 -f "jobtrak-$1"
    else
        $COMPOSE_CMD -f $COMPOSE_FILE logs --tail=50
    fi
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
    check_url_health "Traefik Dashboard" "http://localhost:8080/api/rawdata"

    # Check monitoring services if enabled
    if docker ps --format "{{.Names}}" | grep -q "jobtrak-grafana"; then
        check_url_health "Grafana" "http://localhost:3000/api/health"
    fi

    if docker ps --format "{{.Names}}" | grep -q "jobtrak-prometheus"; then
        check_url_health "Prometheus" "http://localhost:9090/-/healthy"
    fi

    # Show access URLs
    echo ""
    print_header "Access URLs"
    echo "🌐 API (Local): http://localhost:8000"
    echo "🌐 API (Domain): https://$DOMAIN"
    echo "🚦 Traefik Dashboard: http://localhost:8080"
    echo "🗄️ pgAdmin: http://localhost:8082 (start with --profile tools)"
    echo ""

    # Show monitoring URLs if enabled
    if docker ps --format "{{.Names}}" | grep -q "jobtrak-grafana"; then
        echo "📊 Monitoring URLs (--profile monitoring):"
        echo "   📈 Grafana: http://localhost:3000 (admin/admin123)"
        echo "   📊 Prometheus: http://localhost:9090"
        echo "   🔍 Loki: http://localhost:3100"
        echo "   📋 cAdvisor: http://localhost:8081"
        echo ""
    fi

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

# Function to handle database operations
database_operations() {
    local operation="$1"

    case "$operation" in
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
                    tablename,
                    pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) as size
                FROM pg_tables
                WHERE schemaname = 'public';
            " 2>/dev/null || print_error "Could not connect to database"
            ;;
    esac
}

# Function to handle service-specific operations
service_operations() {
    local operation="$1"
    local service="$2"

    case "$operation" in
        "logs")
            show_logs "$service"
            ;;
        "shell")
            if [ -z "$service" ]; then
                print_error "Please specify a service for shell access"
                print_status "Available services: api, postgres, redis, traefik, grafana, prometheus"
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

# Function to setup monitoring configurations
setup_monitoring_configs() {
    print_status "Setting up monitoring configurations..."

    # Create Prometheus config
    mkdir -p prometheus
    if [ ! -f "prometheus/prometheus.yml" ]; then
        cat > prometheus/prometheus.yml << 'EOF'
global:
  scrape_interval: 15s
  evaluation_interval: 15s

rule_files:
  # - "first_rules.yml"
  # - "second_rules.yml"

scrape_configs:
  - job_name: 'prometheus'
    static_configs:
      - targets: ['localhost:9090']

  - job_name: 'jobtrak-api'
    static_configs:
      - targets: ['jobtrak-api:8000']
    metrics_path: '/metrics'
    scrape_interval: 30s

  - job_name: 'cadvisor'
    static_configs:
      - targets: ['cadvisor:8080']

  - job_name: 'postgres'
    static_configs:
      - targets: ['postgres:5432']

  - job_name: 'redis'
    static_configs:
      - targets: ['redis:6379']
EOF
    fi

    # Create Loki config
    mkdir -p loki
    if [ ! -f "loki/local-config.yaml" ]; then
        cat > loki/local-config.yaml << 'EOF'
auth_enabled: false

server:
  http_listen_port: 3100
  grpc_listen_port: 9096

ingester:
  wal:
    enabled: true
    dir: /loki/wal
  lifecycler:
    address: 127.0.0.1
    ring:
      kvstore:
        store: inmemory
      replication_factor: 1
    final_sleep: 0s
  chunk_idle_period: 1h
  max_chunk_age: 1h
  chunk_target_size: 1048576
  chunk_retain_period: 30s
  max_transfer_retries: 0

schema_config:
  configs:
    - from: 2020-10-24
      store: boltdb-shipper
      object_store: filesystem
      schema: v11
      index:
        prefix: index_
        period: 24h

storage_config:
  boltdb_shipper:
    active_index_directory: /loki/boltdb-shipper-active
    cache_location: /loki/boltdb-shipper-cache
    cache_ttl: 24h
    shared_store: filesystem
  filesystem:
    directory: /loki/chunks

compactor:
  working_directory: /loki/boltdb-shipper-compactor
  shared_store: filesystem

limits_config:
  reject_old_samples: true
  reject_old_samples_max_age: 168h

chunk_store_config:
  max_look_back_period: 0s

table_manager:
  retention_deletes_enabled: false
  retention_period: 0s

ruler:
  storage:
    type: local
    local:
      directory: /loki/rules
  rule_path: /loki/rules
  alertmanager_url: http://localhost:9093
  ring:
    kvstore:
      store: inmemory
  enable_api: true
EOF
    fi

    # Create Promtail config
    mkdir -p promtail
    if [ ! -f "promtail/config.yml" ]; then
        cat > promtail/config.yml << 'EOF'
server:
  http_listen_port: 9080
  grpc_listen_port: 0

positions:
  filename: /tmp/positions.yaml

clients:
  - url: http://loki:3100/loki/api/v1/push

scrape_configs:
  - job_name: containers
    static_configs:
      - targets:
          - localhost
        labels:
          job: containerlogs
          __path__: /var/lib/docker/containers/*/*log

    pipeline_stages:
      - json:
          expressions:
            output: log
            stream: stream
            attrs:
      - json:
          source: attrs
          expressions:
            tag:
      - regex:
          source: tag
          expression: (?P<container_name>(?:[^|]*))\|(?P<image_name>(?:[^|]*))\|(?P<image_id>(?:[^|]*))\|(?P<container_id>(?:[^|]*))
      - timestamp:
          format: RFC3339Nano
          source: time
      - labels:
          stream:
          container_name:
          image_name:
          image_id:
          container_id:
      - output:
          source: output

  - job_name: jobtrak-logs
    static_configs:
      - targets:
          - localhost
        labels:
          job: jobtrak
          __path__: /var/log/jobtrak/*.log

  - job_name: syslog
    static_configs:
      - targets:
          - localhost
        labels:
          job: syslog
          __path__: /var/log/syslog
EOF
    fi

    # Create Grafana provisioning
    mkdir -p grafana/provisioning/datasources grafana/provisioning/dashboards grafana/dashboards

    if [ ! -f "grafana/provisioning/datasources/datasources.yaml" ]; then
        cat > grafana/provisioning/datasources/datasources.yaml << 'EOF'
apiVersion: 1

datasources:
  - name: Prometheus
    type: prometheus
    access: proxy
    url: http://prometheus:9090
    isDefault: true

  - name: Loki
    type: loki
    access: proxy
    url: http://loki:3100

  - name: PostgreSQL
    type: postgres
    url: postgres:5432
    database: jobtrak
    user: jobtrak_user
    secureJsonData:
      password: jobtrak_secure_password_2024
    jsonData:
      sslmode: disable
EOF
    fi

    if [ ! -f "grafana/provisioning/dashboards/dashboards.yaml" ]; then
        cat > grafana/provisioning/dashboards/dashboards.yaml << 'EOF'
apiVersion: 1

providers:
  - name: 'default'
    orgId: 1
    folder: ''
    type: file
    disableDeletion: false
    updateIntervalSeconds: 10
    allowUiUpdates: true
    options:
      path: /var/lib/grafana/dashboards
EOF
    fi

    print_success "Monitoring configurations created"
}

# Function to setup Grafana dashboards
setup_grafana_dashboards() {
    print_status "Setting up Grafana dashboards..."

    # Wait for Grafana to be ready
    local counter=0
    while [ $counter -lt 60 ]; do
        if curl -f "http://localhost:3000/api/health" >/dev/null 2>&1; then
            break
        fi
        counter=$((counter + 1))
        sleep 1
    done

    # Create a simple JobTrak dashboard
    if [ ! -f "grafana/dashboards/jobtrak-dashboard.json" ]; then
        cat > grafana/dashboards/jobtrak-dashboard.json << 'EOF'
{
  "dashboard": {
    "id": null,
    "title": "JobTrak Dashboard",
    "tags": ["jobtrak"],
    "timezone": "browser",
    "panels": [
      {
        "id": 1,
        "title": "API Response Time",
        "type": "graph",
        "targets": [
          {
            "expr": "histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m]))",
            "legendFormat": "95th percentile"
          }
        ]
      },
      {
        "id": 2,
        "title": "Database Connections",
        "type": "graph",
        "targets": [
          {
            "expr": "pg_stat_database_numbackends",
            "legendFormat": "Active connections"
          }
        ]
      },
      {
        "id": 3,
        "title": "Redis Memory Usage",
        "type": "graph",
        "targets": [
          {
            "expr": "redis_memory_used_bytes",
            "legendFormat": "Memory used"
          }
        ]
      }
    ],
    "time": {
      "from": "now-6h",
      "to": "now"
    },
    "refresh": "5s"
  }
}
EOF
    fi

    print_success "Grafana dashboards configured"
}
    print_header "Testing API Endpoints"

    local base_url="http://localhost:8000"
    local endpoints=("/health" "/docs")

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
  logs [service] - Show logs (optionally for specific service)

Database Commands:
  db init        - Initialize database schema
  db reset       - Reset database (WARNING: deletes all data)
  db backup      - Create database backup
  db status      - Show database status

Service Commands:
  shell [service]- Open shell in service container
  test           - Test API endpoints

Options:
  --profile [profile] - Start with specific profile (tools)

Examples:
  ./start.sh                    # Start all services
  ./start.sh start --profile tools  # Start with pgAdmin
  ./start.sh logs api           # Show API logs
  ./start.sh db init            # Initialize database
  ./start.sh status             # Show service status
  ./start.sh shell postgres     # Open PostgreSQL shell

Access Points:
  🌐 API: http://localhost:8000 or https://jobtrackai.duckdns.org
  🚦 Traefik: http://localhost:8080
  🗄️ pgAdmin: http://localhost:8082 (with --profile tools)
  🖥️ Selenium VNC: http://localhost:7900

EOF
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
                wait_for_services
            elif [[ "$*" == *"--profile tools"* ]]; then
                print_status "Starting with tools profile (pgAdmin included)"
                $COMPOSE_CMD -f $COMPOSE_FILE --profile tools up --build -d
                wait_for_services
            elif [[ "$*" == *"--profile monitoring"* ]]; then
                print_status "Starting with monitoring profile (Grafana, Prometheus, Loki included)"
                setup_monitoring_configs
                $COMPOSE_CMD -f $COMPOSE_FILE --profile monitoring up --build -d
                wait_for_services
                setup_grafana_dashboards
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

        "db")
            database_operations "$arg2"
            ;;

        "test")
            test_api
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