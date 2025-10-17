# JobTrak API Scaling Deployment Script for Windows
# This script scales your API from 1 to 3 instances
# Updated to match current docker-compose.yml configuration

param(
    [Parameter(Position=0)]
    [ValidateSet("deploy", "status", "rollback", "logs")]
    [string]$Action = "deploy",

    [Parameter(Position=1)]
    [string]$ServiceName = ""
)

function Write-ColorOutput {
    param(
        [string]$Message,
        [string]$Type = "Info"
    )

    switch ($Type) {
        "Info" { Write-Host "[INFO] $Message" -ForegroundColor Green }
        "Warning" { Write-Host "[WARNING] $Message" -ForegroundColor Yellow }
        "Error" { Write-Host "[ERROR] $Message" -ForegroundColor Red }
        "Success" { Write-Host "[SUCCESS] $Message" -ForegroundColor Cyan }
    }
}

function Test-Docker {
    Write-ColorOutput "Checking if Docker is running..." "Info"

    try {
        $null = docker info 2>$null
        if ($LASTEXITCODE -eq 0) {
            Write-ColorOutput "Docker is running" "Success"
            return $true
        }
        else {
            Write-ColorOutput "Docker is not running. Please start Docker Desktop first." "Error"
            return $false
        }
    }
    catch {
        Write-ColorOutput "Docker is not running. Please start Docker Desktop first." "Error"
        return $false
    }
}

function Ensure-Network {
    Write-ColorOutput "Ensuring Docker network exists..." "Info"

    $networkExists = docker network ls --format "{{.Name}}" | Where-Object { $_ -eq "my-network" }

    if (-not $networkExists) {
        Write-ColorOutput "Creating Docker network 'my-network'..." "Info"
        docker network create my-network
        Write-ColorOutput "Docker network created" "Success"
    } else {
        Write-ColorOutput "Docker network 'my-network' already exists" "Success"
    }
}

function Backup-CurrentSetup {
    Write-ColorOutput "Creating backup of current setup..." "Info"

    $timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
    $backupDir = ".\backup_$timestamp"

    New-Item -ItemType Directory -Path $backupDir -Force | Out-Null

    if (Test-Path "docker-compose.yml") {
        Copy-Item "docker-compose.yml" "$backupDir\" -Force
        Write-ColorOutput "Backed up docker-compose.yml" "Info"
    }

    $postgresContainer = docker ps --format "{{.Names}}" | Where-Object { $_ -match "jobtrak-postgres" }
    if ($postgresContainer) {
        Write-ColorOutput "Creating database backup..." "Info"
        try {
            docker exec jobtrak-postgres pg_dump -U jobtrak_user jobtrak > "$backupDir\database_backup.sql" 2>$null
            if ($LASTEXITCODE -eq 0) {
                Write-ColorOutput "Database backup created" "Success"
            } else {
                Write-ColorOutput "Database backup failed" "Warning"
            }
        }
        catch {
            Write-ColorOutput "Database backup failed" "Warning"
        }
    }

    $backupDir | Out-File -FilePath ".last_backup" -Encoding UTF8
    Write-ColorOutput "Backup completed in $backupDir" "Success"
}

function Deploy-ScaledSetup {
    Write-ColorOutput "Stopping current services..." "Info"
    docker-compose down --timeout 30 2>$null

    # Ensure network exists
    Ensure-Network

    # Create necessary directories
    Write-ColorOutput "Creating necessary directories..." "Info"
    $directories = @("logs", "database", "output")
    foreach ($dir in $directories) {
        if (-not (Test-Path $dir)) {
            New-Item -ItemType Directory -Path $dir -Force | Out-Null
            Write-ColorOutput "Created directory: $dir" "Info"
        }
    }

    Write-ColorOutput "Starting database services first..." "Info"
    docker-compose up -d postgres redis

    Write-ColorOutput "Waiting for database to be ready..." "Info"
    Start-Sleep -Seconds 10

    $maxAttempts = 12
    for ($i = 1; $i -le $maxAttempts; $i++) {
        try {
            $null = docker exec jobtrak-postgres pg_isready -U jobtrak_user -d jobtrak 2>$null
            if ($LASTEXITCODE -eq 0) {
                Write-ColorOutput "PostgreSQL is ready" "Success"
                break
            }
        }
        catch {
            # Continue waiting
        }

        if ($i -eq $maxAttempts) {
            Write-ColorOutput "PostgreSQL failed to start after 2 minutes" "Error"
            Write-ColorOutput "Checking PostgreSQL logs..." "Info"
            docker logs jobtrak-postgres --tail 20
            exit 1
        }

        Write-ColorOutput "Waiting for PostgreSQL... (attempt $i/$maxAttempts)" "Info"
        Start-Sleep -Seconds 10
    }

    # Verify Redis is ready
    Write-ColorOutput "Checking Redis status..." "Info"
    $redisReady = $false
    for ($i = 1; $i -le 5; $i++) {
        try {
            $null = docker exec jobtrak-redis redis-cli ping 2>$null
            if ($LASTEXITCODE -eq 0) {
                Write-ColorOutput "Redis is ready" "Success"
                $redisReady = $true
                break
            }
        }
        catch {
            # Continue waiting
        }
        Start-Sleep -Seconds 2
    }

    if (-not $redisReady) {
        Write-ColorOutput "Redis may not be fully ready, continuing anyway..." "Warning"
    }

    Write-ColorOutput "Starting API instances..." "Info"
    docker-compose up -d jobtrak-api-1 jobtrak-api-2 jobtrak-api-3

    Write-ColorOutput "Waiting for API services to be ready..." "Info"
    Start-Sleep -Seconds 20
}

function Test-Deployment {
    Write-ColorOutput "Verifying deployment..." "Info"

    $services = @("jobtrak-api-1", "jobtrak-api-2", "jobtrak-api-3", "jobtrak-postgres", "jobtrak-redis")
    $runningContainers = docker ps --format "{{.Names}}"

    $allRunning = $true
    foreach ($service in $services) {
        if ($runningContainers -contains $service) {
            Write-ColorOutput "$service is running" "Success"
        }
        else {
            Write-ColorOutput "$service is not running" "Error"
            $allRunning = $false
        }
    }

    if ($allRunning) {
        Write-ColorOutput "Testing API endpoints..." "Info"

        # Test each API instance
        $ports = @(8001)  # Only API-1 has exposed port for direct access
        foreach ($port in $ports) {
            try {
                $response = Invoke-WebRequest -Uri "http://localhost:$port/health" -TimeoutSec 5 -UseBasicParsing
                if ($response.StatusCode -eq 200) {
                    Write-ColorOutput "API on port $port is responding" "Success"
                }
            }
            catch {
                Write-ColorOutput "API on port $port is not responding (may still be starting)" "Warning"
            }
        }
    }

    Write-ColorOutput "Deployment verification completed!" "Success"
}

function Show-Status {
    Write-Host ""
    Write-Host "=== JobTrak Scaling Status ===" -ForegroundColor Cyan
    Write-Host ""

    Write-Host "Running Containers:" -ForegroundColor Yellow
    docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}" | Where-Object { $_ -match "jobtrak" -or $_ -match "NAME" }
    Write-Host ""

    Write-Host "Service URLs:" -ForegroundColor Yellow
    Write-Host "- API Instance 1 (Direct): http://localhost:8001"
    Write-Host "- PostgreSQL: localhost:5432"
    Write-Host "- Redis: localhost:6379"
    Write-Host ""

    Write-Host "Note: API instances 2 and 3 are running but not directly accessible." -ForegroundColor Gray
    Write-Host "Consider adding a load balancer (like Traefik or Nginx) to distribute traffic." -ForegroundColor Gray
    Write-Host ""

    Write-Host "Resource Usage:" -ForegroundColor Yellow
    docker stats --no-stream --format "table {{.Container}}\t{{.CPUPerc}}\t{{.MemUsage}}" | Where-Object { $_ -match "jobtrak" -or $_ -match "CONTAINER" }
    Write-Host ""

    Write-ColorOutput "Your application is now running with 3 API instances!" "Success"
    Write-ColorOutput "Monitor logs with: docker-compose logs -f [service_name]" "Info"
}

function Invoke-Rollback {
    if (Test-Path ".last_backup") {
        $backupDir = (Get-Content ".last_backup" -Raw).Trim()

        if (Test-Path "$backupDir\docker-compose.yml") {
            Write-ColorOutput "Rolling back to previous setup..." "Warning"
            docker-compose down --timeout 30
            Copy-Item "$backupDir\docker-compose.yml" "." -Force

            # Restore database if backup exists
            if (Test-Path "$backupDir\database_backup.sql") {
                Write-ColorOutput "Restoring database backup..." "Info"
                docker-compose up -d postgres
                Start-Sleep -Seconds 10

                Get-Content "$backupDir\database_backup.sql" | docker exec -i jobtrak-postgres psql -U jobtrak_user -d jobtrak
                Write-ColorOutput "Database restored" "Success"
            }

            docker-compose up -d
            Write-ColorOutput "Rollback completed" "Success"
        }
        else {
            Write-ColorOutput "Backup file not found" "Error"
        }
    }
    else {
        Write-ColorOutput "No backup reference found" "Error"
    }
}

function Show-Logs {
    param([string]$Service)

    if ($Service) {
        if ($Service -in @("api-1", "api-2", "api-3")) {
            $Service = "jobtrak-$Service"
        }
        docker-compose logs -f $Service
    }
    else {
        # Show logs for all services
        docker-compose logs -f
    }
}

# Main execution
Write-Host ""
Write-Host "================================================" -ForegroundColor Green
Write-Host "        JobTrak API Scaling                     " -ForegroundColor Green
Write-Host "    Deploy 3 API Instances                      " -ForegroundColor Green
Write-Host "================================================" -ForegroundColor Green
Write-Host ""

switch ($Action) {
    "deploy" {
        if (-not (Test-Docker)) { exit 1 }
        Backup-CurrentSetup
        Deploy-ScaledSetup
        Test-Deployment
        Show-Status
    }
    "status" {
        if (-not (Test-Docker)) { exit 1 }
        Show-Status
    }
    "rollback" {
        if (-not (Test-Docker)) { exit 1 }
        Invoke-Rollback
    }
    "logs" {
        if (-not (Test-Docker)) { exit 1 }
        Show-Logs -Service $ServiceName
    }
    default {
        Write-Host "Usage: .\deploy-scaling.ps1 [deploy|status|rollback|logs] [service_name]"
        Write-Host ""
        Write-Host "Commands:"
        Write-Host "  deploy   - Deploy 3 API instances (default)"
        Write-Host "  status   - Show current status"
        Write-Host "  rollback - Rollback to previous setup"
        Write-Host "  logs     - Show logs (optional: specify service name)"
        Write-Host ""
        Write-Host "Examples:"
        Write-Host "  .\deploy-scaling.ps1 deploy"
        Write-Host "  .\deploy-scaling.ps1 status"
        Write-Host "  .\deploy-scaling.ps1 rollback"
        Write-Host "  .\deploy-scaling.ps1 logs api-1"
        Write-Host "  .\deploy-scaling.ps1 logs postgres"
    }
}