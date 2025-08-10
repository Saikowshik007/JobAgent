# JobTrak API Scaling Deployment Script for Windows
# This script scales your API from 1 to 3 instances with load balancing

param(
    [Parameter(Position=0)]
    [ValidateSet("deploy", "status", "rollback", "logs")]
    [string]$Action = "deploy",

    [Parameter(Position=1)]
    [string]$ServiceName = ""
)

# Function to write colored output
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

# Check if Docker is running
function Test-Docker {
    Write-ColorOutput "Checking if Docker is running..." "Info"

    try {
        $dockerInfo = docker info 2>$null
        if ($LASTEXITCODE -eq 0) {
            Write-ColorOutput "Docker is running ✓" "Success"
            return $true
        }
    }
    catch {
        Write-ColorOutput "Docker is not running. Please start Docker Desktop first." "Error"
        return $false
    }

    Write-ColorOutput "Docker is not running. Please start Docker Desktop first." "Error"
    return $false
}

# Backup current setup
function Backup-CurrentSetup {
    Write-ColorOutput "Creating backup of current setup..." "Info"

    $timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
    $backupDir = ".\backup_$timestamp"

    # Create backup directory
    New-Item -ItemType Directory -Path $backupDir -Force | Out-Null

    # Backup current docker-compose file
    if (Test-Path "docker-compose.yml") {
        Copy-Item "docker-compose.yml" "$backupDir\" -Force
        Write-ColorOutput "Backed up docker-compose.yml" "Info"
    }

    # Backup database if running
    $postgresContainer = docker ps --format "{{.Names}}" | Where-Object { $_ -match "postgres" }
    if ($postgresContainer) {
        Write-ColorOutput "Creating database backup..." "Info"
        try {
            docker exec $postgresContainer pg_dump -U jobtrak_user jobtrak > "$backupDir\database_backup.sql"
            Write-ColorOutput "Database backup created" "Success"
        }
        catch {
            Write-ColorOutput "Database backup failed" "Warning"
        }
    }

    # Save backup directory reference
    $backupDir | Out-File -FilePath ".last_backup" -Encoding UTF8
    Write-ColorOutput "Backup completed in $backupDir" "Success"
}

# Deploy the scaled setup
function Deploy-ScaledSetup {
    Write-ColorOutput "Stopping current services..." "Info"
    docker-compose down --timeout 30 2>$null

    Write-ColorOutput "Starting scaled services..." "Info"

    # Start database and Redis first
    docker-compose up -d postgres redis selenium-chrome

    Write-ColorOutput "Waiting for database to be ready..." "Info"
    Start-Sleep -Seconds 15

    # Wait for PostgreSQL to be ready
    $maxAttempts = 12
    for ($i = 1; $i -le $maxAttempts; $i++) {
        try {
            $result = docker exec jobtrak-postgres pg_isready -U jobtrak_user -d jobtrak 2>$null
            if ($LASTEXITCODE -eq 0) {
                Write-ColorOutput "PostgreSQL is ready ✓" "Success"
                break
            }
        }
        catch {
            # Continue waiting
        }

        if ($i -eq $maxAttempts) {
            Write-ColorOutput "PostgreSQL failed to start after 2 minutes" "Error"
            exit 1
        }

        Write-ColorOutput "Waiting for PostgreSQL... (attempt $i/$maxAttempts)" "Info"
        Start-Sleep -Seconds 10
    }

    # Start all API instances and Traefik
    Write-ColorOutput "Starting API instances and load balancer..." "Info"
    docker-compose up -d

    Write-ColorOutput "Waiting for services to be ready..." "Info"
    Start-Sleep -Seconds 20
}

# Verify the deployment
function Test-Deployment {
    Write-ColorOutput "Verifying deployment..." "Info"

    # Check if all containers are running
    $services = @("jobtrak-api-1", "jobtrak-api-2", "jobtrak-api-3", "jobtrak-postgres", "jobtrak-redis", "jobtrak-traefik")

    $runningContainers = docker ps --format "{{.Names}}"

    foreach ($service in $services) {
        if ($runningContainers -contains $service) {
            Write-ColorOutput "$service is running ✓" "Success"
        }
        else {
            Write-ColorOutput "$service is not running ✗" "Error"
        }
    }

    # Test load balancer
    Write-ColorOutput "Testing load balancer..." "Info"
    Start-Sleep -Seconds 10

    # Check if Traefik dashboard is accessible
    try {
        $response = Invoke-WebRequest -Uri "http://localhost:8080" -TimeoutSec 5 -UseBasicParsing
        if ($response.StatusCode -eq 200) {
            Write-ColorOutput "Traefik dashboard accessible ✓" "Success"
        }
    }
    catch {
        Write-ColorOutput "Traefik dashboard not accessible" "Warning"
    }

    Write-ColorOutput "Deployment verification completed!" "Success"
}

# Show status
function Show-Status {
    Write-Host ""
    Write-Host "=== JobTrak Scaling Status ===" -ForegroundColor Cyan
    Write-Host ""

    # Show running containers
    Write-Host "Running Containers:" -ForegroundColor Yellow
    $containers = docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}" | Where-Object { $_ -match "jobtrak" }
    $containers | ForEach-Object { Write-Host $_ }
    Write-Host ""

    # Show service URLs
    Write-Host "Service URLs:" -ForegroundColor Yellow
    Write-Host "• Application: https://jobtrackai.duckdns.org"
    Write-Host "• Traefik Dashboard: http://localhost:8080"
    Write-Host ""

    # Show quick resource usage
    Write-Host "Resource Usage:" -ForegroundColor Yellow
    $stats = docker stats --no-stream --format "table {{.Container}}\t{{.CPUPerc}}\t{{.MemUsage}}" | Where-Object { $_ -match "jobtrak" }
    $stats | ForEach-Object { Write-Host $_ }
    Write-Host ""

    Write-ColorOutput "Your application is now running with 3 API instances!" "Success"
    Write-ColorOutput "Monitor with: docker-compose logs -f" "Info"
}

# Rollback function
function Invoke-Rollback {
    if (Test-Path ".last_backup") {
        $backupDir = Get-Content ".last_backup" -Raw
        $backupDir = $backupDir.Trim()

        if (Test-Path "$backupDir\docker-compose.yml") {
            Write-ColorOutput "Rolling back to previous setup..." "Warning"
            docker-compose down --timeout 30
            Copy-Item "$backupDir\docker-compose.yml" "." -Force
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

# Show logs
function Show-Logs {
    param([string]$Service)

    if ($Service) {
        docker-compose logs -f $Service
    }
    else {
        docker-compose logs -f
    }
}

# Main function
function Main {
    # Display banner
    Write-Host ""
    Write-Host "+==============================================+" -ForegroundColor Green
    Write-Host "|        JobTrak API Scaling                   |" -ForegroundColor Green
    Write-Host "|    1 -> 3 Instances + Load Balancer         |" -ForegroundColor Green
    Write-Host "+==============================================+" -ForegroundColor Green
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
            Show-Status
        }
        "rollback" {
            Invoke-Rollback
        }
        "logs" {
            Show-Logs -Service $ServiceName
        }
        default {
            Write-Host "Usage: .\deploy-scaling.ps1 [deploy|status|rollback|logs] [service_name]"
            Write-Host ""
            Write-Host "Commands:"
            Write-Host "  deploy   - Scale API to 3 instances (default)"
            Write-Host "  status   - Show current status"
            Write-Host "  rollback - Rollback to previous setup"
            Write-Host "  logs     - Show logs (optional: specify service name)"
            Write-Host ""
            Write-Host "Examples:"
            Write-Host "  .\deploy-scaling.ps1 deploy"
            Write-Host "  .\deploy-scaling.ps1 status"
            Write-Host "  .\deploy-scaling.ps1 logs jobtrak-api-1"
        }
    }
}

# Check execution policy
try {
    $executionPolicy = Get-ExecutionPolicy
    if ($executionPolicy -eq "Restricted") {
        Write-ColorOutput "PowerShell execution policy is restricted. You may need to run:" "Warning"
        Write-Host "Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser" -ForegroundColor Yellow
        Write-Host ""
    }
}
catch {
    # Ignore execution policy check errors
}

# Run main function
Main