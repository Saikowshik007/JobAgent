#!/bin/bash
set -e

echo "🚀 Starting JobTrak services..."

# Start data layer
echo "📊 Starting data services..."
docker-compose -f docker-compose-data.yaml --profile tools up -d

# Wait for PostgreSQL to be ready
echo "⏳ Waiting for PostgreSQL..."
until docker exec jobtrak-postgres pg_isready -U jobtrak_user -d jobtrak; do
  sleep 2
done

# Start application layer
echo "🖥️  Starting application services..."
docker-compose -f docker-compose.yml up -d --build

echo "✅ All services started!"
echo ""
echo "🌐 Access Points:"
echo "   • API: http://localhost:8000"
echo "   • PgAdmin: http://localhost:8082 (admin@jobtrak.local / admin123)"
echo "   • Redis Commander: http://localhost:8081 (admin / admin123)"
echo "   • Traefik Dashboard: http://localhost:8080"
echo ""
echo "📊 Database:"
echo "   • Host: localhost:5432"
echo "   • User: jobtrak_user"
echo "   • Password: jobtrak_secure_password_2024"
echo "   • Database: jobtrak"#!/bin/bash
                             set -e

                             echo "🚀 Starting JobTrak services..."

                             # Start data layer
                             echo "📊 Starting data services..."
                             docker-compose -f docker-compose-data.yaml --profile tools up -d

                             # Wait for PostgreSQL to be ready
                             echo "⏳ Waiting for PostgreSQL..."
                             until docker exec jobtrak-postgres pg_isready -U jobtrak_user -d jobtrak; do
                               sleep 2
                             done

                             # Start application layer
                             echo "🖥️  Starting application services..."
                             docker-compose -f docker-compose.yml up -d --build

                             echo "✅ All services started!"
                             echo ""
                             echo "🌐 Access Points:"
                             echo "   • API: http://localhost:8000"
                             echo "   • PgAdmin: http://localhost:8082 (admin@jobtrak.local / admin123)"
                             echo "   • Redis Commander: http://localhost:8081 (admin / admin123)"
                             echo "   • Traefik Dashboard: http://localhost:8080"
                             echo ""
                             echo "📊 Database:"
                             echo "   • Host: localhost:5432"
                             echo "   • User: jobtrak_user"
                             echo "   • Password: jobtrak_secure_password_2024"
                             echo "   • Database: jobtrak"