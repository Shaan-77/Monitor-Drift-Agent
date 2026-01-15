#!/bin/bash
# Entrypoint script for Monitor/Drift Agent container

set -e

echo "Starting Monitor/Drift Agent..."

# Wait for PostgreSQL to be ready
if [ -n "$DATABASE_URL" ]; then
    echo "Waiting for database to be ready..."
    # TODO: Implement database connection check
    # until pg_isready -h postgres -U monitor_drift; do
    #     sleep 2
    # done
    echo "Database is ready"
fi

# Run database migrations (if needed)
# TODO: Implement database migration script
# echo "Running database migrations..."
# python -m alembic upgrade head

# Validate configuration
echo "Validating configuration..."
# TODO: Add configuration validation
# python -c "from config.settings import settings; assert settings.validate()"

# Start the application
echo "Starting application..."
exec "$@"
