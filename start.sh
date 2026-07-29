#!/usr/bin/env sh
set -eu

if ! command -v docker >/dev/null 2>&1 || ! docker compose version >/dev/null 2>&1; then
  echo "Docker Engine with the Compose plugin is required."
  exit 1
fi

if [ ! -f .env ]; then
  echo "Create .env first: cp .env.example .env"
  exit 1
fi

compose_profile=""
if [ "${1:-}" = "--observability" ]; then
  compose_profile="--profile observability"
elif [ -n "${1:-}" ]; then
  echo "Usage: ./start.sh [--observability]"
  exit 1
fi

# shellcheck disable=SC1091
. ./.env
api_port=${API_PORT:-8000}

# shellcheck disable=SC2086
docker compose $compose_profile up --build --detach

attempt=0
while [ "$attempt" -lt 30 ]; do
  if curl --fail --silent "http://localhost:${api_port}/health" >/dev/null; then
    echo "JobAgent is running at http://localhost:${api_port}"
    exit 0
  fi
  attempt=$((attempt + 1))
  sleep 2
done

echo "The API did not become healthy. Inspect logs with: docker compose logs api"
exit 1
