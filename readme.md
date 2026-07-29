# JobAgent API

FastAPI backend for extracting job requirements and producing factual,
job-targeted resume YAML. It runs with PostgreSQL and Redis.

## Recommended Ubuntu setup

Use **Docker Engine with the Docker Compose plugin**. It starts the API,
PostgreSQL, and Redis as one isolated stack with persistent named volumes.

```bash
cp .env.example .env
# Set a strong POSTGRES_PASSWORD in .env
chmod +x start.sh
./start.sh
```

The API is available at `http://localhost:8000` by default. Confirm it with:

```bash
curl http://localhost:8000/health
```

Useful commands:

```bash
docker compose logs -f api
docker compose ps
docker compose down              # stops services; preserves database volumes
docker compose down -v           # deletes all local JobAgent data
```

## Configuration

`.env` is intentionally ignored by Git. Set `POSTGRES_PASSWORD` before
starting. The API receives each user's OpenAI API key in its existing request
body, so no server-wide OpenAI key is required for resume generation.

## Runtime choices

- **Docker Compose — recommended.** One command, repeatable dependencies, and
  production-like local behavior.
- **Podman — supported alternative.** This Compose file is portable, but
  `podman compose` delegates to an external compose provider, adding an extra
  dependency on Ubuntu.
- **Standalone Python — development only.** You must manually install and
  operate PostgreSQL, Redis, Python, environment variables, and a service
  manager such as systemd.

## Resume generation safeguards

Generation builds a job-to-resume evidence map before editing. It marks real
gaps instead of inventing skills, then performs a final grounding review and
reverts sections that introduce unsupported claims.
