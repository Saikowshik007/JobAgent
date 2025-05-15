# JobTrak - LinkedIn Job Search & Tracking System

A tool for automated LinkedIn job searching and application tracking with a modern FastAPI-based API.

## Features

- **Automated LinkedIn Job Search**: Search for jobs based on keywords, location, and filters
- **Job Tracking**: Store and manage job applications
- **Search History**: Keep track of previous job searches
- **Customizable Filters**: Filter by experience level, job type, date posted, and more
- **Easy Apply Detection**: Identify jobs with LinkedIn's Easy Apply option

## Setup & Installation

### Prerequisites

- Python 3.8+
- Chrome browser
- LinkedIn account credentials

### Installation

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Set up your LinkedIn credentials as environment variables:
   ```bash
   export LINKEDIN_EMAIL="your.email@example.com"
   export LINKEDIN_PASSWORD="your-password"
   ```

### Configuration

Create a `config.yaml` file or use the default configuration:

```yaml
database:
  path: "job_tracker.db"
cache:
  job_cache_size: 1000
  search_cache_size: 100
linkedin:
  email: ""  # Can be set here instead of environment variables
  password: ""  # Can be set here instead of environment variables
  headless: true
api:
  host: "0.0.0.0"
  port: 8000
  debug: false
paths:
  output_dir: "output"
```

## Running the Application

Start the application with:

```bash
python jobtrak_system.py
```

The API server will start at http://localhost:8000 with auto-generated API documentation at:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## Core API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/status` | GET | Get system status and job statistics |
| `/api/jobs/search` | POST | Search for jobs on LinkedIn |
| `/api/jobs` | GET | Get jobs from the database with optional filtering |
| `/api/jobs/{job_id}` | GET | Get a specific job by ID |
| `/api/jobs/{job_id}/status` | PUT | Update the status of a job |
| `/api/resume/generate` | POST | Generate a resume for a job (placeholder) |
| `/api/resume/upload-to-simplify` | POST | Upload a resume to Simplify.jobs (placeholder) |

## Example API Usage

### Search for Jobs

```bash
curl -X 'POST' \
  'http://localhost:8000/api/jobs/search' \
  -H 'Content-Type: application/json' \
  -d '{
  "keywords": "Python Developer",
  "location": "San Francisco, CA",
  "filters": {
    "experience_level": ["Entry level", "Associate"],
    "job_type": ["Full-time"],
    "date_posted": "Past week",
    "workplace_type": ["Remote"],
    "easy_apply": true
  },
  "max_jobs": 20
}'
```

### Get All Jobs

```bash
curl -X 'GET' 'http://localhost:8000/api/jobs?limit=10&offset=0'
```

### Update Job Status

```bash
curl -X 'PUT' \
  'http://localhost:8000/api/jobs/12345/status' \
  -H 'Content-Type: application/json' \
  -d '{
  "status": "INTERESTED"
}'
```

## License

MIT License
