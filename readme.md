# JobTrak API

A comprehensive job tracking and resume generation API built with FastAPI, featuring intelligent resume tailoring using AI/LLM integration, job application management, and Simplify.jobs integration.

## 🚀 Features

### Core Functionality
- **Job Analysis & Tracking**: Analyze job postings from URLs and track application status
- **AI-Powered Resume Generation**: Generate tailored resumes using LLM integration (OpenAI)
- **Resume Management**: Upload, update, and manage multiple resume versions
- **Application Status Tracking**: Track jobs through various stages (NEW, INTERESTED, APPLIED, etc.)
- **Simplify.jobs Integration**: Upload resumes directly to Simplify.jobs platform

### Advanced Features
- **Smart Caching**: Multi-layer caching system for optimal performance
- **Batch Operations**: Efficient bulk processing of jobs and resumes
- **Orphan Prevention**: Intelligent handling of job-resume relationships
- **ATS Optimization**: Resume optimization for Applicant Tracking Systems
- **Background Processing**: Asynchronous resume generation with status tracking

## 🏗️ Architecture

### Technology Stack
- **Backend**: FastAPI (Python 3.8+)
- **Database**: PostgreSQL with asyncpg
- **AI/LLM**: OpenAI GPT integration via LangChain
- **Caching**: Multi-layer in-memory caching system
- **Web Scraping**: BeautifulSoup for job posting analysis
- **Authentication**: Header-based API key system

### System Components
1. **API Layer**: FastAPI routers handling HTTP requests
2. **Business Logic**: Service classes for resume generation and job management
3. **Data Layer**: Database and caching abstraction
4. **AI Integration**: LangChain-based resume optimization
5. **External APIs**: Simplify.jobs integration

## 📋 API Endpoints

### Job Management
```
POST   /api/jobs/analyze           # Analyze job posting from URL
GET    /api/jobs/                  # List all jobs with filtering
GET    /api/jobs/{job_id}          # Get specific job details
PUT    /api/jobs/{job_id}/status   # Update job application status
DELETE /api/jobs/{job_id}          # Delete job (with cascade options)
DELETE /api/jobs/batch             # Bulk delete jobs
GET    /api/jobs/stats             # Get job statistics
GET    /api/jobs/{job_id}/resumes  # Get resumes for specific job
```

### Resume Management
```
POST   /api/resume/generate        # Generate tailored resume
GET    /api/resume/{resume_id}/download      # Download resume (YAML)
GET    /api/resume/{resume_id}/status        # Check generation status
POST   /api/resume/upload          # Upload custom resume
POST   /api/resume/{resume_id}/update-yaml  # Update resume content
DELETE /api/resume/{resume_id}     # Delete resume
GET    /api/resume/                # List all resumes
GET    /api/resume/active          # Get active generation processes
```

### Simplify.jobs Integration
```
POST   /api/simplify/upload-resume-pdf      # Upload PDF to Simplify
POST   /api/simplify/store-tokens           # Store authentication tokens
GET    /api/simplify/check-session          # Validate session
GET    /api/simplify/get-tokens             # Retrieve stored tokens
```

### System Management
```
GET    /api/status                 # System health check
DELETE /api/cache/clear            # Clear user cache
POST   /api/cache/cleanup          # Clean expired cache
GET    /api/cache/stats            # Cache statistics
```

## 🛠️ Installation & Setup

### Prerequisites
- Python 3.8+
- PostgreSQL 12+
- OpenAI API Key

### Environment Variables
```bash
# Database
DATABASE_URL=postgresql://user:password@localhost:5432/jobtrak

# OpenAI Integration
OPENAI_API_KEY=your_openai_api_key_here

# Cache Configuration
JOB_CACHE_SIZE=1000
SEARCH_CACHE_SIZE=1000

# Server Configuration
HOST=0.0.0.0
PORT=8000
RELOAD=false

# CORS Configuration
API_DEBUG=false  # Set to true for additional CORS origins
```

### Installation Steps

1. **Clone the repository**
```bash
git clone <repository-url>
cd jobtrak-api
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Set up environment variables**
```bash
cp .env.example .env
# Edit .env with your configuration
```

5. **Initialize database**
```bash
python -c "
import asyncio
from data.database import Database
async def init_db():
    db = Database()
    await db.initialize_db()
asyncio.run(init_db())
"
```

6. **Run the application**
```bash
python main.py
# Or with uvicorn directly:
uvicorn main:app --host 0.0.0.0 --port 8000
```

## 📊 Data Models

### Job Status Lifecycle
```
NEW → INTERESTED → RESUME_GENERATED → APPLIED → INTERVIEW/REJECTED → OFFER/DECLINED
```

### Core Data Models

**Job**
- `id`: Unique identifier
- `job_url`: Source URL of job posting
- `status`: Current application status
- `date_found`: When job was discovered
- `applied_date`: Application submission date
- `resume_id`: Associated resume ID
- `metadata`: Parsed job details (title, company, requirements, etc.)

**Resume**
- `id`: Unique identifier
- `job_id`: Associated job (optional)
- `file_path`: Resume file location
- `yaml_content`: Structured resume data
- `date_created`: Creation timestamp
- `uploaded_to_simplify`: Simplify.jobs upload status

## 🤖 AI-Powered Resume Generation

### Resume Optimization Process

1. **Job Analysis**: Parse job posting to extract requirements, skills, and keywords
2. **Content Matching**: Identify relevant experience and skills from user's resume
3. **Tailored Generation**: Generate customized resume sections:
   - Professional summary/objective
   - Skill matching and prioritization
   - Experience highlighting
   - Project relevance scoring
4. **ATS Optimization**: Incorporate relevant keywords for ATS scanning

### Supported Resume Sections
- **Objective/Summary**: AI-generated professional summary
- **Experience**: Tailored experience highlights with relevance scoring
- **Projects**: Project descriptions optimized for job relevance
- **Skills**: Technical and non-technical skills matching
- **Education**: Formatted education information

## 💾 Caching Strategy

### Multi-Layer Caching System

1. **In-Memory Cache**: Fast access for frequently used data
   - Job cache with LRU eviction
   - Resume generation status cache
   - Search results cache

2. **Database Cache**: Persistent storage with optimized queries
   - Batch operations for performance
   - Connection pooling
   - Prepared statements

3. **Cache Management**: Intelligent cache invalidation and cleanup
   - TTL-based expiration
   - User-specific cache clearing
   - Background cleanup processes

## 🔧 Configuration

### Cache Configuration
```python
# Job cache settings
JOB_CACHE_MAX_SIZE = 1000
JOB_CACHE_TTL_SECONDS = 3600
JOB_CACHE_MAX_MEMORY_MB = 100

# Resume cache settings
RESUME_CACHE_TTL_SECONDS = 7200
```

### Database Configuration
```python
# Connection pool settings
DATABASE_MIN_POOL_SIZE = 2
DATABASE_MAX_POOL_SIZE = 20
DATABASE_COMMAND_TIMEOUT = 30
```

### LLM Configuration
```python
# Model settings
MODEL_NAME = "gpt-3.5-turbo"
MODEL_TEMPERATURE = 0.1
ENABLE_LLM_CACHE = True
```

## 🚦 Usage Examples

### Analyze a Job Posting
```bash
curl -X POST "http://localhost:8000/api/jobs/analyze" \
  -H "X-Api-Key: your_api_key" \
  -H "X-User-Id: user123" \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "job_url=https://example.com/job-posting"
```

### Generate Tailored Resume
```bash
curl -X POST "http://localhost:8000/api/resume/generate" \
  -H "X-Api-Key: your_api_key" \
  -H "X-User-Id: user123" \
  -H "Content-Type: application/json" \
  -d '{
    "job_id": "job123",
    "template": "standard",
    "customize": true,
    "resume_data": {
      "basic": {"name": "John Doe", "email": "john@example.com"},
      "experiences": [...],
      "skills": [...]
    }
  }'
```

### Check Resume Generation Status
```bash
curl -X GET "http://localhost:8000/api/resume/resume123/status" \
  -H "X-Api-Key: your_api_key" \
  -H "X-User-Id: user123"
```

## 🔐 Authentication

The API uses header-based authentication:
- `X-Api-Key`: Your OpenAI API key for LLM operations
- `X-User-Id`: User identifier for data isolation

## 🎯 Best Practices

### Resume Data Structure
Provide resume data in YAML format:
```yaml
basic:
  name: "John Doe"
  email: "john@example.com"
  phone: "+1-555-0123"

experiences:
  - company: "Tech Corp"
    titles:
      - name: "Senior Developer"
        startdate: "2020-01-01"
        enddate: "present"
    highlights:
      - "Led development of microservices architecture"
      - "Improved system performance by 40%"

skills:
  - category: "Technical"
    skills: ["Python", "JavaScript", "AWS", "Docker"]
```

### Error Handling
The API returns structured error responses:
```json
{
  "detail": "Error description",
  "status_code": 400,
  "timestamp": "2024-01-01T12:00:00Z"
}
```

### Rate Limiting
- Resume generation: Processed asynchronously
- Batch operations: Recommended for bulk updates
- Cache usage: Reduces API calls and improves performance

## 🧹 Maintenance

### Database Maintenance
```bash
# Clean up expired cache entries
curl -X POST "http://localhost:8000/api/cache/cleanup"

# Clear user-specific cache
curl -X DELETE "http://localhost:8000/api/cache/clear" \
  -H "X-User-Id: user123"
```

### Monitoring
- Health check endpoint: `GET /health`
- System status: `GET /api/status`
- Cache statistics: `GET /api/cache/stats`
