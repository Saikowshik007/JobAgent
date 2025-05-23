# Pydantic models for API validation
from enum import Enum
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field


class JobStatusEnum(str, Enum):
    """Enum for job status values matching the JobStatus class."""
    NEW = "NEW"
    INTERESTED = "INTERESTED"
    RESUME_GENERATED = "RESUME_GENERATED"
    APPLIED = "APPLIED"
    REJECTED = "REJECTED"
    INTERVIEW = "INTERVIEW"
    OFFER = "OFFER"
    DECLINED = "DECLINED"

class FilterOptions(BaseModel):
    """Search filter options."""
    experience_level: Optional[List[str]] = None
    job_type: Optional[List[str]] = None
    date_posted: Optional[str] = None
    workplace_type: Optional[List[str]] = None
    easy_apply: Optional[bool] = None

class JobSearchRequest(BaseModel):
    """Job search request model."""
    keywords: str = Field(..., description="Job title or keywords")
    location: str = Field(..., description="Job location")
    filters: Optional[FilterOptions] = Field(default_factory=FilterOptions, description="Search filters")
    max_jobs: Optional[int] = Field(default=20, description="Maximum number of jobs to return")
    headless: Optional[bool] = Field(default=True, description="Run browser in headless mode")

class JobStatusUpdateRequest(BaseModel):
    """Job status update request model."""
    status: JobStatusEnum = Field(..., description="New job status")

class GenerateResumeRequest(BaseModel):
    """Resume generation request model."""
    job_id: str = Field(..., description="ID of the job to generate resume for")
    template: Optional[str] = Field("standard", description="Resume template to use")
    customize: Optional[bool] = Field(True, description="Whether to customize resume for the job")
    resume_data: Optional[Dict[str, Any]] = Field(None, description="User's resume data in YAML format")

class UploadToSimplifyRequest(BaseModel):
    """Resume upload to Simplify request model."""
    job_id: str = Field(..., description="ID of the job")
    resume_id: Optional[str] = None

class SimplifyLoginRequest(BaseModel):
    username: str
    password: str

class SimplifyAPIRequest(BaseModel):
    endpoint: str
    method: str = "GET"
    data: Optional[Dict[str, Any]] = None

class SubmitSimplifySessionRequest(BaseModel):
    session_id: str
    cookies: Dict[str, str]
    user_agent: Optional[str] = None

# In-memory storage for Simplify sessions (use Redis in production)
simplify_sessions: Dict[str, Dict[str, Any]] = {}
user_simplify_sessions: Dict[str, Dict[str, Any]] = {}

class SimplifyJobsIntegration:
    def __init__(self):
        self.site_key = "6LcStf4UAAAAAIVZo9JUJ3PntTfRBhvXLKBTGww8"
        self.base_url = "https://api.simplify.jobs/v2"

    def create_session_for_user(self, user_id: str, username: str) -> str:
        """Create a new Simplify login session for a user"""
        session_id = str(uuid.uuid4())

        simplify_sessions[session_id] = {
            "user_id": user_id,
            "username": username,
            "status": "pending",
            "created_at": datetime.now(),
            "cookies_received": False
        }

        return session_id

    def get_login_page_html(self, session_id: str, username: str) -> str:
        """Generate the login page HTML for manual authentication"""
        return f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Connect to Simplify Jobs</title>
            <style>
                body {{
                    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', sans-serif;
                    max-width: 900px;
                    margin: 0 auto;
                    padding: 20px;
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    min-height: 100vh;
                    color: #333;
                }}
                .container {{
                    background: white;
                    padding: 40px;
                    border-radius: 16px;
                    box-shadow: 0 20px 40px rgba(0,0,0,0.1);
                }}
                .header {{
                    text-align: center;
                    margin-bottom: 30px;
                }}
                .header h1 {{
                    color: #4f46e5;
                    margin-bottom: 10px;
                    font-size: 2rem;
                }}
                .status {{
                    padding: 16px;
                    border-radius: 8px;
                    margin: 20px 0;
                    font-weight: 500;
                }}
                .pending {{ background: #fef3c7; border-left: 4px solid #f59e0b; }}
                .success {{ background: #d1fae5; border-left: 4px solid #10b981; }}
                .error {{ background: #fee2e2; border-left: 4px solid #ef4444; }}
                button {{
                    background: #4f46e5;
                    color: white;
                    border: none;
                    padding: 14px 28px;
                    border-radius: 8px;
                    cursor: pointer;
                    font-size: 16px;
                    font-weight: 600;
                    margin: 10px 5px;
                    transition: all 0.2s;
                    display: inline-flex;
                    align-items: center;
                    gap: 8px;
                }}
                button:hover {{ background: #4338ca; transform: translateY(-1px); }}
                button:disabled {{ 
                    background: #9ca3af; 
                    cursor: not-allowed; 
                    transform: none;
                }}
                .step {{
                    margin: 25px 0;
                    padding: 20px;
                    background: #f8fafc;
                    border-radius: 8px;
                    border: 1px solid #e2e8f0;
                }}
                .step h3 {{
                    margin-top: 0;
                    color: #1e293b;
                    display: flex;
                    align-items: center;
                    gap: 8px;
                }}
                .step-number {{
                    background: #4f46e5;
                    color: white;
                    width: 24px;
                    height: 24px;
                    border-radius: 50%;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    font-size: 14px;
                    font-weight: bold;
                }}
                .info-box {{
                    background: #eff6ff;
                    border: 1px solid #bfdbfe;
                    border-radius: 6px;
                    padding: 12px;
                    margin: 15px 0;
                    font-size: 14px;
                }}
                .success-box {{
                    background: #f0fdf4;
                    border: 1px solid #bbf7d0;
                    color: #166534;
                    text-align: center;
                    padding: 20px;
                    border-radius: 8px;
                    margin: 20px 0;
                }}
                .username-display {{
                    background: #f1f5f9;
                    padding: 8px 12px;
                    border-radius: 4px;
                    font-family: monospace;
                    display: inline-block;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>🔗 Connect to Simplify Jobs</h1>
                    <p>Link your Simplify Jobs account to enable automatic job applications</p>
                    <div class="info-box">
                        <strong>Account:</strong> <span class="username-display">{username}</span><br>
                        <strong>Session:</strong> <span class="username-display">{session_id[:8]}...</span>
                    </div>
                </div>
                
                <div id="status" class="status pending">
                    <strong>Status:</strong> <span id="status-text">Ready to connect</span>
                </div>
                
                <div class="step">
                    <h3><span class="step-number">1</span>Login to Simplify Jobs</h3>
                    <p>Click below to open Simplify Jobs and complete your login manually (including CAPTCHA).</p>
                    <button id="loginBtn" onclick="openLoginTab()">
                        🚀 Open Simplify Jobs Login
                    </button>
                </div>
                
                <div class="step">
                    <h3><span class="step-number">2</span>Capture Session</h3>
                    <p>After successful login, return here and click to capture your session cookies.</p>
                    <button id="captureBtn" onclick="captureSession()" disabled>
                        🍪 Capture Session Data
                    </button>
                </div>
                
                <div class="step">
                    <h3><span class="step-number">3</span>Verify Connection</h3>
                    <p>Test the connection to make sure everything works properly.</p>
                    <button id="verifyBtn" onclick="verifySession()" disabled>
                        ✅ Verify & Complete Setup
                    </button>
                </div>
                
                <div id="results"></div>
            </div>

            <script>
                const sessionId = '{session_id}';
                let loginTabReference = null;

                function updateStatus(text, type = 'pending') {{
                    const statusDiv = document.getElementById('status');
                    const statusText = document.getElementById('status-text');
                    statusText.textContent = text;
                    statusDiv.className = `status ${{type}}`;
                }}

                function openLoginTab() {{
                    updateStatus('Opening Simplify Jobs login...', 'pending');
                    
                    loginTabReference = window.open(
                        'https://simplify.jobs/auth/login', 
                        'simplify_login',
                        'width=1200,height=800,scrollbars=yes,resizable=yes'
                    );
                    
                    setTimeout(() => {{
                        document.getElementById('captureBtn').disabled = false;
                        updateStatus('Complete login in the new tab, then click "Capture Session Data"', 'pending');
                    }}, 2000);
                    
                    const checkClosed = setInterval(() => {{
                        if (loginTabReference && loginTabReference.closed) {{
                            clearInterval(checkClosed);
                            updateStatus('Login tab closed. If login was successful, you can capture the session.', 'pending');
                        }}
                    }}, 1000);
                }}

                async function captureSession() {{
                    updateStatus('Attempting to capture session...', 'pending');
                    
                    const cookieString = prompt(
                        "Please copy and paste your Simplify Jobs cookies:\\n\\n" +
                        "METHOD 1 - Easy way:\\n" +
                        "1. Go to your logged-in Simplify Jobs tab\\n" +
                        "2. Press F12 to open DevTools\\n" +
                        "3. Go to Console tab\\n" +
                        "4. Type: document.cookie\\n" +
                        "5. Copy the result and paste below\\n\\n" +
                        "METHOD 2 - Manual way:\\n" +
                        "1. Open DevTools (F12)\\n" +
                        "2. Go to Application → Cookies → https://simplify.jobs\\n" +
                        "3. Copy all cookie name=value pairs\\n\\n" +
                        "Paste cookies here:"
                    );
                    
                    if (!cookieString) {{
                        updateStatus('Session capture cancelled', 'error');
                        return;
                    }}
                    
                    try {{
                        const response = await fetch('/api/simplify/submit-session', {{
                            method: 'POST',
                            headers: {{
                                'Content-Type': 'application/json',
                            }},
                            body: JSON.stringify({{
                                session_id: sessionId,
                                cookie_string: cookieString.trim(),
                                user_agent: navigator.userAgent
                            }})
                        }});
                        
                        const result = await response.json();
                        
                        if (response.ok) {{
                            updateStatus('Session captured successfully!', 'success');
                            document.getElementById('verifyBtn').disabled = false;
                            document.getElementById('captureBtn').disabled = true;
                        }} else {{
                            throw new Error(result.detail || 'Failed to capture session');
                        }}
                    }} catch (error) {{
                        updateStatus('Failed to capture session: ' + error.message, 'error');
                    }}
                }}

                async function verifySession() {{
                    updateStatus('Verifying connection to Simplify Jobs...', 'pending');
                    
                    try {{
                        const response = await fetch(`/api/simplify/verify-session/${{sessionId}}`);
                        const result = await response.json();
                        
                        if (result.valid) {{
                            updateStatus('✅ Connection verified successfully!', 'success');
                            document.getElementById('results').innerHTML = `
                                <div class="success-box">
                                    <h4>🎉 Simplify Jobs Connected!</h4>
                                    <p>Your account is now linked and ready for automated job applications.</p>
                                    <p><strong>You can close this tab and return to the main application.</strong></p>
                                </div>
                            `;
                            
                            // Notify parent window
                            if (window.opener) {{
                                window.opener.postMessage({{
                                    type: 'SIMPLIFY_LOGIN_SUCCESS',
                                    sessionId: sessionId,
                                    userId: result.user_id
                                }}, '*');
                                
                                // Auto-close after 3 seconds
                                setTimeout(() => {{
                                    window.close();
                                }}, 3000);
                            }}
                        }} else {{
                            updateStatus('❌ Connection verification failed: ' + result.error, 'error');
                        }}
                    }} catch (error) {{
                        updateStatus('❌ Verification error: ' + error.message, 'error');
                    }}
                }}
                
                // Auto-check session status on page load
                window.addEventListener('load', async () => {{
                    try {{
                        const response = await fetch(`/api/simplify/session-status/${{sessionId}}`);
                        const status = await response.json();
                        
                        if (status.cookies_received) {{
                            document.getElementById('captureBtn').disabled = true;
                            document.getElementById('verifyBtn').disabled = false;
                            updateStatus('Session already captured. Ready to verify.', 'success');
                        }}
                    }} catch (error) {{
                        console.log('Could not check session status:', error);
                    }}
                }});
            </script>
        </body>
        </html>
        """