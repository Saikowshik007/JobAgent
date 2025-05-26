"""
Simplify.jobs integration routes for session management and resume uploads.
"""
from fastapi import APIRouter, Depends, HTTPException, Form, File, UploadFile
from datetime import datetime
import requests
import logging

from core.dependencies import get_user_id

logger = logging.getLogger(__name__)
router = APIRouter()

# Global user sessions storage (in production, consider using Redis or database)
user_sessions = {}

@router.post("/upload-resume-pdf")
async def upload_resume_pdf_to_simplify(
        resume_pdf: UploadFile = File(...),
        resume_id: str = Form(...),
        job_id: str = Form(None),
        user_id: str = Depends(get_user_id)
):
    """Upload a PDF resume to Simplify using stored session data"""
    try:
        # Check if user has session data
        if user_id not in user_sessions:
            raise HTTPException(
                status_code=400,
                detail="No Simplify session found. Please complete the session capture step first."
            )

        session = user_sessions[user_id]

        # Validate the uploaded file
        if not resume_pdf.filename.lower().endswith('.pdf'):
            raise HTTPException(status_code=400, detail="Only PDF files are supported")

        # Read the PDF content
        pdf_content = await resume_pdf.read()

        if len(pdf_content) == 0:
            raise HTTPException(status_code=400, detail="PDF file is empty")

        logger.info(f"Received PDF file: {resume_pdf.filename}, size: {len(pdf_content)} bytes")

        # Prepare headers to match the curl command exactly
        headers = {
            'sec-ch-ua-platform': '"Windows"',
            'X-CSRF-TOKEN': session['csrf_token'],
            'Referer': 'https://simplify.jobs/',
            'sec-ch-ua': '"Chromium";v="136", "Google Chrome";v="136", "Not.A/Brand";v="99"',
            'sec-ch-ua-mobile': '?0',
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/136.0.0.0 Safari/537.36',
            'DNT': '1',
            'accept': '*/*',
            'accept-language': 'en-US,en;q=0.9',
            'origin': 'https://simplify.jobs',
            'sec-fetch-dest': 'empty',
            'sec-fetch-mode': 'cors',
            'sec-fetch-site': 'same-site'
        }

        # Add optional headers if available in session
        if session.get('baggage'):
            headers['baggage'] = session['baggage']
        if session.get('sentry_trace'):
            headers['sentry-trace'] = session['sentry_trace']

        logger.info(f"CSRF Token: {session['csrf_token'][:20]}...")
        cookies = {
            'authorization': session['authorization'],
            'csrf': session['csrf_token']
        }

        # Prepare the file for upload
        files = {
            'file': (resume_pdf.filename, pdf_content, 'application/pdf')
        }

        # Make the request to Simplify's API
        response = requests.post(
            'https://api.simplify.jobs/v2/candidate/me/resume/upload',
            files=files,
            headers=headers,
            cookies=cookies,
            timeout=30
        )

        logger.info(f"Simplify API response: {response.status_code}")
        logger.info(f"Response headers: {dict(response.headers)}")

        if response.status_code == 201:
            try:
                response_data = response.json()
                return {
                    "message": "Resume PDF uploaded successfully to Simplify",
                    "data": response_data,
                    "resume_id": resume_id,
                    "job_id": job_id,
                    "pdf_size": len(pdf_content)
                }
            except:
                return {
                    "message": "Resume PDF uploaded successfully to Simplify",
                    "resume_id": resume_id,
                    "job_id": job_id,
                    "pdf_size": len(pdf_content),
                    "response_text": response.text[:500]  # First 500 chars of response
                }
        else:
            logger.error(f"Simplify upload failed: {response.status_code} - {response.text}")
            raise HTTPException(
                status_code=response.status_code,
                detail=f"Simplify upload failed: {response.text}"
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error uploading PDF resume to Simplify: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/get-tokens")
async def get_simplify_tokens(user_id: str = Depends(get_user_id)):
    """Get stored Simplify tokens for the user"""
    try:
        if user_id not in user_sessions:
            raise HTTPException(
                status_code=404,
                detail="No stored tokens found. Please capture tokens first."
            )

        session = user_sessions[user_id]

        # Return only the necessary tokens (don't expose everything)
        return {
            "authorization": session.get('authorization'),
            "csrf": session.get('csrf_token'),
            "has_tokens": bool(session.get('authorization') and session.get('csrf_token')),
            "stored_at": session.get('stored_at').isoformat() if session.get('stored_at') else None
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting stored tokens for user {user_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/check-session")
async def check_simplify_session(user_id: str = Depends(get_user_id)):
    """Check if user has a valid Simplify session and validate tokens"""
    has_session = user_id in user_sessions
    session_age = None
    is_valid = False

    if has_session:
        session = user_sessions[user_id]
        stored_at = session['stored_at']
        session_age = (datetime.now() - stored_at).total_seconds() / 3600  # hours

        # Validate tokens with Simplify API
        try:
            logger.info(f"Validating Simplify tokens for user {user_id}")

            # Prepare headers exactly like the curl command
            headers = {
                'accept': '*/*',
                'accept-language': 'en-US,en;q=0.9',
                'client': 'Dunder',
                'content-length': '0',
                'content-type': 'application/json',
                'dnt': '1',
                'origin': 'https://simplify.jobs',
                'priority': 'u=1, i',
                'referer': 'https://simplify.jobs/',
                'sec-ch-ua': '"Chromium";v="136", "Google Chrome";v="136", "Not.A/Brand";v="99"',
                'sec-ch-ua-mobile': '?0',
                'sec-ch-ua-platform': '"Windows"',
                'sec-fetch-dest': 'empty',
                'sec-fetch-mode': 'cors',
                'sec-fetch-site': 'same-site',
                'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/136.0.0.0 Safari/537.36',
                'x-csrf-token': session.get('csrf_token', '')
            }

            # Add optional headers if available
            if session.get('baggage'):
                headers['baggage'] = session['baggage']
            if session.get('sentry_trace'):
                headers['sentry-trace'] = session['sentry_trace']

            # Prepare cookies
            cookies = {
                'authorization': session.get('authorization', ''),
                'csrf': session.get('csrf_token', '')
            }

            # Add any additional cookies if they were stored
            if session.get('raw_cookies'):
                try:
                    raw_cookies = session['raw_cookies']
                    for cookie in raw_cookies.split(';'):
                        if '=' in cookie:
                            name, value = cookie.strip().split('=', 1)
                            if name not in cookies:  # Don't override auth/csrf
                                cookies[name] = value
                except Exception as e:
                    logger.warning(f"Failed to parse raw cookies: {e}")

            # Make validation request to Simplify
            response = requests.post(
                'https://api.simplify.jobs/v2/auth/validate',
                headers=headers,
                cookies=cookies,
                timeout=10
            )

            logger.info(f"Simplify auth validation response: {response.status_code}")

            if response.status_code == 200:
                is_valid = True
                logger.info(f"✅ Tokens are valid for user {user_id}")
            else:
                logger.warning(f"❌ Tokens are invalid for user {user_id}: {response.status_code} - {response.text}")
                # Discard invalid session
                del user_sessions[user_id]
                has_session = False
                session_age = None

        except requests.exceptions.RequestException as e:
            logger.error(f"❌ Failed to validate tokens for user {user_id}: {e}")
            # Network error - keep session but mark as unvalidated
            is_valid = False

        except Exception as e:
            logger.error(f"❌ Unexpected error validating tokens for user {user_id}: {e}")
            # Unexpected error - discard session to be safe
            if user_id in user_sessions:
                del user_sessions[user_id]
            has_session = False
            session_age = None
            is_valid = False

    return {
        "has_session": has_session,
        "session_age_hours": session_age,
        "is_valid": is_valid,
        "message": "Session validated with Simplify API" if is_valid else "Session invalid or validation failed"
    }

@router.post("/store-tokens")
async def store_simplify_tokens(
        request_data: dict,
        user_id: str = Depends(get_user_id)
):
    """Store both CSRF and authorization tokens for Simplify with immediate validation"""
    try:
        csrf_token = request_data.get('csrf')
        auth_token = request_data.get('authorization')

        if not csrf_token or not auth_token:
            raise HTTPException(
                status_code=400,
                detail="Both 'csrf' and 'authorization' tokens are required"
            )

        # Store the tokens temporarily for validation
        temp_session = {
            'authorization': auth_token,
            'csrf_token': csrf_token,
            'stored_at': datetime.now(),
            'user_id': user_id,
            'capture_method': 'manual_tokens'
        }

        # Validate tokens immediately before storing permanently
        try:
            logger.info(f"Validating provided tokens for user {user_id}")

            headers = {
                'accept': '*/*',
                'accept-language': 'en-US,en;q=0.9',
                'client': 'Dunder',
                'content-length': '0',
                'content-type': 'application/json',
                'dnt': '1',
                'origin': 'https://simplify.jobs',
                'referer': 'https://simplify.jobs/',
                'sec-ch-ua': '"Chromium";v="136", "Google Chrome";v="136", "Not.A/Brand";v="99"',
                'sec-ch-ua-mobile': '?0',
                'sec-ch-ua-platform': '"Windows"',
                'sec-fetch-dest': 'empty',
                'sec-fetch-mode': 'cors',
                'sec-fetch-site': 'same-site',
                'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/136.0.0.0 Safari/537.36',
                'x-csrf-token': csrf_token
            }

            cookies = {
                'authorization': auth_token,
                'csrf': csrf_token
            }

            response = requests.post(
                'https://api.simplify.jobs/v2/auth/validate',
                headers=headers,
                cookies=cookies,
                timeout=10
            )

            if response.status_code != 200:
                logger.warning(f"Token validation failed: {response.status_code} - {response.text}")
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid tokens: Simplify API returned {response.status_code}. Please check your tokens and try again."
                )

            logger.info(f"✅ Tokens validated successfully for user {user_id}")

        except requests.exceptions.RequestException as e:
            logger.error(f"Network error during token validation: {e}")
            raise HTTPException(
                status_code=503,
                detail="Unable to validate tokens due to network error. Please try again later."
            )

        # Tokens are valid, store them permanently
        user_sessions[user_id] = temp_session

        logger.info(f"Stored validated Simplify tokens for user {user_id}")
        return {
            "message": "Tokens validated and stored successfully",
            "validated_at": datetime.now().isoformat()
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error storing tokens: {e}")
        raise HTTPException(status_code=500, detail=str(e))