"""Convert an uploaded text-based PDF resume into JobAgent's canonical resume data."""
from io import BytesIO
from pathlib import Path
from typing import Any

from pypdf import PdfReader

from dataModels.resume import SourceResumeExtractionOutput
from services.langchain_helpers import invoke_structured


MAX_SOURCE_PDF_BYTES = 8 * 1024 * 1024
SAMPLE_RESUME_PATH = Path(__file__).resolve().parent.parent / "prompts" / "sample-resume.yaml"


def extract_pdf_text(pdf_content: bytes) -> str:
    """Extract text from a text-based PDF without storing the source document."""
    if not pdf_content:
        raise ValueError("The PDF is empty")
    if len(pdf_content) > MAX_SOURCE_PDF_BYTES:
        raise ValueError("The PDF exceeds the 8 MB upload limit")
    try:
        reader = PdfReader(BytesIO(pdf_content))
        text = "\n".join(page.extract_text() or "" for page in reader.pages).strip()
    except Exception as error:
        raise ValueError("The uploaded file is not a readable PDF") from error
    if len(text) < 40:
        raise ValueError("No readable text was found. Upload a text-based PDF, not a scanned image.")
    return text


def parse_source_resume(pdf_content: bytes, user, progress_callback=None) -> dict[str, Any]:
    """Use structured generation to produce the canonical editable resume object."""
    source_text = extract_pdf_text(pdf_content)
    if progress_callback:
        progress_callback("structuring", 55, "Mapping resume text into editable fields")
    try:
        schema_example = SAMPLE_RESUME_PATH.read_text(encoding="utf-8")
    except OSError as error:
        raise RuntimeError("The bundled sample resume is unavailable") from error
    result = invoke_structured(
        user,
        "SOURCE_RESUME_PARSER",
        SourceResumeExtractionOutput,
        timeout_seconds=120.0,
        max_retries=1,
        source_resume_text=source_text,
        resume_schema_example=schema_example,
    )
    if progress_callback:
        progress_callback("validating", 85, "Validating extracted resume fields")
    resume_data = result.final_answer.model_dump()
    return _normalise_resume_data(resume_data)


def _normalise_resume_data(data: dict[str, Any]) -> dict[str, Any]:
    """Ensure optional schema fields exist before the client renders or saves the resume."""
    basic = data.get("basic") if isinstance(data.get("basic"), dict) else {}
    basic = {
        "name": str(basic.get("name") or ""),
        "address": str(basic.get("address") or ""),
        "email": str(basic.get("email") or ""),
        "phone": str(basic.get("phone") or ""),
        "websites": [str(item) for item in basic.get("websites", []) if item],
    }
    return {
        "basic": basic,
        "objective": str(data.get("objective") or ""),
        "education": data.get("education") if isinstance(data.get("education"), list) else [],
        "experiences": data.get("experiences") if isinstance(data.get("experiences"), list) else [],
        "projects": data.get("projects") if isinstance(data.get("projects"), list) else [],
        "skills": data.get("skills") if isinstance(data.get("skills"), list) else [],
    }
