"""
media.py — Routes for TTS audio and Mermaid diagram generation.
POST /api/generate_image   → Mermaid mindmap PNG (base64)
POST /api/text_to_speech   → audio bytes (MP3) or URL if Supabase configured
"""
from __future__ import annotations
import io
import logging
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import Response

from backend.dependencies import get_groq_client, get_tts_engine, get_supabase_client
from backend.models.schemas import ImageRequest, ImageResponse, TTSRequest
from backend.services.diagram_service import (
    generate_concept_explanation,
    generate_mermaid_code,
    render_mermaid_image,
)
from backend.services.tts_service import generate_speech_gtts
from backend.services.supabase_service import upload_audio
from backend.config import settings

router = APIRouter(prefix="/api", tags=["media"])
logger = logging.getLogger(__name__)


@router.post("/generate_image", response_model=ImageResponse)
async def generate_image_endpoint(
    body: ImageRequest,
    groq_client=Depends(get_groq_client),
):
    """Generate a Mermaid mindmap PNG for the given ML concept prompt."""
    if groq_client is None:
        raise HTTPException(status_code=503, detail="Groq API not available")

    mermaid_code = generate_mermaid_code(body.prompt, groq_client)
    explanation = generate_concept_explanation(body.prompt, groq_client)
    image_b64 = render_mermaid_image(mermaid_code)

    if not image_b64:
        raise HTTPException(status_code=500, detail="Failed to render diagram")

    return ImageResponse(
        image=f"data:image/png;base64,{image_b64}",
        code=mermaid_code,
        explanation=explanation or "No explanation available.",
    )


@router.post("/text_to_speech")
async def text_to_speech(
    body: TTSRequest,
    supabase=Depends(get_supabase_client),
):
    """
    Convert text to speech.
    - If Supabase is available: upload to Storage and return a JSON URL.
    - Otherwise: stream MP3 bytes directly in the response.
    """
    text = body.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="No text provided")

    try:
        audio_bytes = generate_speech_gtts(text)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"TTS failed: {exc}")

    # Attempt Supabase Storage upload
    if supabase:
        public_url = upload_audio(
            supabase,
            bucket=settings.supabase_storage_bucket,
            session_id=body.session_id or "anon",
            audio_bytes=audio_bytes,
        )
        if public_url:
            return {"audio_url": public_url}

    # Fallback: return raw MP3 bytes
    return Response(content=audio_bytes, media_type="audio/mpeg")
