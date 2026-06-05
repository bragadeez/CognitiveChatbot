"""
chat.py — Routes for the learning-style chatbot.
POST /api/llm_response   → dispatches to the appropriate style handler
POST /api/get_video      → YouTube video search (Visual/video mode)
"""
from __future__ import annotations
import logging
from fastapi import APIRouter, Depends, HTTPException

from backend.dependencies import get_groq_client, get_vectorstore
from backend.models.schemas import LLMRequest, LLMResponse, VideoRequest, VideoResponse
from backend.services import llm_service
from backend.services.video_service import search_youtube_video

router = APIRouter(prefix="/api", tags=["chat"])
logger = logging.getLogger(__name__)


@router.post("/llm_response", response_model=LLMResponse)
async def llm_response(
    body: LLMRequest,
    groq_client=Depends(get_groq_client),
    vectorstore=Depends(get_vectorstore),
):
    """
    Dispatch to the appropriate learning-style handler.
    The `history` field carries previous turns for context preservation.
    History is capped on the frontend; the backend trusts whatever is sent.
    """
    if groq_client is None:
        raise HTTPException(status_code=503, detail="Groq API not available")

    # Trim history to last 10 pairs (safety cap, frontend should already do this)
    trimmed_history = body.history[-20:] if len(body.history) > 20 else body.history

    style = body.style.lower()
    if style == "visual":
        result = llm_service.handle_visual_response(
            body.query, body.visual_type, groq_client, vectorstore, trimmed_history
        )
    elif style == "auditory":
        result = llm_service.handle_auditory_response(
            body.query, groq_client, vectorstore, trimmed_history
        )
    elif style == "reading/writing":
        result = llm_service.handle_reading_response(
            body.query, groq_client, vectorstore, trimmed_history
        )
    elif style == "kinesthetic":
        result = llm_service.handle_kinesthetic_response(
            body.query, groq_client, vectorstore, trimmed_history
        )
    else:
        raise HTTPException(status_code=400, detail=f"Invalid learning style: {body.style}")

    return result


@router.post("/get_video", response_model=VideoResponse)
async def get_video(body: VideoRequest):
    """Dedicated endpoint for YouTube video search (Visual/video sub-mode)."""
    video = search_youtube_video(body.query)
    if not video:
        raise HTTPException(status_code=404, detail="No relevant videos found")
    return VideoResponse(**video.model_dump())
