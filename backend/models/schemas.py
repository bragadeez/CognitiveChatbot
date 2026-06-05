"""
schemas.py — All Pydantic request/response models for the API.
Strict typing replaces hand-rolled JSON parsing throughout the old app.py.
"""
from __future__ import annotations
from typing import Literal, Optional
from pydantic import BaseModel, Field


# ── Learning Style ──────────────────────────────────────────────────────────

LearningStyle = Literal["Visual", "Auditory", "Reading/Writing", "Kinesthetic"]
VisualMode = Literal["image", "video"]

# ── Chat History ────────────────────────────────────────────────────────────

class HistoryMessage(BaseModel):
    """A single message in the conversation history sent from the frontend."""
    role: Literal["user", "assistant"]
    content: str

# ── Assessment ──────────────────────────────────────────────────────────────

class QuestionResponse(BaseModel):
    question: Optional[str] = None
    index: Optional[int] = None
    total: Optional[int] = None
    done: bool = False
    style: Optional[LearningStyle] = None
    confidence: Optional[float] = None

class AnswerRequest(BaseModel):
    answer: Literal["Strongly Agree", "Agree", "Neutral", "Disagree", "Strongly Disagree"]
    session_id: str = Field(..., description="Browser-unique UUID for session isolation")

class SubmitAnswerResponse(BaseModel):
    status: str = "ok"
    done: bool = False
    style: Optional[LearningStyle] = None
    confidence: Optional[float] = None

# ── LLM Chat ────────────────────────────────────────────────────────────────

class LLMRequest(BaseModel):
    query: str = Field(..., min_length=1)
    style: LearningStyle
    visual_type: VisualMode = "image"
    history: list[HistoryMessage] = Field(
        default_factory=list,
        description="Previous conversation turns (newest last, capped at 10 pairs)"
    )

class ResourceData(BaseModel):
    title: str
    url: str
    description: str

class DiagramData(BaseModel):
    image: str           # base64 data URI
    code: str            # raw Mermaid code
    explanation: str

class VideoData(BaseModel):
    title: str
    url: str
    channel: str
    duration: str = ""
    views: str = ""

class LLMResponse(BaseModel):
    response: Optional[str] = None
    audio: Optional[str] = None         # URL to audio file
    image: Optional[str] = None         # base64 for mindmap
    code: Optional[str] = None          # Mermaid code
    explanation: Optional[str] = None
    resource: Optional[ResourceData] = None
    video_data: Optional[VideoData] = None
    type: Literal["text", "video", "diagram", "audio", "resource"] = "text"
    success: bool = True

# ── Video ────────────────────────────────────────────────────────────────────

class VideoRequest(BaseModel):
    query: str = Field(..., min_length=1)

class VideoResponse(BaseModel):
    title: str
    url: str
    channel: str
    duration: str = ""
    views: str = ""

# ── Image / Diagram ──────────────────────────────────────────────────────────

class ImageRequest(BaseModel):
    prompt: str = Field(..., min_length=1)

class ImageResponse(BaseModel):
    image: str
    code: str
    explanation: str

# ── TTS ──────────────────────────────────────────────────────────────────────

class TTSRequest(BaseModel):
    text: str = Field(..., min_length=1)
    session_id: str = ""

# ── Health ───────────────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status: str = "ok"
    groq: bool = False
    vectorstore: bool = False
    svm: bool = False
    supabase: bool = False
