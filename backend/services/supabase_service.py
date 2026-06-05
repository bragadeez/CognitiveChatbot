"""
supabase_service.py — Supabase Auth, Database, and Storage integration.

- Anonymous auth: each browser gets a unique session_id UUID
- DB: stores assessment results per session
- Storage: hosts per-session TTS audio files
"""
from __future__ import annotations
import logging
import uuid
from typing import Optional

logger = logging.getLogger(__name__)

# In-memory session store for quiz responses: { session_id: [int, ...] }
# Used as fallback if Supabase is unavailable; also used during the quiz
# (responses accumulate here before final DB write on completion)
_session_responses: dict[str, list[int]] = {}


def load_supabase_client(url: str, anon_key: str):
    """Create and return a Supabase client, or None if credentials missing."""
    if not url or not anon_key:
        logger.warning("Supabase credentials not set — running without Supabase.")
        return None
    try:
        from supabase import create_client
        client = create_client(url, anon_key)
        logger.info("✓ Supabase client connected")
        return client
    except Exception as exc:
        logger.error("❌ Error connecting to Supabase: %s", exc)
        return None


def create_session() -> str:
    """Generate a new unique session UUID."""
    return str(uuid.uuid4())


# ── In-memory quiz response management ───────────────────────────────────────

def add_response(session_id: str, value: int) -> list[int]:
    """Append a Likert response to the session's in-memory list."""
    if session_id not in _session_responses:
        _session_responses[session_id] = []
    _session_responses[session_id].append(value)
    return _session_responses[session_id]


def get_responses(session_id: str) -> list[int]:
    """Return all responses accumulated so far for a session."""
    return _session_responses.get(session_id, [])


def clear_responses(session_id: str) -> None:
    """Remove the session's responses after prediction."""
    _session_responses.pop(session_id, None)


# ── Supabase DB ───────────────────────────────────────────────────────────────

def save_assessment_result(
    supabase_client,
    session_id: str,
    style: str,
    confidence: float,
) -> bool:
    """Persist the assessment result to Supabase `assessment_results` table."""
    if supabase_client is None:
        return False
    try:
        supabase_client.table("assessment_results").insert({
            "session_id": session_id,
            "style": style,
            "confidence": confidence,
        }).execute()
        return True
    except Exception as exc:
        logger.error("Error saving assessment result to Supabase: %s", exc)
        return False


# ── Supabase Storage ──────────────────────────────────────────────────────────

def upload_audio(
    supabase_client,
    bucket: str,
    session_id: str,
    audio_bytes: bytes,
    mime_type: str = "audio/mpeg",
) -> Optional[str]:
    """
    Upload audio bytes to Supabase Storage and return a public URL.
    Returns None if Supabase is unavailable.
    """
    if supabase_client is None:
        return None
    try:
        path = f"{session_id}/response.mp3"
        supabase_client.storage.from_(bucket).upload(
            path, audio_bytes, {"content-type": mime_type, "upsert": "true"}
        )
        public_url = supabase_client.storage.from_(bucket).get_public_url(path)
        return public_url
    except Exception as exc:
        logger.error("Error uploading audio to Supabase Storage: %s", exc)
        return None
