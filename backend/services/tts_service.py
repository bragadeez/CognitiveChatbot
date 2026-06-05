"""
tts_service.py — Text-to-speech using gTTS (primary) and Coqui TTS (high-quality).
Provides two backends; callers choose based on quality vs. speed needs.
"""
from __future__ import annotations
import io
import logging
import uuid
from pathlib import Path
from typing import Optional, Tuple

logger = logging.getLogger(__name__)


def load_coqui_tts(model_name: str):
    """Load Coqui TTS model; returns None if unavailable."""
    try:
        from TTS.api import TTS
        engine = TTS(model_name=model_name)
        logger.info("✓ Coqui TTS model loaded: %s", model_name)
        return engine
    except Exception as exc:
        logger.warning("Coqui TTS unavailable (%s), will fall back to gTTS.", exc)
        return None


def generate_speech_gtts(text: str) -> bytes:
    """Generate MP3 audio bytes using gTTS (fast, requires internet)."""
    from gtts import gTTS
    mp3_fp = io.BytesIO()
    tts = gTTS(text=text, lang="en")
    tts.write_to_fp(mp3_fp)
    mp3_fp.seek(0)
    return mp3_fp.read()


def generate_speech_coqui(text: str, coqui_engine, output_dir: Path, session_id: str = "") -> Optional[str]:
    """
    Generate a WAV file with Coqui TTS and return its path.
    Uses session_id in the filename to avoid concurrent-user file collisions.
    """
    try:
        cleaned = text.replace("\n", " ").strip()
        if len(cleaned) > 500:
            cleaned = cleaned[:500] + "..."
        fname = f"speech_{session_id or uuid.uuid4().hex[:8]}.wav"
        output_path = output_dir / fname
        coqui_engine.tts_to_file(text=cleaned, file_path=str(output_path))
        return str(output_path)
    except Exception as exc:
        logger.error("Error in Coqui TTS generation: %s", exc)
        return None
