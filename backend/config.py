"""
config.py — Centralised settings for the Cognitive Chatbot backend.
All environment variables are defined here via pydantic-settings,
so every part of the app imports from one place.
"""
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=Path(__file__).parent.parent / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ── Groq ────────────────────────────────────────────────────────────────
    api_key: str
    groq_model: str = "llama-3.3-70b-versatile"

    # ── Supabase ─────────────────────────────────────────────────────────────
    supabase_url: str = ""
    supabase_anon_key: str = ""
    supabase_storage_bucket: str = "tts-audio"

    # ── ML model paths (relative to project root) ────────────────────────────
    svm_model_path: Path = Path(__file__).parent.parent / "models" / "svm_LS_model.pkl"
    svm_scaler_path: Path = Path(__file__).parent.parent / "models" / "svm_scaler.pkl"
    vectorstore_path: Path = Path(__file__).parent.parent / "vectorstore"

    # ── Data paths ────────────────────────────────────────────────────────────
    questionnaire_path: Path = Path(__file__).parent.parent / "data" / "questionnaire.json"

    # ── Embeddings model ──────────────────────────────────────────────────────
    embedding_model: str = "sentence-transformers/all-mpnet-base-v2"

    # ── TTS ───────────────────────────────────────────────────────────────────
    coqui_model: str = "tts_models/en/ljspeech/tacotron2-DDC"
    tts_output_dir: Path = Path(__file__).parent.parent / "static"

    # ── LLM ───────────────────────────────────────────────────────────────────
    max_chat_history: int = 10  # max message pairs kept in context
    rag_top_k: int = 3


# Singleton — import this everywhere
settings = Settings()
