"""
dependencies.py — FastAPI dependency providers for heavy singletons.
All expensive resources (Groq client, FAISS vectorstore, SVM model) are
loaded ONCE on app startup via the lifespan context in main.py, then
injected into route handlers via Depends().
"""
from __future__ import annotations
from typing import Optional
import logging

logger = logging.getLogger(__name__)

# Singleton holders — populated during app lifespan startup
_groq_client = None
_vectorstore = None
_svm_model = None
_svm_scaler = None
_tts_engine = None
_supabase_client = None


def set_groq_client(client) -> None:
    global _groq_client
    _groq_client = client


def set_vectorstore(vs) -> None:
    global _vectorstore
    _vectorstore = vs


def set_svm_model(model, scaler) -> None:
    global _svm_model, _svm_scaler
    _svm_model = model
    _svm_scaler = scaler


def set_tts_engine(engine) -> None:
    global _tts_engine
    _tts_engine = engine


def set_supabase_client(client) -> None:
    global _supabase_client
    _supabase_client = client


# ── FastAPI dependency functions ─────────────────────────────────────────────

def get_groq_client():
    return _groq_client


def get_vectorstore():
    return _vectorstore


def get_svm_model():
    return _svm_model


def get_svm_scaler():
    return _svm_scaler


def get_tts_engine():
    return _tts_engine


def get_supabase_client():
    return _supabase_client
