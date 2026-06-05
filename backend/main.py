"""
main.py — FastAPI application entry point.

Startup lifecycle:
  1. Load settings from .env via config.py
  2. Connect to Groq (test ping)
  3. Load FAISS vectorstore
  4. Load SVM model + scaler
  5. Initialise Coqui TTS (optional)
  6. Connect to Supabase (optional)
  7. Mount all routers
  8. Serve static frontend build (or dev proxy via Vite)

Run with:
    uvicorn backend.main:app --reload --port 8000
"""
from __future__ import annotations
import logging
import warnings
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pathlib import Path

from backend.config import settings
from backend import dependencies
from backend.services.rag_service import load_vectorstore
from backend.services.ml_service import load_svm_model
from backend.services.tts_service import load_coqui_tts
from backend.services.supabase_service import load_supabase_client
from backend.routers import assessment, chat, media, health

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logging.getLogger("transformers").setLevel(logging.ERROR)
warnings.simplefilter("ignore", FutureWarning)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load all heavy resources once on startup; clean up on shutdown."""
    logger.info("🚀 Starting Cognitive Chatbot backend...")

    # ── Groq ─────────────────────────────────────────────────────────────────
    try:
        from groq import Groq
        client = Groq(api_key=settings.api_key)
        # Minimal connectivity test
        client.chat.completions.create(
            messages=[{"role": "user", "content": "Hi"}],
            model=settings.groq_model,
            max_tokens=5,
        )
        dependencies.set_groq_client(client)
        logger.info("✓ Groq API connected")
    except Exception as exc:
        logger.error("❌ Groq connection failed: %s", exc)

    # ── FAISS Vectorstore ─────────────────────────────────────────────────────
    vs = load_vectorstore(settings.vectorstore_path, settings.embedding_model)
    dependencies.set_vectorstore(vs)

    # ── SVM Model ─────────────────────────────────────────────────────────────
    model, scaler = load_svm_model(settings.svm_model_path, settings.svm_scaler_path)
    dependencies.set_svm_model(model, scaler)

    # ── Coqui TTS (optional) ──────────────────────────────────────────────────
    tts_engine = load_coqui_tts(settings.coqui_model)
    dependencies.set_tts_engine(tts_engine)

    # ── Supabase ──────────────────────────────────────────────────────────────
    sb = load_supabase_client(settings.supabase_url, settings.supabase_anon_key)
    dependencies.set_supabase_client(sb)

    logger.info("✅ All services initialised — server ready.")
    yield
    logger.info("🛑 Shutting down...")


# ── FastAPI App ───────────────────────────────────────────────────────────────

app = FastAPI(
    title="Cognitive Chatbot API",
    description="VARK learning-style assessment + personalised ML tutor chatbot",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── API Routers ───────────────────────────────────────────────────────────────
app.include_router(assessment.router)
app.include_router(chat.router)
app.include_router(media.router)
app.include_router(health.router)

# ── Serve React frontend (production build) ───────────────────────────────────
_frontend_dist = Path(__file__).parent.parent / "frontend" / "dist"
if _frontend_dist.exists():
    app.mount("/assets", StaticFiles(directory=str(_frontend_dist / "assets")), name="assets")

    @app.get("/{full_path:path}", include_in_schema=False)
    async def serve_spa(full_path: str):
        index = _frontend_dist / "index.html"
        return FileResponse(str(index))
