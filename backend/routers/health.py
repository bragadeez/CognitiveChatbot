"""
health.py — Lightweight health check endpoint for monitoring all subsystems.
GET /api/health → returns status of Groq, FAISS, SVM, and Supabase
"""
from fastapi import APIRouter, Depends
from backend.dependencies import get_groq_client, get_vectorstore, get_svm_model, get_supabase_client
from backend.models.schemas import HealthResponse

router = APIRouter(prefix="/api", tags=["health"])


@router.get("/health", response_model=HealthResponse)
async def health(
    groq=Depends(get_groq_client),
    vs=Depends(get_vectorstore),
    svm=Depends(get_svm_model),
    supabase=Depends(get_supabase_client),
):
    return HealthResponse(
        status="ok",
        groq=groq is not None,
        vectorstore=vs is not None,
        svm=svm is not None,
        supabase=supabase is not None,
    )
