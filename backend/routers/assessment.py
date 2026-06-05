"""
assessment.py — Routes for the VARK learning style quiz.
GET  /api/next_question   → returns the next question or done signal
POST /api/submit_answer   → accumulates answers; on last question, predicts style
GET  /api/after_result    → returns style + confidence for result page
"""
from __future__ import annotations
import logging
from fastapi import APIRouter, Depends, Query, HTTPException
from fastapi.responses import JSONResponse

from backend.dependencies import get_svm_model, get_svm_scaler, get_supabase_client
from backend.models.schemas import (
    AnswerRequest,
    QuestionResponse,
    SubmitAnswerResponse,
)
from backend.services.ml_service import QUESTIONS, SCALE, predict_style
from backend.services.supabase_service import (
    add_response,
    clear_responses,
    get_responses,
    save_assessment_result,
)

router = APIRouter(prefix="/api", tags=["assessment"])
logger = logging.getLogger(__name__)


@router.get("/next_question", response_model=QuestionResponse)
async def next_question(index: int = Query(0, ge=0)):
    """Return the question at the given index, or signal completion."""
    if index < len(QUESTIONS):
        return QuestionResponse(
            question=QUESTIONS[index],
            index=index + 1,
            total=len(QUESTIONS),
            done=False,
        )
    return QuestionResponse(done=True)


@router.post("/submit_answer", response_model=SubmitAnswerResponse)
async def submit_answer(
    body: AnswerRequest,
    svm_model=Depends(get_svm_model),
    svm_scaler=Depends(get_svm_scaler),
    supabase=Depends(get_supabase_client),
):
    """
    Record a single Likert answer for the given session.
    When all 15 answers are collected, runs the SVM prediction.
    """
    if svm_model is None or svm_scaler is None:
        raise HTTPException(status_code=503, detail="SVM model not loaded")

    numeric_answer = SCALE.get(body.answer)
    if numeric_answer is None:
        raise HTTPException(status_code=400, detail=f"Invalid answer: {body.answer}")

    responses = add_response(body.session_id, numeric_answer)
    logger.debug("Session %s: %d/%d responses", body.session_id, len(responses), len(QUESTIONS))

    if len(responses) == len(QUESTIONS):
        style, confidence = predict_style(responses, svm_model, svm_scaler)
        clear_responses(body.session_id)
        save_assessment_result(supabase, body.session_id, style, confidence)
        return SubmitAnswerResponse(done=True, style=style, confidence=confidence)

    return SubmitAnswerResponse(status="ok", done=False)


@router.get("/after_result")
async def after_result(style: str = Query(...), confidence: float = Query(...)):
    """Simple passthrough used for server-side rendering (kept for compat)."""
    return {"style": style, "confidence": confidence}
