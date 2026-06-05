"""
llm_service.py — Groq LLM wrapper with RAG and conversation history support.

Each learning style handler receives the full conversation history so the
LLM can maintain context across turns. History is sent as a list of
role/content dicts and is prepended to the current prompt.
"""
from __future__ import annotations
import logging
from typing import Optional

from backend.models.schemas import HistoryMessage, LLMResponse, ResourceData
from backend.services.rag_service import get_relevant_context
from backend.services.diagram_service import generate_mermaid_code, generate_concept_explanation, render_mermaid_image
from backend.services.video_service import search_youtube_video
from backend.services.resource_service import search_learning_resource

logger = logging.getLogger(__name__)


def _build_history_messages(history: list[HistoryMessage]) -> list[dict]:
    """Convert frontend history objects into Groq-compatible message dicts."""
    return [{"role": msg.role, "content": msg.content} for msg in history]


# ── Visual ─────────────────────────────────────────────────────────────────

def handle_visual_response(
    query: str,
    visual_type: str,
    groq_client,
    vectorstore,
    history: list[HistoryMessage],
) -> LLMResponse:
    """Visual mode: either YouTube video or Mermaid mindmap diagram."""
    try:
        if visual_type == "video":
            video_data = search_youtube_video(query)
            if video_data:
                return LLMResponse(type="video", video_data=video_data, success=True)
            return LLMResponse(type="video", success=False, response="No relevant videos found.")
        else:
            mermaid_code = generate_mermaid_code(query, groq_client)
            explanation = generate_concept_explanation(query, groq_client)
            image_b64 = render_mermaid_image(mermaid_code)
            if not image_b64:
                return LLMResponse(type="diagram", success=False, response="Failed to render diagram.")
            return LLMResponse(
                type="diagram",
                image=f"data:image/png;base64,{image_b64}",
                code=mermaid_code,
                explanation=explanation or "No explanation available.",
                success=True,
            )
    except Exception as exc:
        logger.error("Error in handle_visual_response: %s", exc)
        return LLMResponse(type="diagram", success=False, response=str(exc))


# ── Auditory ────────────────────────────────────────────────────────────────

def handle_auditory_response(
    query: str,
    groq_client,
    vectorstore,
    history: list[HistoryMessage],
) -> LLMResponse:
    """Auditory mode: concise spoken-style text + TTS audio URL.
    History is included so the LLM knows what was already explained."""
    try:
        system_msg = (
            "You are an expert ML teacher for auditory learners. "
            "Explain the concept in 2–4 sentences using simple, conversational language. "
            "Include one brief analogy. Avoid technical jargon. "
            "If this is a follow-up question, build on what was previously discussed."
        )
        messages = [{"role": "system", "content": system_msg}]
        messages.extend(_build_history_messages(history))
        messages.append({"role": "user", "content": query})

        response_text = groq_client.chat.completions.create(
            messages=messages,
            model="llama-3.3-70b-versatile",
            max_tokens=200,
            temperature=0.3,
        ).choices[0].message.content.strip()

        return LLMResponse(type="audio", response=response_text, success=True)
    except Exception as exc:
        logger.error("Error in handle_auditory_response: %s", exc)
        return LLMResponse(type="audio", success=False, response=str(exc))


# ── Reading / Writing ────────────────────────────────────────────────────────

def handle_reading_response(
    query: str,
    groq_client,
    vectorstore,
    history: list[HistoryMessage],
) -> LLMResponse:
    """Reading/Writing mode: rich structured markdown with RAG context.
    History preserved so user can ask follow-ups on the same topic."""
    try:
        context = get_relevant_context(vectorstore, query)
        
        context_block = f"\n\nRelevant textbook excerpts:\n{context}\n" if context else ""
        
        system_msg = (
            "You are an expert ML teacher for reading/writing learners. "
            "Provide a well-structured, detailed response using markdown formatting. "
            "If this is a follow-up question, reference what was previously discussed."
        )
        user_prompt = (
            f"{context_block}"
            f"Explain the following in a clear, structured way with these sections:\n"
            f"**Definition** · **Key Points** · **Advantages** · **Disadvantages** · **Example**\n\n"
            f"Question: {query}"
        )

        messages = [{"role": "system", "content": system_msg}]
        messages.extend(_build_history_messages(history))
        messages.append({"role": "user", "content": user_prompt})

        response_text = groq_client.chat.completions.create(
            messages=messages,
            model="llama-3.3-70b-versatile",
            max_tokens=600,
            temperature=0.3,
        ).choices[0].message.content.strip()

        return LLMResponse(type="text", response=response_text, success=True)
    except Exception as exc:
        logger.error("Error in handle_reading_response: %s", exc)
        return LLMResponse(type="text", success=False, response=str(exc))


# ── Kinesthetic ──────────────────────────────────────────────────────────────

def handle_kinesthetic_response(
    query: str,
    groq_client,
    vectorstore,
    history: list[HistoryMessage],
) -> LLMResponse:
    """Kinesthetic mode: hands-on practical explanation + GeeksforGeeks resource card."""
    try:
        system_msg = (
            "You are an expert ML teacher for kinesthetic learners. "
            "Explain concepts using real-world analogies and practical, hands-on examples. "
            "Focus on 'doing' and 'experiencing'. "
            "If this is a follow-up, build naturally on the previous exchange."
        )
        user_prompt = (
            f"Give a hands-on, practical explanation:\n"
            f"1. Real-world analogy or scenario (2–3 sentences)\n"
            f"2. Practical experiment or example to try (2–3 sentences)\n"
            f"3. Key takeaway in simple terms (1–2 sentences)\n\n"
            f"Question: {query}"
        )

        messages = [{"role": "system", "content": system_msg}]
        messages.extend(_build_history_messages(history))
        messages.append({"role": "user", "content": user_prompt})

        response_text = groq_client.chat.completions.create(
            messages=messages,
            model="llama-3.3-70b-versatile",
            max_tokens=350,
            temperature=0.5,
        ).choices[0].message.content.strip()

        resource_dict = search_learning_resource(query, groq_client)
        resource = ResourceData(**resource_dict) if resource_dict else None

        return LLMResponse(type="resource", response=response_text, resource=resource, success=True)
    except Exception as exc:
        logger.error("Error in handle_kinesthetic_response: %s", exc)
        return LLMResponse(type="resource", success=False, response=str(exc))
