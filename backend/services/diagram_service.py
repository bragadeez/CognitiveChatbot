"""
diagram_service.py — Mermaid mindmap generation and rendering via mermaid.ink.
Extracted from app.py; uses Groq to generate valid Mermaid syntax,
then fetches a rendered PNG from the mermaid.ink API.
"""
from __future__ import annotations
import base64
import logging
import re
from typing import Optional

import requests

logger = logging.getLogger(__name__)

_FALLBACK_DIAGRAM = """mindmap
    root((Topic))
        concept1
            detail1
            detail2
        concept2
            detail3
            detail4"""

_SYSTEM_MSG = """You are an expert at creating Mermaid mindmap diagrams.
Follow this exact format:

mindmap
    root((Main Topic))
        key1
            subkey1
            subkey2
        key2
            subkey3
            subkey4

Rules:
- Start with 'mindmap'
- Use 4 spaces for indentation
- Use double parentheses only for root node
- Keep node text simple and short
- No special characters except ()
- No empty lines between nodes"""


def generate_mermaid_code(prompt: str, groq_client) -> str:
    """Generate a Mermaid mindmap diagram for an ML concept."""
    try:
        completion = groq_client.chat.completions.create(
            messages=[
                {"role": "system", "content": _SYSTEM_MSG},
                {"role": "user", "content": f"Create a mindmap diagram explaining this ML concept: {prompt}"},
            ],
            model="llama-3.3-70b-versatile",
            max_tokens=400,
            temperature=0.2,
        )
        raw = completion.choices[0].message.content.strip()
        return _clean_mermaid(raw)
    except Exception as exc:
        logger.error("Error generating Mermaid code: %s", exc)
        return _FALLBACK_DIAGRAM


def _clean_mermaid(code: str) -> str:
    """Normalize indentation and strip special characters from Mermaid code."""
    if not code.startswith("mindmap"):
        code = "mindmap\n" + code

    lines = code.split("\n")
    cleaned: list[str] = []
    for line in lines:
        indent = len(line) - len(line.lstrip())
        content = "".join(c for c in line.strip() if c.isalnum() or c in " ()")
        if content:
            cleaned.append(" " * indent + content)
    return "\n".join(cleaned)


def generate_concept_explanation(prompt: str, groq_client) -> Optional[str]:
    """Generate a brief 2–3 sentence explanation of an ML concept."""
    try:
        completion = groq_client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are an ML expert. Provide a brief, clear explanation in 2–3 sentences."},
                {"role": "user", "content": f"Explain this ML concept briefly: {prompt}"},
            ],
            model="llama-3.3-70b-versatile",
            max_tokens=150,
            temperature=0.3,
        )
        return completion.choices[0].message.content.strip()
    except Exception as exc:
        logger.error("Error generating explanation: %s", exc)
        return None


def render_mermaid_image(mermaid_code: str) -> Optional[str]:
    """Render Mermaid code to a PNG via mermaid.ink; returns base64 string."""
    try:
        encoded = base64.urlsafe_b64encode(mermaid_code.encode("utf-8")).decode("utf-8")
        url = f"https://mermaid.ink/img/{encoded}"
        resp = requests.get(url, timeout=15)
        if resp.status_code == 200:
            return base64.b64encode(resp.content).decode("utf-8")
        logger.warning("mermaid.ink returned status %d", resp.status_code)
        return None
    except Exception as exc:
        logger.error("Error rendering Mermaid image: %s", exc)
        return None
