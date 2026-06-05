"""
video_service.py — YouTube search with ML-relevance ranking.
Logic extracted verbatim from app.py, now as a pure function.
"""
from __future__ import annotations
import logging
from typing import Optional
from backend.models.schemas import VideoData

logger = logging.getLogger(__name__)

_ML_KEYWORDS = [
    "machine learning", "ml", "data science", "algorithm",
    "neural network", "deep learning", "ai", "artificial intelligence",
    "tutorial", "explained", "introduction",
]
_EDUCATIONAL_TITLE_WORDS = ["tutorial", "explained", "introduction", "guide"]
_EDUCATIONAL_CHANNEL_WORDS = ["academy", "education", "learning", "tech"]


def _relevance_score(video: dict) -> int:
    title = video.get("title", "").lower()
    channel = video.get("channel", "").lower()
    score = sum(1 for kw in _ML_KEYWORDS if kw in title or kw in channel)
    if any(w in title for w in _EDUCATIONAL_TITLE_WORDS):
        score += 2
    if any(w in channel for w in _EDUCATIONAL_CHANNEL_WORDS):
        score += 1
    duration = video.get("duration", "")
    if duration and len(duration) > 4:
        score += 1
    return score


def search_youtube_video(query: str) -> Optional[VideoData]:
    """Search YouTube for the most relevant educational video on the query."""
    try:
        from youtube_search import YoutubeSearch
        enhanced_query = f"{query} machine learning tutorial explanation"
        results = YoutubeSearch(enhanced_query, max_results=5).to_dict()
        if not results:
            return None

        ranked = sorted(results, key=_relevance_score, reverse=True)
        best = ranked[0]
        video_id = best.get("id") or best.get("url_suffix", "").split("v=")[-1]

        return VideoData(
            title=best.get("title", "Untitled Video"),
            url=f"https://www.youtube.com/watch?v={video_id}",
            channel=best.get("channel", "Unknown Channel"),
            duration=best.get("duration", ""),
            views=best.get("views", ""),
        )
    except Exception as exc:
        logger.error("Error fetching YouTube video: %s", exc)
        return None
