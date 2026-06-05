"""
ml_service.py — SVM learning-style classifier loading and prediction.
Extracted from app.py; operates as a pure function with no global state.
"""
from __future__ import annotations
import logging
from pathlib import Path

import joblib
import numpy as np

logger = logging.getLogger(__name__)

STYLE_MAP: dict[int, str] = {0: "Auditory", 1: "Reading/Writing", 2: "Visual"}

# Likert scale → numeric (matches the training dataset encoding)
SCALE: dict[str, int] = {
    "Strongly Agree": 5,
    "Agree": 4,
    "Neutral": 3,
    "Disagree": 2,
    "Strongly Disagree": 1,
}

# The 15 fixed assessment questions in dataset order
QUESTIONS: list[str] = [
    "I learn better by reading what the teacher writes on the board.",          # Visual 1
    "I learn better by reading instructions than by listening to instructions.", # Visual 2
    "I understand better when I read instructions.",                            # Visual 3
    "I learn better by reading than by listening to someone.",                  # Visual 4
    "I learn more by reading textbooks than by listening to lectures.",         # Visual 5
    "When the teacher tells me the instructions, I understand better.",         # Auditory 1
    "I learn better in class when listening to the teacher than reading the textbook.",  # Auditory 2
    "I understand things better in class when the teacher gives a lecture.",   # Auditory 3
    "I learn better in class when I listen to someone.",                       # Auditory 4
    "I remember things I have heard in class better than things I have read.", # Auditory 5
    "I prefer to learn by doing something in class.",                          # Kinesthetic 1
    "When I do things in class, I learn better.",                              # Kinesthetic 2
    "I enjoy learning in class by doing experiments.",                         # Kinesthetic 3
    "I understand things better in class when I participate in role-playing.", # Kinesthetic 4
    "I learn best in class when I can participate in related activities.",     # Kinesthetic 5
]


def load_svm_model(model_path: Path, scaler_path: Path):
    """Load SVM model and scaler from .pkl files. Returns (model, scaler) or (None, None)."""
    try:
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        logger.info("✓ SVM model loaded from %s", model_path)
        return model, scaler
    except Exception as exc:
        logger.error("❌ Error loading SVM model: %s", exc)
        return None, None


def predict_style(
    responses: list[int],
    svm_model,
    svm_scaler,
) -> tuple[str, float]:
    """
    Predict learning style from a list of 15 Likert numeric responses.

    Returns:
        (predicted_style, confidence_percentage)
    """
    features = np.array(responses).reshape(1, -1)
    features_scaled = svm_scaler.transform(features)
    prediction = svm_model.predict(features_scaled)[0]
    probabilities = svm_model.predict_proba(features_scaled)[0]
    confidence = float(probabilities[prediction] * 100)
    style = STYLE_MAP.get(int(prediction), "Reading/Writing")
    logger.info("Predicted style: %s (%.2f%% confidence)", style, confidence)
    return style, round(confidence, 2)
