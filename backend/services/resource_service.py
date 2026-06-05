"""
resource_service.py — GeeksforGeeks learning resource lookup.
Contains the full GFG topic map from app.py, plus Groq fallback URL generation.
"""
from __future__ import annotations
import logging
from typing import Optional

logger = logging.getLogger(__name__)

_GFG_BASE = "https://www.geeksforgeeks.org/"

# Comprehensive ML topic → GFG article slug mapping
_GFG_TOPIC_MAP: dict[str, str] = {
    "supervised learning": "supervised-unsupervised-learning",
    "unsupervised learning": "supervised-unsupervised-learning",
    "neural network": "introduction-to-artificial-neutral-networks",
    "cnn": "introduction-convolution-neural-network",
    "convolutional neural network": "introduction-convolution-neural-network",
    "rnn": "introduction-to-recurrent-neural-network",
    "recurrent neural network": "introduction-to-recurrent-neural-network",
    "decision tree": "decision-tree-introduction-example",
    "random forest": "random-forest-algorithm-in-machine-learning",
    "svm": "support-vector-machine-algorithm",
    "support vector machine": "support-vector-machine-algorithm",
    "k means": "k-means-clustering-introduction",
    "kmeans": "k-means-clustering-introduction",
    "linear regression": "ml-linear-regression",
    "logistic regression": "understanding-logistic-regression",
    "knn": "k-nearest-neighbours",
    "k nearest neighbor": "k-nearest-neighbours",
    "naive bayes": "naive-bayes-classifiers",
    "gradient descent": "gradient-descent-algorithm-and-its-variants",
    "backpropagation": "backpropagation-in-neural-network",
    "overfitting": "underfitting-and-overfitting-in-machine-learning",
    "underfitting": "underfitting-and-overfitting-in-machine-learning",
    "regularization": "regularization-in-machine-learning",
    "cross validation": "cross-validation-machine-learning",
    "pca": "principal-component-analysis-pca",
    "principal component analysis": "principal-component-analysis-pca",
    "dimensionality reduction": "dimensionality-reduction",
    "ensemble learning": "ensemble-methods-in-machine-learning",
    "boosting": "boosting-in-machine-learning-boosting-and-adaboost",
    "bagging": "bagging-in-machine-learning",
    "lstm": "deep-learning-introduction-to-long-short-term-memory",
    "gru": "gated-recurrent-unit-networks",
    "autoencoder": "auto-encoders",
    "gan": "generative-adversarial-network-gan",
    "generative adversarial network": "generative-adversarial-network-gan",
    "transformer": "transformer-neural-network",
    "attention mechanism": "attention-mechanism",
    "reinforcement learning": "what-is-reinforcement-learning",
    "q learning": "q-learning-in-python",
    "deep learning": "introduction-deep-learning",
    "machine learning": "machine-learning",
    "activation function": "activation-functions-neural-networks",
    "loss function": "loss-functions-in-machine-learning",
    "optimizer": "optimization-techniques-for-gradient-descent",
    "batch normalization": "batch-normalization-ml",
    "dropout": "dropout-in-neural-networks",
    "transfer learning": "ml-introduction-to-transfer-learning",
    "data preprocessing": "data-preprocessing-machine-learning-python",
    "feature engineering": "feature-engineering",
    "feature selection": "feature-selection-techniques-in-machine-learning",
    "confusion matrix": "confusion-matrix-machine-learning",
    "precision recall": "precision-and-recall-in-machine-learning",
    "f1 score": "f1-score-in-machine-learning",
    "roc curve": "roc-curve-in-machine-learning",
    "bias variance": "bias-variance-tradeoff-machine-learning",
}


def search_learning_resource(query: str, groq_client=None) -> dict:
    """
    Find the most relevant GeeksforGeeks article for an ML topic.
    Falls back to a Groq-generated URL, then to a GFG search URL.
    """
    query_lower = query.lower().strip()

    # Direct map lookup
    for key, slug in _GFG_TOPIC_MAP.items():
        if key in query_lower or query_lower in key:
            return {
                "title": f"GeeksforGeeks: {query}",
                "url": f"{_GFG_BASE}{slug}/",
                "description": f"Comprehensive tutorial and hands-on examples for {query} on GeeksforGeeks",
            }

    # Groq-assisted URL generation
    if groq_client:
        try:
            resource_prompt = (
                f'Given the ML topic "{query}", generate a GeeksforGeeks article URL.\n'
                "Format: https://www.geeksforgeeks.org/[slug]/\n"
                "Provide ONLY:\n"
                "TITLE: GeeksforGeeks: [topic]\n"
                "URL: https://www.geeksforgeeks.org/[slug]/\n"
                "DESCRIPTION: [one sentence]"
            )
            raw = groq_client.chat.completions.create(
                messages=[
                    {"role": "system", "content": "Generate valid GeeksforGeeks article URLs for ML topics."},
                    {"role": "user", "content": resource_prompt},
                ],
                model="llama-3.3-70b-versatile",
                max_tokens=150,
                temperature=0.2,
            ).choices[0].message.content.strip()

            title = url = description = ""
            for line in raw.split("\n"):
                if line.startswith("TITLE:"):
                    title = line[6:].strip()
                elif line.startswith("URL:"):
                    url = line[4:].strip()
                elif line.startswith("DESCRIPTION:"):
                    description = line[12:].strip()

            if url and "geeksforgeeks.org" in url.lower():
                return {
                    "title": title or f"GeeksforGeeks: {query}",
                    "url": url,
                    "description": description or f"Learn about {query} with practical examples",
                }
        except Exception as exc:
            logger.error("Error in Groq resource generation: %s", exc)

    # Final fallback: GFG search
    return {
        "title": f"GeeksforGeeks: {query}",
        "url": f"{_GFG_BASE}?s={query.replace(' ', '+')}",
        "description": f"Search GeeksforGeeks for tutorials on {query}",
    }
