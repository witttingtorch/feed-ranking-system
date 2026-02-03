"""
FAANG-style multi-objective re-ranking module.

Responsibilities:
- Balance relevance, diversity, and retention
- Enforce ranking tradeoffs explicitly
- Operate on a small candidate set

This module assumes:
- Relevance scores come from the Rank stage
- Diversity is computed at the list level
"""

from typing import List, Dict
import numpy as np

# -----------------------------
# Diversity Utility
# -----------------------------

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)


def diversity_penalty(
    candidate_embedding: np.ndarray,
    selected_embeddings: List[np.ndarray],
) -> float:
    """
    Penalize candidates that are too similar
    to already selected items.
    """
    if not selected_embeddings:
        return 0.0

    similarities = [
        cosine_similarity(candidate_embedding, emb)
        for emb in selected_embeddings
    ]
    return max(similarities)


# -----------------------------
# Multi-Objective Scoring
# -----------------------------

def compute_final_score(
    relevance: float,
    diversity: float,
    retention: float,
    weights: Dict[str, float],
) -> float:
    """
    Combine objectives into a single scalar score.
    """
    return (
        weights["relevance"] * relevance
        - weights["diversity"] * diversity
        + weights["retention"] * retention
    )


# -----------------------------
# Re-ranking Algorithm
# -----------------------------

def rerank(
    candidate_ids: List[int],
    relevance_scores: Dict[int, float],
    embeddings: Dict[int, np.ndarray],
    retention_scores: Dict[int, float],
    weights: Dict[str, float],
    top_k: int = 20,
) -> List[int]:
    """
    Greedy re-ranking using a multi-objective score.
    """

    selected = []
    selected_embeddings = []

    remaining = set(candidate_ids)

    for _ in range(min(top_k, len(candidate_ids))):
        best_id = None
        best_score = -np.inf

        for item_id in remaining:
            rel = relevance_scores.get(item_id, 0.0)
            ret = retention_scores.get(item_id, 0.0)

            div = diversity_penalty(
                embeddings[item_id],
                selected_embeddings
            )

            score = compute_final_score(
                relevance=rel,
                diversity=div,
                retention=ret,
                weights=weights
            )

            if score > best_score:
                best_score = score
                best_id = item_id

        selected.append(best_id)
        selected_embeddings.append(embeddings[best_id])
        remaining.remove(best_id)

    return selected


# -----------------------------
# Example Usage
# -----------------------------

if __name__ == "__main__":
    np.random.seed(0)

    candidate_ids = list(range(50))
    relevance_scores = {i: np.random.rand() for i in candidate_ids}
    retention_scores = {i: np.random.rand() for i in candidate_ids}
    embeddings = {i: np.random.randn(16) for i in candidate_ids}

    weights = {
        "relevance": 1.0,
        "diversity": 0.3,
        "retention": 0.2,
    }

    ranked = rerank(
        candidate_ids=candidate_ids,
        relevance_scores=relevance_scores,
        embeddings=embeddings,
        retention_scores=retention_scores,
        weights=weights,
        top_k=10,
    )

    print("Top ranked items:", ranked)
