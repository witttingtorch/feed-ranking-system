"""
FAANG-style candidate generation (recall) module.

Responsibilities:
- Retrieve a high-recall candidate set
- Optimize for coverage and latency
- Combine multiple recall sources
"""

from typing import List, Dict
import numpy as np
import pandas as pd

# -----------------------------
# Embedding Recall (ANN-ready)
# -----------------------------

def embedding_recall(
    user_embedding: np.ndarray,
    item_embeddings: pd.DataFrame,
    top_k: int = 200,
) -> List[int]:
    """
    Retrieve candidates via embedding similarity.
    This implementation uses dot product for simplicity.

    In production:
    - Replace with FAISS / ScaNN
    - Pre-index item embeddings
    """
    scores = item_embeddings.values @ user_embedding
    top_indices = np.argsort(scores)[-top_k:][::-1]
    return item_embeddings.index[top_indices].tolist()


# -----------------------------
# Heuristic Recall
# -----------------------------

def heuristic_recall(
    recent_item_ids: List[int],
    trending_item_ids: List[int],
    follow_item_ids: List[int],
    max_items: int = 100,
) -> List[int]:
    """
    Rule-based recall to ensure coverage and cold-start handling.
    """
    candidates = []

    candidates.extend(recent_item_ids)
    candidates.extend(trending_item_ids)
    candidates.extend(follow_item_ids)

    # Preserve order, remove duplicates
    seen = set()
    deduped = []
    for item_id in candidates:
        if item_id not in seen:
            seen.add(item_id)
            deduped.append(item_id)

    return deduped[:max_items]


# -----------------------------
# Candidate Merger
# -----------------------------

def generate_candidates(
    user_embedding: np.ndarray,
    item_embeddings: pd.DataFrame,
    recent_item_ids: List[int],
    trending_item_ids: List[int],
    follow_item_ids: List[int],
    embedding_k: int = 200,
    heuristic_k: int = 100,
) -> List[int]:
    """
    Merge multiple recall sources into a single candidate set.
    """

    emb_candidates = embedding_recall(
        user_embedding=user_embedding,
        item_embeddings=item_embeddings,
        top_k=embedding_k,
    )

    heuristic_candidates = heuristic_recall(
        recent_item_ids=recent_item_ids,
        trending_item_ids=trending_item_ids,
        follow_item_ids=follow_item_ids,
        max_items=heuristic_k,
    )

    # Merge + deduplicate
    merged = []
    seen = set()

    for source in (emb_candidates, heuristic_candidates):
        for item_id in source:
            if item_id not in seen:
                seen.add(item_id)
                merged.append(item_id)

    return merged


# -----------------------------
# Example Usage
# -----------------------------

if __name__ == "__main__":
    # Dummy data for sanity check
    user_emb = np.random.randn(32)
    items = pd.DataFrame(
        np.random.randn(1000, 32),
        index=np.arange(1000)
    )

    candidates = generate_candidates(
        user_embedding=user_emb,
        item_embeddings=items,
        recent_item_ids=[1, 2, 3],
        trending_item_ids=[10, 11, 12],
        follow_item_ids=[20, 21],
    )

    print(f"Generated {len(candidates)} candidates")
