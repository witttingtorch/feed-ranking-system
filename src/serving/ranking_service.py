"""
FAANG-style ranking inference service.

Responsibilities:
- Online inference only (no training)
- Low-latency, deterministic execution
- Integrates recall, ranking, and re-ranking
"""

import json
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import lightgbm as lgb

from src.recall.candidate_generation import generate_candidates
from src.rerank.multi_objective_rerank import rerank


# -----------------------------
# Configuration
# -----------------------------

MODEL_PATH = Path("models/gbdt_ranker.txt")
CALIBRATOR_PATH = Path("models/calibrator.json")

LATENCY_BUDGET_MS = 60


# -----------------------------
# Calibration Loader
# -----------------------------

class IsotonicCalibrator:
    def __init__(self, payload: Dict):
        self.x = np.array(payload["thresholds"])
        self.y = np.array(payload["values"])

    def predict(self, scores: np.ndarray) -> np.ndarray:
        return np.interp(scores, self.x, self.y)


# -----------------------------
# Ranking Service
# -----------------------------

class RankingService:
    def __init__(self):
        self.model = self._load_model()
        self.calibrator = self._load_calibrator()

    def _load_model(self):
        if not MODEL_PATH.exists():
            raise FileNotFoundError("Ranking model not found")
        return lgb.Booster(model_file=str(MODEL_PATH))

    def _load_calibrator(self):
        if not CALIBRATOR_PATH.exists():
            raise FileNotFoundError("Calibrator not found")
        with open(CALIBRATOR_PATH) as f:
            payload = json.load(f)
        return IsotonicCalibrator(payload)

    def rank(
        self,
        user_embedding: np.ndarray,
        item_embeddings: pd.DataFrame,
        user_context: Dict,
        retention_scores: Dict[int, float],
        weights: Dict[str, float],
        top_k: int = 10,
    ) -> List[int]:
        start = time.time()

        # -------- Recall --------
        candidates = generate_candidates(
            user_embedding=user_embedding,
            item_embeddings=item_embeddings,
            recent_item_ids=user_context.get("recent_items", []),
            trending_item_ids=user_context.get("trending_items", []),
            follow_item_ids=user_context.get("follow_items", []),
        )

        # -------- Feature Join --------
        features = self._build_features(candidates, user_context)
        scores = self.model.predict(features)

        calibrated = self.calibrator.predict(scores)

        relevance_scores = dict(zip(candidates, calibrated))

        # -------- Re-rank --------
        ranked = rerank(
            candidate_ids=candidates,
            relevance_scores=relevance_scores,
            embeddings={i: item_embeddings.loc[i].values for i in candidates},
            retention_scores=retention_scores,
            weights=weights,
            top_k=top_k,
        )

        elapsed_ms = (time.time() - start) * 1000
        if elapsed_ms > LATENCY_BUDGET_MS:
            print(f"[WARN] Ranking latency {elapsed_ms:.1f}ms exceeded budget")

        return ranked

    def _build_features(self, item_ids: List[int], user_context: Dict) -> pd.DataFrame:
        """
        Minimal placeholder feature join.
        In production, this must match training features exactly.
        """
        df = pd.DataFrame({
            "item_popularity": [user_context.get("item_popularity", {}).get(i, 0.0) for i in item_ids],
            "user_activity": user_context.get("user_activity", 0.0),
        })
        return df


# -----------------------------
# Example Usage
# -----------------------------

if __name__ == "__main__":
    np.random.seed(0)

    service = RankingService()

    user_emb = np.random.randn(32)
    item_embs = pd.DataFrame(
        np.random.randn(500, 32),
        index=np.arange(500)
    )

    context = {
        "recent_items": [1, 2, 3],
        "trending_items": [10, 11],
        "follow_items": [20],
        "user_activity": 0.7,
        "item_popularity": {i: np.random.rand() for i in range(500)},
    }

    retention = {i: np.random.rand() for i in range(500)}

    weights = {
        "relevance": 1.0,
        "diversity": 0.3,
        "retention": 0.2,
    }

    ranked_items = service.rank(
        user_embedding=user_emb,
        item_embeddings=item_embs,
        user_context=context,
        retention_scores=retention,
        weights=weights,
        top_k=10,
    )

    print("Final ranked items:", ranked_items)
