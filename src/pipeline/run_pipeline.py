"""
FAANG-style end-to-end ranking pipeline runner.

Stages:
1. Load synthetic logged bandit data
2. Define candidate ranking policies
3. Evaluate policies using IPS / SNIPS
4. Run inference for a sample user
5. Print experiment-style summary

This file mirrors how ranking experiments are reviewed in practice.
"""

import numpy as np
import pandas as pd

from src.evaluation.ips_snips import ips, snips
from src.serving.ranking_service import RankingService


# -----------------------------
# Load Logged Bandit Data
# -----------------------------

DATA_PATH = "data/raw/logged_bandit_data.parquet"

df = pd.read_parquet(DATA_PATH)


# -----------------------------
# Define Policies
# -----------------------------

def baseline_policy(row):
    """
    Baseline policy: repeat logged action
    (represents current production system)
    """
    return row["action"]


def new_policy(row):
    """
    Example new policy:
    Prefer low-index items (toy proxy for a learned policy)
    """
    return 0


# -----------------------------
# Counterfactual Evaluation
# -----------------------------

ips_baseline = ips(df, baseline_policy)
snips_baseline = snips(df, baseline_policy)

ips_new = ips(df, new_policy)
snips_new = snips(df, new_policy)


# -----------------------------
# Online Inference Simulation
# -----------------------------

service = RankingService()

np.random.seed(0)

user_embedding = np.random.randn(32)
item_embeddings = pd.DataFrame(
    np.random.randn(100, 32),
    index=np.arange(100)
)

user_context = {
    "recent_items": [1, 2, 3],
    "trending_items": [10, 11],
    "follow_items": [20],
    "user_activity": 0.6,
    "item_popularity": {i: np.random.rand() for i in range(100)},
}

retention_scores = {i: np.random.rand() for i in range(100)}

weights = {
    "relevance": 1.0,
    "diversity": 0.3,
    "retention": 0.2,
}

ranked_items = service.rank(
    user_embedding=user_embedding,
    item_embeddings=item_embeddings,
    user_context=user_context,
    retention_scores=retention_scores,
    weights=weights,
    top_k=10,
)


# -----------------------------
# Experiment Summary
# -----------------------------

print("\n===== OFFLINE POLICY EVALUATION =====")
print(f"Baseline IPS:   {ips_baseline:.4f}")
print(f"Baseline SNIPS: {snips_baseline:.4f}")
print(f"New Policy IPS:   {ips_new:.4f}")
print(f"New Policy SNIPS: {snips_new:.4f}")

print("\n===== ONLINE INFERENCE SAMPLE =====")
print("Top-10 ranked items:", ranked_items)

print("\n===== DECISION =====")
if snips_new > snips_baseline:
    print("Candidate policy improves estimated value → SAFE TO A/B TEST")
else:
    print("Candidate policy underperforms → DO NOT SHIP")
