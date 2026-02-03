"""
FAANG / WorldQuant-style synthetic logged bandit data generator.

Purpose:
- Simulate biased logged interaction data
- Enable counterfactual evaluation (IPS / SNIPS)
- Reproduce selective feedback realistically
"""

from pathlib import Path
import numpy as np
import pandas as pd


# -----------------------------
# Configuration
# -----------------------------

OUTPUT_PATH = Path("data/raw/logged_bandit_data.parquet")

N_USERS = 1000
N_ITEMS = 50
N_EVENTS = 20000

RANDOM_SEED = 42


# -----------------------------
# True Reward Model (Hidden)
# -----------------------------

def true_reward_prob(user_pref: float, item_attr: float) -> float:
    """
    Ground-truth reward probability (unknown to the learner).
    """
    logit = 2.0 * user_pref * item_attr
    return 1.0 / (1.0 + np.exp(-logit))


# -----------------------------
# Logging Policy
# -----------------------------

def logging_policy_scores(user_pref: float, item_attrs: np.ndarray) -> np.ndarray:
    """
    Biased logging policy (over-exploits high-score items).
    """
    noise = np.random.normal(0, 0.3, size=len(item_attrs))
    return user_pref * item_attrs + noise


def softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - np.max(x))
    return e / e.sum()


# -----------------------------
# Data Generation
# -----------------------------

def generate_logged_data() -> pd.DataFrame:
    np.random.seed(RANDOM_SEED)

    user_prefs = np.random.uniform(-1, 1, size=N_USERS)
    item_attrs = np.random.uniform(-1, 1, size=N_ITEMS)

    records = []

    for _ in range(N_EVENTS):
        user_id = np.random.randint(0, N_USERS)
        user_pref = user_prefs[user_id]

        scores = logging_policy_scores(user_pref, item_attrs)
        probs = softmax(scores)

        action = np.random.choice(N_ITEMS, p=probs)
        propensity = probs[action]

        reward_prob = true_reward_prob(user_pref, item_attrs[action])
        reward = np.random.binomial(1, reward_prob)

        records.append({
            "user_id": user_id,
            "action": action,
            "reward": reward,
            "propensity": propensity,
            "user_pref": user_pref,
            "item_attr": item_attrs[action],
        })

    return pd.DataFrame(records)


# -----------------------------
# Entry Point
# -----------------------------

if __name__ == "__main__":
    df = generate_logged_data()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUTPUT_PATH, index=False)

    print(f"Generated {len(df)} logged events")
    print(f"Saved to {OUTPUT_PATH}")
