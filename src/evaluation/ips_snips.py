"""
FAANG / WorldQuant-style counterfactual evaluation.

Implements:
- Inverse Propensity Scoring (IPS)
- Self-Normalized IPS (SNIPS)

This module evaluates ranking *policies* using logged bandit feedback.
"""

from typing import Callable, Dict
import numpy as np
import pandas as pd


# -----------------------------
# Utilities
# -----------------------------

def clip_propensity(p: float, epsilon: float = 0.01) -> float:
    """
    Prevent extreme importance weights.
    """
    return max(p, epsilon)


# -----------------------------
# IPS
# -----------------------------

def ips(
    df: pd.DataFrame,
    policy_fn: Callable[[pd.Series], int],
    epsilon: float = 0.01,
) -> float:
    """
    Compute IPS estimate of policy value.

    Required columns in df:
    - action: logged action
    - reward: observed reward
    - propensity: logging policy probability
    """

    weights = []
    rewards = []

    for _, row in df.iterrows():
        logged_action = row["action"]
        reward = row["reward"]
        propensity = clip_propensity(row["propensity"], epsilon)

        if policy_fn(row) == logged_action:
            weights.append(1.0 / propensity)
            rewards.append(reward)

    if not rewards:
        return 0.0

    return np.mean(np.array(weights) * np.array(rewards))


# -----------------------------
# SNIPS
# -----------------------------

def snips(
    df: pd.DataFrame,
    policy_fn: Callable[[pd.Series], int],
    epsilon: float = 0.01,
) -> float:
    """
    Compute SNIPS estimate of policy value.
    """

    weighted_rewards = []
    weights = []

    for _, row in df.iterrows():
        logged_action = row["action"]
        reward = row["reward"]
        propensity = clip_propensity(row["propensity"], epsilon)

        if policy_fn(row) == logged_action:
            w = 1.0 / propensity
            weighted_rewards.append(w * reward)
            weights.append(w)

    if not weights:
        return 0.0

    return np.sum(weighted_rewards) / np.sum(weights)


# -----------------------------
# Example Usage
# -----------------------------

if __name__ == "__main__":
    # Synthetic logged bandit data
    np.random.seed(42)
    n = 1000

    df = pd.DataFrame({
        "action": np.random.randint(0, 5, size=n),
        "reward": np.random.binomial(1, 0.2, size=n),
        "propensity": np.random.uniform(0.05, 0.5, size=n),
    })

    # Example policy: always pick action 0
    def new_policy(row):
        return 0

    ips_value = ips(df, new_policy)
    snips_value = snips(df, new_policy)

    print(f"IPS estimate:   {ips_value:.4f}")
    print(f"SNIPS estimate: {snips_value:.4f}")
