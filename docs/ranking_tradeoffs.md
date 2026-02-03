# Ranking Tradeoffs in Feed and Search Systems

## Overview

Ranking systems rarely optimize a single objective.
In production environments, improving one metric often degrades others.

This document describes the **core tradeoffs** involved in feed and search ranking
and how they are managed through multi-objective optimization and system design.

> Principle: *Every ranking gain has a cost somewhere else in the system.*

---

## 1. Relevance vs Diversity

### Relevance

Relevance measures how well ranked items match a user’s immediate intent.
Typical proxies include:

* Click-through rate (CTR)
* Dwell time
* Likes or saves

Optimizing purely for relevance tends to:

* Re-rank similar items repeatedly
* Favor popular creators or topics
* Exploit short-term user preferences

---

### Diversity

Diversity measures how varied the ranked results are across dimensions such as:

* Topics or content categories
* Creators or sources
* Item age or novelty

Common diversity metrics:

* Topic entropy
* Intra-list similarity
* Creator exposure concentration

Increasing diversity:

* Reduces repetition and fatigue
* Improves long-term user satisfaction
* Protects ecosystem health

However, excessive diversity can reduce perceived relevance.

> Tradeoff: *Diversity improves long-term value but may reduce short-term engagement.*

---

## 2. Short-Term Engagement vs Long-Term Retention

### Short-Term Optimization

Metrics such as CTR and session time respond quickly to ranking changes.
They are useful for fast iteration but are often **myopic**.

Failure modes:

* Clickbait amplification
* Sensational or extreme content dominance
* Rapid novelty decay

---

### Long-Term Retention

Long-term retention captures whether users return over days or weeks.
It reflects cumulative user experience rather than single-session success.

Common proxies:

* D1 / D7 / D30 retention
* Session frequency
* Churn probability models

Optimizing for retention often requires:

* Exposure smoothing
* Reduced repetition
* Preference exploration

> A ranking system that maximizes CTR but harms retention is failing its core objective.

---

## 3. Exploration vs Exploitation

### Exploitation

Exploitation ranks items with high predicted relevance based on historical data.

Pros:

* Stable performance
* High immediate engagement

Cons:

* Feedback loops
* Cold-start failure
* Preference lock-in

---

### Exploration

Exploration deliberately surfaces uncertain or under-exposed items.

Benefits:

* Improved data collection
* Discovery of new interests
* Reduced bias in training data

Costs:

* Temporary engagement loss
* Increased variance

Effective ranking systems balance both through:

* Randomized slots
* Uncertainty-aware scoring
* Session-dependent exploration rates

---

## 4. Accuracy vs Latency

More complex models often yield better offline accuracy but:

* Increase inference latency
* Raise infrastructure cost
* Risk violating real-time constraints

In feed systems, latency directly impacts:

* Page load time
* User drop-off
* Downstream services

> A more accurate model that misses latency budgets is not shippable.

This motivates:

* Model simplicity (e.g., GBDT over deep models)
* Feature pruning
* Multi-stage ranking pipelines

---

## 5. User-Level vs Ecosystem-Level Objectives

### User-Level

Focuses on individual satisfaction:

* Relevance
* Personalization
* Immediate enjoyment

### Ecosystem-Level

Focuses on system-wide health:

* Creator fairness
* Content diversity
* Long-tail exposure

Optimizing only user-level metrics can collapse ecosystems
by over-concentrating exposure.

> Ranking systems shape markets, not just feeds.

---

## 6. Multi-Objective Optimization Framework

These tradeoffs are typically managed through weighted objectives:

[
\text{Score} = \alpha \cdot \text{Relevance}
+ \beta \cdot \text{Diversity}
+ \gamma \cdot \text{Retention}
]

Where:

* Weights are tuned via experiments
* Weights may vary by user cohort or session depth

Hard constraints (guardrails) are applied to prevent unacceptable regressions.

---

## 7. Practical Implications for Evaluation

* Offline metrics alone cannot resolve tradeoffs
* Counterfactual evaluation estimates policy value
* A/B testing validates real-world impact
* Guardrails enforce safety and stability

No single metric determines success.

---

## Summary

Ranking is a multi-objective decision problem.

Key takeaways:

* Improving relevance alone is insufficient
* Diversity and retention are first-class objectives
* Tradeoffs must be explicit and managed
* Evaluation must be policy-centric, not model-centric

> Effective ranking systems optimize for **sustained value**, not isolated wins.
