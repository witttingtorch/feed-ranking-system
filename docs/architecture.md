# System Architecture: Multi-Objective Feed Ranking System

## Overview

This system implements a **production-style feed ranking architecture**
inspired by large-scale platforms.

The ranking pipeline is explicitly decomposed into **three stages**:

Recall (Candidate Generation) → Rank (Relevance) → Re-rank (Objectives & Constraints)


This separation enables:
- Scalability
- Independent optimization
- Safe experimentation

> Principle: *Different stages optimize different objectives under different constraints.*

---

## High-Level Request Flow

1. User opens feed / submits search request
2. Candidate generation retrieves a small, high-recall set of items
3. Features are joined across user, item, and context
4. A ranking model scores candidates for relevance
5. A re-ranking layer optimizes multiple objectives and enforces constraints
6. Ranked results are returned to the client

---

## 1. Candidate Generation (Recall Layer)

### Purpose
Reduce the item universe (millions) to a manageable candidate set (hundreds)
under strict latency constraints.

### Techniques
- Embedding-based approximate nearest neighbor (ANN) retrieval
- Heuristic recall (recent interactions, follows, trending content)
- Rule-based inclusion for cold-start and policy requirements

### Design Tradeoffs
| Priority | Explanation |
|-------|-------------|
| High recall | Missing a relevant item here is irrecoverable |
| Low latency | Recall runs on every request |
| Lower precision | Precision is handled downstream |

> Recall optimizes **coverage**, not ranking quality.

---

## 2. Feature Join Layer

### Inputs
- User features (short-term behavior, long-term preferences)
- Item features (popularity, freshness, content attributes)
- Context features (time, device, network conditions)

### Characteristics
- Stateless computation
- Feature versioning support
- Online/offline parity

### Tradeoffs
- Rich features improve ranking but increase latency
- Strict latency budgets limit feature complexity

---

## 3. Ranking Layer (Relevance Model)

### Model Choice
- Gradient Boosted Decision Trees (GBDT)

### Rationale
- Strong performance on tabular data
- Interpretable feature contributions
- Stable training and serving behavior
- Lower latency than deep models

### Output
- Predicted engagement probability (e.g., click-through rate)

### Calibration
Raw model scores are calibrated to probabilities using:
- Platt scaling or isotonic regression

> Calibrated probabilities enable downstream decision-making and fair comparisons.

---

## 4. Re-Ranking Layer (Multi-Objective Optimization)

### Motivation
Pure relevance optimization leads to:
- Content repetition
- Filter bubbles
- Long-term user fatigue

### Objectives
1. Relevance (short-term engagement)
2. Diversity (topic and creator coverage)
3. Long-term retention (return probability proxy)

### Scoring Formulation
```math
FinalScore = α·Relevance + β·Diversity + γ·Retention