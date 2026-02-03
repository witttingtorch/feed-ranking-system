# Multi-Objective Feed Ranking System

A **production-style feed ranking system** that mirrors how large-scale platforms design, evaluate, and ship ranking policies.

This project implements a full **recall → rank → re-rank** pipeline with **counterfactual evaluation** and **latency-aware inference**, inspired by real-world ranking systems.

---

## Why This Project

Ranking systems are not just machine learning models — they are **decision systems under constraints**.

This project was built to answer the interview question:

> **“How would you rank a feed?”**

With:

* Architecture
* Math
* Tradeoffs
* Evaluation strategy

---

## System Overview

```
User Request
   ↓
Candidate Generation (Recall)
   ↓
Feature Join
   ↓
Relevance Ranking (GBDT + Calibration)
   ↓
Multi-Objective Re-Ranking
   ↓
Final Ranked Feed
```

### Core Principles

* Separate **recall** from **ranking**
* Optimize **multiple objectives**, not just CTR
* Evaluate **policies**, not models
* Enforce **guardrails** before shipping

---

## Key Components

### 1. Candidate Generation (Recall)

📁 `src/recall/`

* Embedding-based recall (ANN-ready)
* Heuristic recall (recent, trending, follows)
* Deduplication and merging

**Goal:** Maximize coverage under strict latency constraints.

---

### 2. Relevance Ranking

📁 `src/ranking/`

* Gradient Boosted Decision Trees (LightGBM)
* Tabular feature modeling
* Probability calibration (Isotonic Regression)

**Why GBDT?**

* Strong on tabular data
* Interpretable
* Stable and low-latency in production

---

### 3. Multi-Objective Re-Ranking

📁 `src/rerank/`

Optimizes competing objectives:

* **Relevance** (short-term engagement)
* **Diversity** (topic & creator coverage)
* **Retention** (long-term value proxy)

[
\text{Score} = \alpha \cdot \text{Relevance}
- \beta \cdot \text{Similarity}
+ \gamma \cdot \text{Retention}
]

Implemented using a greedy MMR-style algorithm.

---

### 4. Counterfactual Evaluation (Offline)

📁 `src/evaluation/`

* Inverse Propensity Scoring (IPS)
* Self-Normalized IPS (SNIPS)
* Propensity clipping and support checks

> Offline metrics like AUC are biased under selective feedback.
> This system evaluates **ranking policies** using logged bandit data.

---

### 5. Ranking Inference Service

📁 `src/serving/`

* Model + calibrator loading
* Recall → rank → re-rank orchestration
* Latency budget enforcement (P95-aware)

Designed to resemble a real online ranking service.

---

### 6. Synthetic Logged Bandit Data

📁 `src/data/`

* Biased logging policy
* True (hidden) reward model
* Logged propensities
* Selective feedback

Enables **correct IPS / SNIPS evaluation** without real production data.

---

### 7. End-to-End Pipeline

📁 `src/pipeline/`

Runs the full flow:

1. Load logged bandit data
2. Evaluate policies offline (IPS / SNIPS)
3. Run inference for a sample user
4. Produce a ship / no-ship decision

---

## Evaluation Strategy

### Offline

* Counterfactual policy evaluation (IPS / SNIPS)
* Variance-aware comparison
* Filters candidates before online testing

### Online (Simulated)

* A/B guardrail metrics
* Latency budgets
* Diversity and UX constraints

📁 `docs/ab_guardrails.md`

---

## Project Structure

```
src/
├── recall/        # Candidate generation
├── ranking/       # GBDT relevance model
├── rerank/        # Multi-objective optimization
├── evaluation/    # IPS / SNIPS
├── serving/       # Inference service
├── data/          # Synthetic logging
├── pipeline/      # End-to-end runner
```

---

## How to Run

### 1. Generate synthetic logged data

```bash
python src/data/synthetic_logged_data.py
```

### 2. Train ranking model

```bash
python src/ranking/train_gbdt.py
```

### 3. Run end-to-end pipeline

```bash
python src/pipeline/run_pipeline.py
```

---

## What This Demonstrates

* Ranking system architecture
* Multi-objective optimization
* Causal reasoning under partial feedback
* Production constraints (latency, guardrails)
* Policy-centric evaluation

This project intentionally prioritizes **system design correctness**
over leaderboard-style optimization.

---

## Interview Mapping

| Question                        | Where Answered         |
| ------------------------------- | ---------------------- |
| How do you rank a feed?         | `architecture.md`      |
| How do you handle tradeoffs?    | `ranking_tradeoffs.md` |
| How do you evaluate offline?    | `counterfactuals.md`   |
| How do you prevent regressions? | `ab_guardrails.md`     |

---

## Disclaimer

This project uses **synthetic data** and simplified components.
It is intended for **educational and portfolio purposes**,
not as a production deployment.

---

## Author

Built as a **FAANG-style ranking systems portfolio project**
focused on decision-making, evaluation, and system ownership.
