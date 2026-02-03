# A/B Testing Guardrails for Feed Ranking Systems

## Purpose

A/B testing is used to validate ranking policy changes online.
However, optimizing for a single primary metric (e.g., CTR) can cause
unintended regressions in user experience, system stability, or long-term value.

This document defines **guardrail metrics** that must not regress
when shipping ranking changes.

> Principle: *No experiment ships based on lift alone.*

---

## 1. Experiment Setup

### Unit of Randomization
- User-level randomization
- Sticky assignment across sessions

### Traffic Split
- Control: Existing ranking policy
- Treatment: New ranking policy
- Typical split: 90% / 10% (ramp-up recommended)

### Duration
- Minimum: 7 days
- Required to capture:
  - Weekday / weekend effects
  - Short-term novelty decay

---

## 2. Primary Metrics (Optimization Targets)

These metrics define *success* but **do not guarantee safety**.

| Metric | Definition | Rationale |
|------|-----------|-----------|
| CTR | Clicks / Impressions | Measures immediate relevance |
| Session Time | Avg minutes per session | Proxy for engagement depth |
| D1 / D7 Retention | Return probability | Captures medium-term value |

Primary metrics **may improve**, but guardrails **must not regress**.

---

## 3. Guardrail Metrics (Hard Constraints)

### 3.1 User Experience Guardrails

| Metric | Constraint | Why It Matters |
|-----|-----------|---------------|
| Bounce Rate | ≤ +1% | Prevent clickbait ranking |
| Scroll Depth | ≥ baseline | Avoid shallow engagement |
| Repeated Content Rate | ≤ baseline | Prevent fatigue |

> Rationale: CTR gains from low-quality or repetitive content
are often short-lived and harmful.

---

### 3.2 Diversity & Ecosystem Health

| Metric | Constraint | Why It Matters |
|------|-----------|---------------|
| Topic Entropy | ≥ baseline | Prevent filter bubbles |
| Creator Gini | ≤ baseline | Avoid exposure collapse |
| Long-tail Exposure | ≥ baseline | Protect ecosystem health |

> Ranking systems shape content ecosystems, not just clicks.

---

### 3.3 System & Reliability Guardrails

| Metric | Constraint | Why It Matters |
|------|-----------|---------------|
| P95 Latency | ≤ +10ms | Ranking must stay real-time |
| Error Rate | ≤ baseline | No reliability regressions |
| Timeout Rate | ≤ baseline | Protect availability |

> A better model that violates latency budgets is not shippable.

---

## 4. Statistical Decision Rules

### Significance
- Primary metrics: 95% confidence
- Guardrails: **No statistically significant degradation allowed**

### Shipping Criteria
An experiment may ship **only if**:
1. At least one primary metric improves **significantly**
2. No guardrail metric regresses **significantly**
3. No sustained negative trend is observed during ramp-up

---

## 5. Ramp Strategy

### Recommended Rollout
1. 1% traffic (24 hours)
2. 10% traffic (48–72 hours)
3. 50% traffic
4. 100% rollout

Rollback is triggered automatically if:
- Any guardrail crosses threshold
- Latency budget is violated
- Error rate spikes

---

## 6. Common Failure Modes (Lessons Learned)

### CTR-Only Optimization
- Increases bounce rate
- Reduces session depth
- Harms long-term retention

### Over-Diversification
- Lowers relevance
- Increases cognitive load
- Degrades short-session users

### Ignoring Latency
- Cancels engagement gains
- Impacts downstream systems

> Guardrails exist because **offline wins often fail online**.

---

## 7. Relationship to Offline Evaluation

Offline evaluation (IPS / SNIPS):
- Estimates policy value
- Reduces experiment risk
- Narrows candidate space

Online A/B testing:
- Validates real user impact
- Detects system-level regressions
- Final shipping authority

> Offline evaluation narrows options; online experiments decide.

---

## 8. Summary

A successful ranking experiment:
- Improves primary engagement metrics
- Preserves user experience
- Maintains ecosystem health
- Respects system constraints

Guardrails are not optional — they are **production requirements**.

