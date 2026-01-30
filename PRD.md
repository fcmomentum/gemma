
# PRD: Split-Brain Prophet Transformer Layer

**Version:** 1.0
**Status:** Draft
**Owner:** Engineering Lead / Research Team
**Date:** January 29, 2026

## 1. Executive Summary

The "Split-Brain" architecture introduces a specialized attention mechanism designed to improve the reasoning and planning capabilities of LLMs. By bifurcating specific layers into a **Teacher Stream** (standard causal history) and a **Student Stream** (noisy, future-predicting), we force the model to learn robust global representations and "pre-load" future semantic states before the final output generation.

## 2. Problem Statement

Standard Decoder-only models (GPT) rely solely on Next-Token Prediction (NTP) at the final layer. This leads to:

1. **Short-term bias:** Models prioritize local syntax over global coherence.
2. **Inefficient Planning:** The model cannot "think ahead" without outputting tokens.
3. **feature Sparsity:** Attention heads often learn redundant features (simple copying).

## 3. Technical Specifications

### 3.1. Architecture Overview

The core modification is the replacement of the standard `SelfAttention` module with a `SplitBrainAttention` module in the **middle-to-late layers** of the network.

**Component Diagram:**
`Input -> [Teacher Head] & [Student Head] -> [Gated Fusion] -> [Residual + Norm] -> FFN`

### 3.2. Detailed Requirements

#### A. The Teacher Stream (The "Control")

* **Input:** Standard Input Embedding .
* **Mechanism:** Standard Multi-Head Attention (MHA) or Grouped Query Attention (GQA).
* **Masking:** Standard Causal Mask (Lower Triangular).
* **Output:** .
* **Constraint:** Gradients flow normally.

#### B. The Student Stream (The "Prophet")

* **Input:** Same Input Embedding .
* **Mechanism:** Duplicate MHA/GQA structure (same dimension ).
* **Masking Logic (Crucial):**
* **Base:** Causal Mask (Lower Triangular).
* **Augmentation:** Random Token Masking (Bernoulli distribution).
* **Mask Ratio:** Configurable  (default: 15%).
* **Constraint:** Current position  must *never* see .


* **Objective Target:**
* The Student output  is regressed against the **Teacher output** at .
* **Stop-Gradient:** Applied to the Teacher target to prevent collapse.
* **Loss:** MSE or Cosine Similarity.



#### C. Gated Fusion Mechanism

* **Input:** Concatenation of .
* **Operation:** Learned Vector Gate.



* **Initialization:** Bias  initialized to `+2.0` (sigmoid ) to prioritize Teacher stability at start of training.

### 3.3. Layer Placement Strategy

* **Target Layers:** The block should be inserted at **75% depth** (e.g., layers 9–11 in a 12-layer model).
* **Rationale:** Capitalize on high-level semantic features; avoid disrupting low-level syntax formation.

---

## 4. Engineering Implementation

### 4.1. Hyperparameters

| Parameter | Symbol | Default Value | Description |
| --- | --- | --- | --- |
| **Mask Ratio** |  | `0.15` | Probability of masking a token in Student stream. |
| **Prophet Weight** |  | `0.1` | Weight of auxiliary loss relative to main CE loss. |
| **Target Shift** |  | `+1` | Time-step shift (Student  predicts Teacher ). |
| **Stop Grad** | - | `True` | Detach Teacher target from computation graph. |

### 4.2. Training Loop Modification

The forward pass must return two values: the **Logits** (for CE Loss) and the **Auxiliary Loss** (from the Split-Brain layers).

```python
# Pseudo-code for Loss Calculation
def compute_loss(model_output, targets):
    # 1. Standard Next-Token Prediction
    loss_ce = CrossEntropy(model_output.logits, targets)

    # 2. Auxiliary Prophet Loss (Summed over all Split Layers)
    loss_prophet = 0
    for layer in model.split_layers:
        # L2 Norm between Student(t) and Detached_Teacher(t+1)
        loss_prophet += MSE(layer.student_out[:-1], layer.teacher_out[1:].detach())

    # 3. Total Optimization Objective
    return loss_ce + (lambda_param * loss_prophet)

```

### 4.3. Inference Strategy

* **Mode:** Deterministic.
* **Action:** Disable Random Masking in the Student Stream.
* **Input:** Student sees the exact same clean input as the Teacher.
* **Reasoning:** The Student acts as a "second opinion" or "refiner," reinforcing the features it learned to predict during training.

---

## 5. Success Metrics & Validation

### 5.1. Primary Metrics (Quantitative)

1. **Perplexity (PPL):** Must show improvement (lower PPL) over baseline model with equal parameter count.
2. **Training Efficiency:** Reaching baseline PPL in fewer training steps (measuring sample efficiency).
3. **Long-Context Recall:** Performance on "Needle in a Haystack" tests (verifying if Prophet head improves memory).

### 5.2. Secondary Metrics (Qualitative/Reasoning)

1. **GSM8K / MATH:** Logic benchmarks to test "planning" capability.
2. **ARC (Abstraction and Reasoning Corpus):** Measures robustness to novel patterns.

---

## 6. Risks & Mitigation

| Risk | Probability | Impact | Mitigation Strategy |
| --- | --- | --- | --- |
| **Training Instability** | Medium | High | Use Gate initialization bias to `1.0` (Teacher-only); Ramp up  (loss weight) slowly (warmup). |
| **Compute Overhead** | High | Medium | Split-Brain increases FLOPs by ~30% for that specific layer. Limit implementation to only 2-3 layers max. |
| **Info Leakage** | Low | Critical | Add unit tests to verify Student mask is strictly Causal + Random. Ensure  target is never input. |
| **Inference Drift** | Medium | Medium | If Student hurts inference (due to lack of masks), implement a `scale` parameter to dampen Student contribution at runtime. |

## 7. Development Roadmap

* **Phase 1: Proof of Concept (Week 1)**
* Implement `SplitBrainBlock` in a small NanoGPT (10M params).
* Verify gradient flow and loss convergence.


* **Phase 2: Ablation Study (Week 2)**
* Train Baseline vs. Split-Brain on OpenWebText.
* Compare Validation Loss curves.


* **Phase 3: Scaling (Week 3-4)**
* Integrate into 1B parameter training run.
* Evaluate on standard benchmarks (HellaSwag, MMLU).
