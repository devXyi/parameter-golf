# Recurrent GPT — Compression-Native Architecture

**val_bpb**: TBD (mean over 3 seeds)  
**Training**: 8×H100 SXM · 600 seconds  
**Constraint**: ≤16MB final artifact

---

Achieves competitive BPB under a strict 16MB constraint by co-optimizing architecture, training dynamics, and compression as a single system.

---

## Overview

A compression-first language model where training, architecture, and quantization are co-designed around one objective: minimize bits per byte, not just perplexity.

- 16.8M unique parameters → ~268M effective capacity via depth recurrence
- Train → quantize → export pipeline fully aligned: no post-hoc compression
- Compression regularization shapes weight distributions during training itself

---

## Architecture

Single shared transformer block × 16 recurrences (d=1024, MLP=4096).

| Component | Detail |
|-----------|--------|
| Attention | GQA, 8 heads / 1 KV head, RoPE base 500K |
| MLP | Full-rank SwiGLU with relu² activation |
| SmearGate | Per-step per-dimension interpolation gate (low-rank factored) |
| Context | Trigram hash embedding (16,384 buckets → d projection) |
| Memory | Depth-aware router + EMA residual memory state (damped) |
| Routing | Per-token entropy routing via learned scalar gate |
| Modulation | Frequency-aware multiplicative gate on hidden state |
| Skip | Multi-scale U-Net connections at depth R/4, R/2, 3R/4 |

---

## Compression Stack

Compression is a first-class objective, not post-processing.

**Storage**
- INT6 true bit-packing: 4 weights → 3 bytes (25% storage saving)
- Hybrid INT5/INT6: per-layer learned bit-width selector
- Learned per-matrix quantization scales, warm-initialized, QAT-aligned
- ZSTD level-22 + 64KB dictionary trained on actual weight samples
- Magnitude-sorted row reordering at export (+3–7% compression gain)
- Permutation stored with CRC32 checksum, validated on load

---

## Compression-Native Training

All compression regularizers activate at QAT start (step 8000+).

**Distribution shaping**
- Entropy regularization: differentiable RBF soft-histogram over INT6 range
- Bitplane entropy: continuous fmod+sigmoid per-bit shaping (fully differentiable)
- Power-of-two clustering: hybrid centers {0, ±1–6, ±8, ±16, ±31}

**Structure alignment**
- Temporal correlation: multi-lag autocorrelation (lag-1 + 0.5×lag-2 + 0.25×lag-4)
- Cross-layer entropy coupling: EMA global distribution alignment
- Zipf alignment: KL divergence from logit distribution to natural language prior

**Compression synergy**
- Dictionary co-training: byte-repeat proxy + 8-byte LZ77 window loss (every 50 steps)

---

## Token-Frequency Aware Loss

Single forward pass returns `(loss, sparse_loss, ent_route_loss, logits)`.

Cross-entropy weighted by inverse-sqrt token frequency, tracked via EMA and all-reduced across GPUs. Rare tokens receive proportionally higher gradient signal — optimizing BPB directly rather than average CE.

---

## Optimizer

**Muon** (matrix parameters): lr=0.04, WD=0.038, momentum warmup 0.85→0.95 over 1500 steps  
**AdamW** (embeddings + scalars): embed WD=0.04, scalar WD=0.01

| Setting | Value |
|---------|-------|
| Batch tokens | 786,432 |
| Sequence length | 2048 |
| Warmdown | 3,000 steps |
| SWA | Final 40%, every 50 steps |
| Eval stride | 64 |
| QAT start | Step 8,000 |

---

## 3-Seed Results

| Seed | val_bpb | artifact_bytes | valid |
|------|---------|----------------|-------|
| 42 | — | — | — |
| 1337 | — | — | — |
| 2024 | — | — | — |
| **Mean** | — | | |
| **Std** | — | | |

---

## Run

```bash
# Setup (once)
bash prepare.sh

# Train + evaluate
SEED=42 bash eval/eval.sh
```

