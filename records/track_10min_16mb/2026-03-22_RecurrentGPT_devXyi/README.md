# Recurrent GPT — Compression-Native Architecture

**val_bpb**: TBD — 3-seed official run pending  
**Training**: 8×H100 SXM5 · 600 seconds · ≤16MB artifact

---

Achieves competitive BPB under a strict 16MB constraint by co-optimizing architecture, training dynamics, and compression as a single system.

---

## Overview

- 11.8M unique parameters → ~236M effective capacity via depth recurrence
- Train → quantize → export pipeline fully aligned: no post-hoc compression
- Compression regularization actively shapes weight distributions during training itself

**Confirmed from test runs on 8×H100 SXM5:**

| Metric | Value |
|--------|-------|
| Step time | ~80ms |
| Steps in 600s | ~7,400 |
| Artifact size | ~11–12 MB |
| INT6 compression | ~90% saved vs float32 |
| Quant gap (BPB) | ~0.01 |
| Best test BPB | 1.18 roundtrip (seed 1337) |

---

## Architecture

Single shared transformer block × 20 recurrences (d=768, MLP=3072).

| Component | Detail |
|-----------|--------|
| Attention | GQA, 6 heads / 2 KV heads, RoPE base 500K |
| MLP | Full-rank SwiGLU with relu² activation (4× expansion) |
| SmearGate | Per-step per-dimension interpolation gate (low-rank factored) |
| Context | Trigram hash embedding (16,384 buckets, dim=128 → d) |
| Memory | Depth-aware router + EMA residual memory state (damped) |
| Routing | Per-token entropy routing via learned scalar gate |
| Modulation | Frequency-aware multiplicative gate on hidden state |
| Skip | Multi-scale U-Net connections at depth R/4, R/2, 3R/4 |

---

## Compression Stack

| Layer | Detail |
|-------|--------|
| Packing | INT6 true bit-packing: 4 weights → 3 bytes |
| Precision | Hybrid INT5/INT6 with per-layer learned bit-width selector |
| Scales | Learned per-matrix quantization scales, QAT-aligned |
| Dictionary | ZSTD level-22 + 64KB dictionary trained on weight samples |
| Reordering | Magnitude-sorted row reordering at export (+3–7%) |
| Integrity | Permutation stored with CRC32, validated on load |

---

## Compression-Native Training

Activated at QAT start (step 1,200 ≈ 96s into training).

- Entropy regularization — differentiable RBF soft-histogram over INT6 range
- Bitplane entropy — continuous fmod+sigmoid per-bit shaping
- Power-of-two clustering — hybrid centers {0, ±1–6, ±8, ±16, ±31}
- Dictionary co-training — byte-repeat proxy + 8-byte LZ77 window loss
- Zipf alignment — KL divergence from logit distribution to natural language prior
- **TFAL** — per-token CE weighted by inverse-sqrt EMA token frequency

---

## Training Schedule (confirmed on 8×H100 SXM5)

| Event | Step | Wallclock | Trigger |
|-------|------|-----------|---------|
| Warmup complete | 80 | ~6s | step |
| QAT ON | 1,200 | ~96s | step |
| SWA ON | ~4,650 | ~372s (62%) | wallclock |
| Warmdown | ~5,400 | ~432s (72%) | wallclock |
| Cap | ~7,400 | 600s | hard cap |

**val_bpb progression (test run, seed 1337):**

| Step | val_bpb |
|------|---------|
| 800 | 1.4300 |
| 1600 | 1.3400 |
| 2400 | 1.3100 |
| 3200 | 1.2800 |
| Final (SWA) | **1.1700** |
| INT6 roundtrip | **1.1800** |

---

## Optimizer

**Muon** (matrix params): lr=0.025, WD=0.035, momentum warmup 0.85→0.98  
**AdamW** (embeddings + scalars): embed WD=0.035, scalar WD=0.01

| Setting | Value |
|---------|-------|
| Batch tokens | 1,572,864 |
| Sequence length | 4,096 |
| Grad clip | 0.3 |
| Warmdown steps | 1,200 |
| SWA every | 50 steps |

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
pip install zstandard sentencepiece
python data/cached_challenge_fineweb.py --variant sp1024
SEED=42 torchrun --standalone --nproc_per_node=8 train_gpt.py
```

All parameters set as defaults — no env vars needed.

