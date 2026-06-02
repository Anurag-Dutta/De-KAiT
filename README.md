<div align="center">

# De-KAiT

### Data-Efficient Kolmogorov–Arnold Image Transformer for Visual Classification of Encrypted and Obfuscated Network Traffic Flows

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.4-orange?logo=pytorch)](https://pytorch.org/)
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)
[![IEEE TMLCN](https://img.shields.io/badge/Submitted-IEEE%20TMLCN-blueviolet)]()
[![DRDO Funded](https://img.shields.io/badge/Funded-DRDO%20India-green)]()

**Anurag Dutta¹ · Suneha Raj Molaka² · Sangita Roy³ · Rajat Subhra Chakraborty¹**

¹ Dept. of CSE, IIT Kharagpur &nbsp;|&nbsp; ² Dept. of EE, IIT Kharagpur &nbsp;|&nbsp; ³ Dept. of CSE, TIET Patiala

</div>

---

## Overview

Network traffic classification is a fundamental task in network and communication security.
With the rapid proliferation of encrypted and anonymized traffic (TLS/HTTPS, Tor, commercial VPN
services), payload-centric Deep Packet Inspection (DPI) has become largely infeasible, necessitating
a shift toward **flow-level** analysis.

**De-KAiT** proposes a new visual formulation of flow-level traffic classification using
**recurrence plots** — single-channel structured images that encode pairwise similarities between
normalized packet states (inter-packet arrival timestamps and payload sizes). On top of this
representation, De-KAiT is a lightweight, data-efficient vision transformer combining three ideas:

- 🔁 **Recurrence-Plot Patch Embedding** — single-channel Conv2D patch tokenization directly from recurrence plots, without artificial RGB expansion, preserving flow-level geometry even under Tor/VPN obfuscation
- 🎯 **Dual-Token Self-Attention** — a `[CLS]` (classification) + `[DIST]` (distillation) dual-token mechanism for robust, low-data flow-level aggregation, stabilizing optimization in scarce-data settings
- 🧠 **KAN Feed-Forward Sub-layer (KAN-FFN)** — replaces the fixed-activation MLP with Kolmogorov–Arnold Network-based edge-wise learnable univariate activations, better suited to the non-Gaussian, class-dependent token distributions of recurrence plots

<div align="center">
  <img src="FIGS/arch.png" alt="De-KAiT Architecture" width="88%"/>
  <br/>
  <em>Figure 1: De-KAiT architecture. A 224×224 single-channel recurrence plot is partitioned into
  196 non-overlapping 16×16 patches, linearly projected to d-dimensional tokens, augmented with
  learnable [CLS] and [DIST] tokens and positional embeddings, and processed by L stacked De-KAiT
  encoder blocks (MHSA + KAN-FFN). Final prediction is obtained by averaging the [CLS] and [DIST]
  head outputs.</em>
</div>

---

## Recurrence Plots

Each network flow is converted to a **224×224 single-channel recurrence plot** using two
packet-level features: normalized inter-packet arrival timestamps and normalized payload sizes.
The plot encodes pairwise state similarity via a Heaviside threshold (ε = 0.20), followed by
spatial resampling.

The resulting representations are visually and structurally discriminative:

| Traffic Class | Recurrence Pattern |
|---|---|
| **Browsing** | Irregular checkerboard; no diagonal dominance (bursty HTTP/HTTPS) |
| **Chat** | Structured diagonal + off-diagonal blocks (idle/active alternation) |
| **File Transfer** | Dense, uniformly tiled parallel diagonals (bulk transfer regularity) |
| **Video** | Large solid blocks with sharp transitions (frame-rate-driven stationarity) |
| **VoIP** | Fine-grained diagonal stripes at high global density (constant codec timing) |

Critically, **broad class-specific global geometry is preserved** even under Tor encryption and
VPN protection, where local textures change significantly.

<div align="center">
  <img src="FIGS/rp.png" alt="Recurrence Plots" width="85%"/>
  <br/>
  <em>Figure 2: Recurrence plots generation pipeline.</em>
</div>

---

## Model Variants

| Variant | Embed Dim `d` | FFN Dim `d_ff` | Heads | Patches | Params (M) | FLOPs (G) |
|---|---|---|---|---|---|---|
| **De-KAiT-Tiny** | 128 | 256 | 4 | 196 | **2.94** | **1.11** |
| **De-KAiT-Small** | 160 | 320 | 4 | 196 | 4.58 | 1.71 |

Both variants are evaluated at encoder depths **L ∈ {2, 4, 6, 8}**, with KAN-FFN expansion ratio
`r = 2.0`, patch size `p = 16`, per-head dimension `d_h = 32`, and dropout `0.1`.

For each experimental configuration, both the **best-validation checkpoint** (`_best.pth`) and the
**final epoch checkpoint** (`_last.pth`) are provided to support direct inference, fine-tuning, and
reproducibility verification.

> 📦 **[Download Pre-Trained Weights (Google Drive)](https://drive.google.com/drive/folders/16-FH9uwxcavNENKFX9RsC34alaDvsBkD?usp=sharing)**
