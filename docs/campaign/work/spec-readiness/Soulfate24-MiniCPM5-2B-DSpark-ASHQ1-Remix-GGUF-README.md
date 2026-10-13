---
license: apache-2.0
language:
  - en
  - zh
library_name: transformers
pipeline_tag: text-generation
base_model:
  - openbmb/MiniCPM5-2B
  - openbmb/MiniCPM5-2B-DSpark
tags:
  - minicpm
  - minicpm5
  - llama
  - text-generation
  - long-context
  - tool-calling
  - on-device
  - edge-ai
  - dspark
  - quantization
  - gguf
  - ashq1
  - imatrix
---

**MiniCPM5-2B with DSpark - ASHQ1-Remix**

This is a GGUF quantized version of the original model.

## 📈 Release Benchmarks (wiki.test.raw, symmetric FA-auto reference)

| Model | Size | PPL | KLD | RMS Δp | top-p | Speed |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| Q8_0 (stock) | 4229 MiB | 34.4742 | 0.0105 | 2.39% | 95.6% | 353 t/s |
| **Fidelity-48pc** 🥈 | 3824 MiB | 34.3246† | 0.0247 | 3.42% | 94.0% | 371 t/s |
| **Precision-42pc** ⭐ | 3384 MiB | 34.3791† | 0.0331 | 3.94% | 92.5% | 382 t/s |
| Q6_K-imx (stock) | 3266 MiB | 34.4411† | 0.0335 | 3.99% | 92.4% | 406 t/s |
| **Quality-36pc** | 2849 MiB | 34.6224 | 0.0743 | 5.80% | 88.3% | 466 t/s |
| Q5_K_M-imx (stock) | 2849 MiB | 34.6224 | 0.0743 | 5.80% | 88.3% | 470 t/s |
| **Compact-33pc** | 2630 MiB | 34.6636 | 0.1441 | 7.98% | 83.8% | 512 t/s |
| **Mini-30pc** | 2391 MiB | 33.5214† | 0.1728 | 8.66% | 81.8% | 557 t/s |
| IQ4_XS-imx (stock) | 2268 MiB | 34.8506 | 0.1813 | 9.17% | 81.2% | 457 t/s |
| **Nano-27pc** ✗ | 2152 MiB | 33.7551† | 0.2191 | 10.12% | 78.6% | 453 t/s |
| IQ3_M-imx (stock) | 1985 MiB | 36.4366 | 0.4492 | 14.31% | 70.5% | 539 t/s |
| **Pico-24pc** ✗ | 1913 MiB | 37.8642 | 0.3878 | 13.31% | 72.6% | 545 t/s |

## ℹ️ About ASHQ1-Remix Suite
Activation-aware GGUF quantization whose every ratio, floor, and cap traces to a measured experiment. Plain-BF16-native first; AutoRound lineage supported with explicit saturation bounds. Full seven-tier ladder validated across six model families.

🔗 Link: https://huggingface.co/Soulfate24/AutoRound-ASHQ1-Remix_Double-Quantization_Suite