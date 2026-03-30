# ⚡ LLM Inference Optimization Benchmarker

<div align="center">

![Banner](https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=4,8,16&height=200&section=header&text=LLM%20Inference%20Benchmarker&fontSize=44&fontColor=fff&animation=twinkling&fontAlignY=35&desc=Quantization%20%E2%80%A2%20Batching%20%E2%80%A2%20KV-Cache%20%E2%80%A2%20Roofline%20Analysis%20%E2%80%A2%20Production%20Configs&descAlignY=55&descSize=14)

<p>
  <img src="https://img.shields.io/badge/Python-3.9%2B-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/Streamlit-1.35-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white"/>
  <img src="https://img.shields.io/badge/Roofline%20Model-Included-f59e0b?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/vLLM-Compatible-22C55E?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Models-0.5B%20to%20175B-7c3aed?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/License-MIT-22C55E?style=for-the-badge"/>
</p>

<p>
  <b>Benchmark and compare LLM inference configurations across quantization levels, batch sizes, KV-cache strategies, and hardware profiles — with roofline model analysis and production-ready deployment configs.</b>
</p>

</div>

---

## 🌟 Why This Project?

Deploying LLMs in production involves dozens of tradeoffs that most engineers discover the hard way. This tool makes them visible:

```
❌ Without benchmarking:
   Engineer deploys FP32, batch=1 → 12 tokens/sec → $0.05/1K tokens → unhappy users

✅ With this tool:
   Engineer finds INT4 + batch=16 + continuous batching → 180 tokens/sec → $0.003/1K tokens
   → 15x throughput, 94% cost reduction, same model quality
```

Key question this answers: **for your specific model, hardware, and workload — what's the optimal deployment configuration?**

---

## ✨ Features

| Feature | Description |
|---|---|
| 🔢 **Quantization Comparison** | FP32 / FP16 / INT8 / INT4 — throughput, latency, memory, quality |
| 📦 **Batch Size Analysis** | Sweep batch sizes 1–32, see throughput curves and GPU utilization |
| 🗄️ **KV-Cache Analysis** | Measure the latency reduction from caching attention keys/values |
| 🔄 **Continuous Batching** | Quantify the throughput gain from interleaving requests |
| 📐 **Roofline Model** | Visualize whether your workload is memory-bound or compute-bound |
| ⏱️ **Latency vs Throughput** | Pareto frontier scatter showing tradeoff space |
| 💾 **Memory Profiling** | GPU memory requirements with OOM risk warnings |
| 💸 **Cost Analysis** | $/1K tokens for each configuration on real hardware costs |
| 🔬 **Hardware Profiles** | A100, H100, RTX 4090, A10G, CPU |
| 📋 **Production Config** | Auto-generates optimal vLLM deployment command |
| ⬇️ **JSON Export** | Download all results for further analysis |

---

## 🖥️ Demo

```
╔══════════════════════════════════════════════════════════════════╗
║  ⚡ LLM Inference Benchmarker                                    ║
║  Model: 7B Llama · Hardware: A100 80GB                          ║
╠══════════════════════════════════════════════════════════════════╣
║                                                                  ║
║  🏆 Best Results                                                 ║
║  ┌──────────┬────────────┬───────────┬───────────────────┐      ║
║  │ 847 t/s  │  12.3ms    │  3.5 GB   │  $0.0003 / 1K     │      ║
║  │ Peak TPS │ Min Latency│ Min Memory│ Min Cost          │      ║
║  └──────────┴────────────┴───────────┴───────────────────┘      ║
║                                                                  ║
║  📊 Throughput Curve (batch size vs t/s)                         ║
║  INT4: ████████████████████ 847 t/s                             ║
║  INT8: ████████████████     623 t/s                             ║
║  FP16: ████████             312 t/s                             ║
║  FP32: ████                 184 t/s                             ║
║                                                                  ║
║  💡 Key Insight: INT4 + batch=16 + continuous batching          ║
║     is 4.6x faster than FP32 + batch=1 baseline                ║
╚══════════════════════════════════════════════════════════════════╝
```

---

## 📦 Installation

```bash
git clone https://github.com/YOUR_USERNAME/llm-inference-optimizer.git
cd llm-inference-optimizer
pip install -r requirements.txt
streamlit run app.py
```

---

## 🧠 The Roofline Model — Why It Matters

The roofline model is the foundational analysis for inference optimization. Every LLM workload is bounded by one of two things:

```
Memory Bandwidth Bound:          Compute Bound:
  Too slow moving weights           Too many FLOPs per token
  → Quantize (INT4/INT8)           → FlashAttention, fused kernels
  → Increase batch size             → Better hardware (H100 > A100)
  → Better memory bandwidth         → Model distillation
```

```
Performance
(TFLOPS)     │           ████████████████  ← Compute ceiling
             │        ███
             │     ███   ← Bandwidth slope
             │  ███
             │██
             └─────────────────────────────────► Arithmetic Intensity
                Memory-bound     Compute-bound  (FLOP/byte)
```

Points on the slope = memory-bound (most LLM decode scenarios)
Points on the ceiling = compute-bound (batch prefill scenarios)

---

## 📊 Quantization Quick Reference

| Format | Memory | Speed | Quality Loss | Use When |
|---|---|---|---|---|
| FP32 | 4 bytes/param | 1x | 0% | Debug only |
| FP16 | 2 bytes/param | 1.8x | ~0.1% | Production baseline |
| INT8 | 1 byte/param | 2.8x | ~0.4% | Cost-sensitive production |
| INT4 | 0.5 bytes/param | 3.6x | ~1.1% | High-throughput, cost-critical |

---

## 🔑 Key Optimization Concepts

**KV-Cache:** During autoregressive decoding, attention keys and values for previous tokens are cached to avoid recomputation. This reduces FLOPs per decode step from O(n²) to O(n) — critical for long contexts.

**Continuous Batching (iteration-level batching):** Instead of waiting for all requests in a batch to finish before starting new ones, requests are added/removed mid-batch at each token. Pioneered by [Orca](https://arxiv.org/abs/2207.04836), implemented in vLLM. Typically gives 2-3x throughput improvement.

**PagedAttention:** vLLM's memory management technique — treats KV-cache like virtual memory pages, enabling near-zero memory waste and efficient batching. Enable with `--block-size 16`.

---

## 🏭 Supported Hardware Profiles

| Hardware | Mem BW | FP16 TFLOPS | VRAM |
|---|---|---|---|
| NVIDIA H100 80GB | 3.35 TB/s | 989 | 80 GB |
| NVIDIA A100 80GB | 2.0 TB/s | 312 | 80 GB |
| NVIDIA RTX 4090 | 1.0 TB/s | 165 | 24 GB |
| NVIDIA A10G | 0.6 TB/s | 62 | 24 GB |
| CPU (64-core) | 0.05 TB/s | 1 | 512 GB |

---

## 📁 Project Structure

```
llm-inference-optimizer/
│
├── app.py              # 🧠 Full Streamlit app — simulation + 5 analysis tabs
├── requirements.txt    # 📦 4 minimal dependencies
├── .gitignore
├── LICENSE
└── README.md
```

---

## 🛠️ Tech Stack

| Technology | Purpose |
|---|---|
| Streamlit | Interactive web UI |
| Plotly | Interactive charts (throughput curves, roofline, scatter) |
| Pandas | Results dataframes |
| NumPy | Roofline model computation |

**No GPU required** — the simulation engine uses roofline model math to produce realistic performance estimates based on published hardware specs.

---

## 🗺️ Roadmap

- [ ] Real benchmarking mode — actually load and run HuggingFace models
- [ ] vLLM, TGI, TensorRT-LLM backend comparison
- [ ] FlashAttention-2 vs standard attention analysis
- [ ] Multi-GPU tensor parallelism modeling
- [ ] Speculative decoding speedup estimation
- [ ] Export comparison report as PDF

---

## 📄 License

MIT — see [LICENSE](LICENSE)

---

<div align="center">

**⭐ Star this repo if you find it useful!**

*Understanding inference bottlenecks is the first step to fixing them.*

![Footer](https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=4,8,16&height=100&section=footer)

</div>
