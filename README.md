# babbletest

The project’s goal is achieving absolute parity between a high-level GPT implementation and a ground-up, simplified version.

---

# GPT Logic Parity: Simplified vs. Original

This project demonstrates a bit-perfect reconstruction of a Nano-GPT architecture. It verifies that a heavily condensed 100-line implementation maintains 100% mathematical parity with the original codebase over 30 training steps.

## 🎯 The Objective

To prove that complex Transformer logic (Multi-head Attention, RMSNorm, Adam Optimizer) can be simplified for readability and "code-golfed" without losing a single decimal point of precision.

This is an exercise in "logic compression," standing entirely on the shoulders of **Andrej Karpathy** and his legendary [karpathy/microgpt.py](https://gist.github.com/karpathy/8627fe009c40f57531cb18360106ce95) work. We have taken Karpathy’s masterfully engineered original, and leveraged advanced AI to refine and reduce the character count without sacrificing any of it's mathematical integrity. We hold the Karpathy original work in the highest regard; it remains the definitive blueprint for anyone seeking to understand the inner workings of modern transformers.

## 🛠 Features

* **Micro-Autograd Engine:** A custom `Value` class with full operator overloading (`__add__`, `__mul__`, `__pow__`, etc.) and a topological sort for backpropagation.
* **GPT Architecture:** * 16-dimensional embeddings.
* 4-head Self-Attention mechanism.
* Adam optimizer with Cosine Learning Rate decay.

## 📊 Verification Metrics (30 Steps)

Running `fast_verify_03.py` compares both models across four critical dimensions:

| Metric | Status | Result |
| --- | --- | --- |
| **Final Loss** | ✅ PASS | `2.641596` |
| **Weight Sum** | ✅ PASS | `18.099116` |
| **Param Count** | ✅ PASS | `4064` |
| **Inference** | ✅ PASS | Bit-identical character output |

## 🚀 Quick Start

1. **Run the Parity Check:**
```bash
python fast_verify_03.py

```

2. **Configuration:**
The hyperparameters are set for a quick "sanity check" (30 steps). For full training to produce more realistic names, adjust `num_steps` to `500` in `simplified.py`:
```python
num_steps = 30 
n_embd, n_head, n_layer, block_size, lr = 16, 4, 1, 8, 1e-2

```

## 📝 Key Lessons

* **Precision:** Small changes in operator overloading (like `__radd__`) are required to make standard Python functions like `sum()` compatible with custom Autograd objects.
* **Reproducibility:** By fixing `random.seed(42)`, we ensure that even the randomly initialized weights are identical across different script versions.
