### CHANGELOG.md

**v1.0: Port**

* Mirrored Karpathy’s original GPT logic.
* **Anchored** `random.seed(42)` at the script start to guarantee identical weight initialization.
* Result: Functional baseline with deterministic outputs.
* Variable name minification to reduce character counts.
* Line count = 98.

**v2.0: Compression**

* Compacted `Value` engine; added `__radd__` for `sum()` compatibility.
* Condensed Transformer blocks into list comprehensions.
* Result: Massive line count reduction; parity confirmed at 3 steps.
* Line count = 87.

**v3.0: Readability**

* Restored descriptive variable names (`other`, `logits`, `token_id`).
* Standardized hyperparameter block for quick experimentation.
* Result: Bit-perfect parity at 30 steps.
* Line count = 93.
