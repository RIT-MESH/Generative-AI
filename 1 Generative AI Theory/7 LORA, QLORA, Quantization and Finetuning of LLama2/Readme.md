# Deep Dive: LoRA, QLoRA, Quantization, and Fine-Tuning for LLaMA 2



https://github.com/user-attachments/assets/e4f431dd-1099-4875-b685-e5fa4761fba0





## 1. LoRA (Low-Rank Adaptation)
LoRA adapts large pre-trained models efficiently by introducing small, trainable updates to specific layers without altering the original weights.

### Mechanics
- **Target**: Applied to transformer weight matrices (e.g., `W_q`, `W_k`, `W_v`, `W_o`).
- **Core Idea**:
  `W' = W + ΔW`, where `ΔW = A * B`
  - `A` is in `ℝ^(d×r)`
  - `B` is in `ℝ^(r×k)`
  - `r << min(d, k)`
- **Training**: Freeze `W`, update only `A` and `B`.

### Math Insight
- Parameters updated by LoRA: `d * r + r * k`
- Example (d = 4096, k = 4096, r = 8): LoRA updates ~65K parameters vs. 16M for full fine-tuning.

### In LLaMA 2
- Efficiently fine-tunes attention weights in large models (e.g., 70B).
- Use Case: Fine-tune on GitHub code with low VRAM.

### Hyperparameters
- **Rank (r)**: Controls capacity.
- **Scaling (alpha)**: Use `(alpha / r) * A * B` to stabilize training.

---

## 2. QLoRA (Quantized Low-Rank Adaptation)
QLoRA integrates quantization with LoRA to enable fine-tuning large models on limited hardware.

### Mechanics
- **Step 1: Quantization** to 4-bit (e.g., INT4).
  - Weights are stored as integers and dequantized to FP16 during computation.
- **Step 2: Apply LoRA** in higher precision.

### Innovations
- **NF4**: 4-bit format optimized for normal distributions.
- **Double Quantization**: Compresses scaling factors.
- **Paged Optimizers**: Efficient GPU memory usage.

### Math Insight
```
w_q = round((w - min) / scale)
w ≈ min + w_q * scale
```
- 70B FP16 model (~140GB) → QLoRA (~35GB).

### In LLaMA 2
- Enables fine-tuning LLaMA 2-70B on 24GB GPUs.
- Use Case: Fine-tune for customer support.

### Trade-offs
- Small accuracy drop due to quantization noise, often compensated by LoRA.
- Requires libraries like `bitsandbytes` for implementation.

---

## 3. Quantization
Reduces numerical precision for efficient deployment and inference.

### Mechanics
- **Types**:
  - Symmetric: `w_q = round(w / scale)`
  - Asymmetric: `w_q = round((w - zero) / scale)`
- **Block-wise**: Shared scaling factors across groups.
- **PTQ vs. QAT**:
  - PTQ: Post-training quantization (fast but less accurate).
  - QAT: Quantization-aware training (simulates quantization during training).

### Math Insight
```
scale = (max - min) / 255
w_q = round((w - min) / scale)
```
- Error = `|w - (min + w_q * scale)|`

### In LLaMA 2
- FP16 → INT8 or 4-bit reduces model size (e.g., 13B → ~6.5GB).
- Use Case: Run LLaMA 2-7B on Raspberry Pi (offline, low-memory).

### Challenges
- Accuracy drop, especially in 4-bit without QAT.
- INT8 hardware support is more common than INT4.

---

## 4. Fine-Tuning
Updates **all** model parameters, giving full flexibility but at high compute/memory cost.

### Mechanics
- Start with pre-trained LLaMA 2.
- Use task-specific loss (e.g., cross-entropy).
- Optimize all weights with gradient descent (e.g., AdamW).

### Math Insight
```
Loss: L = -∑ y * log(y_hat)
Update: W_new = W_old - learning_rate * ∇L
```
- Full 70B model: needs ~140GB VRAM (plus optimizer states).

### In LLaMA 2
- Requires A100 cluster.
- Risk of overfitting or catastrophic forgetting (mitigated with dropout/regularization).

### Comparison
- LoRA updates ~0.1% of parameters.
- QLoRA = LoRA + 4-bit quantized model.

---

## Practical Example: LLaMA 2 for Legal Summarization
| Technique           | VRAM Usage | Epochs | Notes                          |
|---------------------|-------------|--------|---------------------------------|
| Full Fine-Tuning    | 26GB        | 10     | Multi-GPU setup                |
| LoRA                | 16GB        | 2      | `r = 16`                       |
| QLoRA               | 12GB        | 2      | 4-bit base + LoRA adapters     |
| Quantized Inference | ~6.5GB      | N/A    | Deployment only                |

---

## Summary Table
| Technique      | Parameters Updated | Memory Use      | Compute Cost | Accuracy Trade-off | LLaMA 2 Use Case        |
|----------------|--------------------|-----------------|--------------|--------------------|-------------------------|
| Fine-Tuning    | All (e.g., 70B)    | Very High       | Very High    | None               | Full task adaptation    |
| LoRA           | Few (e.g., 65M)    | Low             | Low          | Minimal            | Efficient fine-tuning   |
| QLoRA          | Few (e.g., 65M)    | Very Low        | Low          | Small              | Single-GPU fine-tuning  |
| Quantization   | None               | Very Low        | None         | Moderate           | Deployment optimization |

---

