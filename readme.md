# NeuralNetwork from Scratch

This repository contains a collection of deep learning components and models implemented **from scratch**, inspired by Andrej Karpathy’s excellent educational series on building neural networks without frameworks.

It includes:
- A minimal **autograd engine** (`micrograd`) for scalar-based automatic differentiation.
- A fully functioning **MLP-based character-level language model** (`makemore`) trained on a dataset of names using only raw PyTorch tensors — no high-level APIs like `nn.Module`.

---

## 🔬 micrograd – Automatic Differentiation Engine

The `Value` class provides a lightweight reverse-mode autodiff engine capable of computing gradients of scalar expressions using backpropagation.

### Features:
- Operator overloading: `+`, `*`, `**`, `-`, `/`, `tanh()`, `exp()` etc.
- Gradient propagation via `.backward()`
- Manual construction of computation graphs
- Integrated with a simple graph visualizer (`draw_dot()`)

### Example:
```python
from engine import Value

x = Value(2.0, label='x')
y = Value(3.0, label='y')
z = x * y + x**2
z.label = 'z'
z.backward()
```

## 🤖 makemore_mlp – Character-Level Name Generator

This model learns to generate names, character-by-character, using a single-layer **MLP with BatchNorm** and **learned embeddings**.

### Model Architecture

- **Character Embedding Layer**
- **Fully Connected MLP Layer** (with Tanh activation and BatchNorm)
- **Output Layer** projecting to vocabulary logits
- Trained using **Cross-Entropy Loss**

---

### 📄 Dataset

- `names.txt`: A list of names, one per line.
- Tokenized with a special `.` token representing the end of a name.

---

### ⚙️ Hyperparameters

| Parameter        | Value   |
|------------------|---------|
| Block size       | 3       |
| Embedding size   | 15      |
| Hidden neurons   | 200     |
| Batch size       | 32      |
| Training steps   | 200,000 |

---

### 🧪 Training Procedure

During training, the model:

- Randomly samples mini-batches
- Trains using raw tensor operations (no `nn.Linear`, etc.)
- Applies **batch normalization manually**
- Tracks training loss over time
- Saves the trained model as `makemore.pth`

After training:

- Prints final loss on both training and dev sets
- Saves the final model state using `torch.save()`

---

### 🧪 Training & Evaluation

```bash
# Run training
python makemore_mlp.py
```

### 📦 Output Example

Once trained, you can generate new names using the learned model (name generation script not included yet):


```
Generated names:
- torah
- alic
- jenifer
```