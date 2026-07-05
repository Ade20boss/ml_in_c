# 🧠 Machine Learning in C — From First Principles

A growing collection of **bare-metal machine learning experiments** built entirely in C.

No frameworks.
No NumPy.
No PyTorch.
No autograd.

Just math, memory, and control — every matrix multiply, every gradient, and every byte of
memory hand-written.

---

## ⚙️ Philosophy

Most people *use* machine learning. This repository is about **understanding it**.

Every model here is implemented from scratch — no external libraries, no hidden
abstractions, no automatic differentiation.

> If you can't build it from the metal up, you don't really understand it.

---

## 🧭 The Progression

This repo is a deliberate climb: each stage adds one layer of understanding, ending in a
generalized, from-scratch neural network framework.

```
01_single_weight  →  one parameter, then two (linear regression)
02_double_weight  →  a neuron, then a hand-built 2-layer XOR network
03_kestrel        →  Kestrel: a generalized NN framework (any architecture)
```

---

## 📁 Stage Breakdown

### 01 — Single Weight Models
**Goal:** the absolute fundamentals of learning a parameter.

- **`zero_bias_linear_regressor/`** — model `y = wx`. Learns a single weight from data
  (`{1,2},{2,4},...`) via MSE cost, a numerical derivative, and gradient descent.
- **`complete_bias_linear_regressor/`** (`neuron.c`) — model `y = wx + b`. Adds a bias so
  the line no longer has to pass through the origin. Trains on a small "hours → cost"
  dataset and ends with an **interactive predictor** (enter hours, get a prediction).

Core loop introduced here: initialize randomly → predict → measure cost (MSE) → estimate
gradient (finite difference) → nudge parameters → repeat.

### 02 — Double Weight Models
**Goal:** move from regression → classification, then from a single neuron → a network.

- **`gates.c`** — a single neuron: `y = sigmoid(w₁x₁ + w₂x₂ + b)`. Learns the **OR gate**,
  adds the **sigmoid** activation to squash outputs into (0, 1), and a **classifier**
  (threshold at 0.5) to turn probability into a hard decision.
- **`full_gates.c`** — the leap: a **hand-built 2-layer network that learns XOR**. Two
  hidden neurons feed one output neuron (9 parameters total), all packed into a single
  contiguous array and trained with finite difference in a loop. XOR isn't linearly
  separable, so this is the problem that *requires* a hidden layer — the first real network
  in the repo. Prints the output neuron and both hidden neurons so you can see what each
  learned.

This is the transition point: **"predicting numbers" → "making decisions" → "a network."**

### 03 — Kestrel: the Framework
**Kestrel** — a from-scratch neural network framework in pure C. Named for the small, sharp
bird of prey: lean and precise. This is where the hardcoded XOR network of stage 02 becomes
a *generalized* system that works for any architecture.

**What's built:**
- **Custom arena allocator** — one upfront allocation, then O(1) bump-allocation, O(1)
  reset, and single-shot cleanup. Chosen deliberately: its strengths (fast bulk allocation,
  contiguous cache-friendly memory, trivial teardown) map onto exactly what training needs.
- **Matrix type with stride** — enables zero-copy row *views* (`row_matricize`) into a
  larger matrix, so training examples are sliced without copying. Core ops: dot product
  (matrix multiply), bias add, sigmoid activation, fill, randomize, copy, print.
- **Generalized network** — an array-based `Neural_network` (arrays of weight, bias, and
  activation matrices) built from an architecture array like `{2, 2, 1}` via `nn_allocate`.
  It derives every matrix shape from the architecture, so it handles **any depth and any
  layer sizes**. (`{3, 4, 2, 1}` works just as well as XOR's `{2, 2, 1}`.)
- **Generalized forward pass** — `Nn_forward_pass`: a single loop over layers
  (`dot → bias → activate`, chaining `inputs[i] → inputs[i+1]`), for any architecture.
- **`Nn_randomize`** — initialize all weights and biases across the whole network.

**Structure:** Kestrel is a single-header library — `kestrel.h` holds declarations up top
and the implementation guarded by `#ifdef KESTREL_CODE` (the stb/Tsoding pattern);
`kestrel.c` defines `KESTREL_CODE`, includes the header, and drives the demo.

---

## 🛠️ Core Loop

Every model in this repo, from one parameter to the framework, follows the same pipeline:

1. Initialize parameters randomly
2. Forward pass → compute predictions
3. Measure error (Mean Squared Error)
4. Estimate gradients (finite differences)
5. Update parameters (gradient descent)
6. Repeat until convergence

No magic. Just iteration — the same principle behind models with billions of parameters.

---

## 🧪 Build & Run

Each stage has its own `build.sh`.

```bash
# Stage 01 — bias linear regressor (interactive)
cd 01_single_weight/complete_bias_linear_regressor && chmod +x build.sh && ./build.sh

# Stage 02 — XOR via a hand-built 2-layer network
cd 02_double_weight && gcc -Wall -Wextra -o full_gates full_gates.c -lm && ./full_gates

# Stage 03 — Kestrel framework
cd 03_kestrel && chmod +x build.sh && ./build.sh
```

Kestrel builds with `-Wall -Wextra -Werror` — warnings are treated as errors.
No dependencies beyond a C compiler and libm.

---

## 🧩 Key Concepts You'll See

Mean Squared Error · numerical differentiation (finite differences) · gradient descent ·
learning-rate tuning · bias as a baseline offset · the sigmoid non-linearity · threshold
classification · why XOR needs a hidden layer · arena allocation · matrices with strided
zero-copy views · deriving layer shapes from an architecture array.

---

## 🧭 Roadmap

- [x] Single-weight linear regression (`y = wx`)
- [x] Weight + bias regression (`y = wx + b`) with interactive prediction
- [x] Sigmoid activation
- [x] Single-neuron classification (OR gate)
- [x] Hand-built multi-layer network (XOR, 2 hidden neurons)
- [x] Matrix abstraction with dynamic memory (arena allocator, strided views)
- [x] Generalized network allocation & forward pass (any architecture)
- [ ] Generalized training loop (cost / finite-difference / learn over the network)
- [ ] Backpropagation (manual, from scratch)
- [ ] Argmax classifier for multi-output networks
- [ ] Train on a real dataset (MNIST)
- [ ] Batching (process multiple examples per forward pass)

---

## 🔥 Why C?

Because C forces you to confront reality: memory is explicit, computation is visible,
nothing is hidden. You don't call `.fit()` — you **build it**.

---

## 📌 TL;DR

Modern models have trillions of parameters. Some of these have one. Others have two. But
every single one follows the same principles being built here — one honest, from-scratch
step at a time.
