# 🧠 Machine Learning in C — From First Principles

A growing collection of **bare-metal machine learning experiments** built entirely in C.

No frameworks.
No NumPy.
No PyTorch.

Just math, memory, and control.

---

## ⚙️ Philosophy

Most people *use* machine learning.

This repository is about **understanding it**.

Every model here is implemented from scratch:

* No external libraries
* No hidden abstractions
* No automatic differentiation

The goal is simple:

> If you can't build it from the metal up, you don't really understand it.

---

## 🚀 What This Repo Explores

* Linear regression (with and without bias)
* Numerical gradient estimation (finite differences)
* Gradient descent optimization
* Multi-input models
* Logical gates (OR, AND, NAND)
* Foundations for building neural networks

This evolves step-by-step toward a **full C-based ML framework**.

---

## 📁 Project Structure

```
.
├── 01_single_weight
│   ├── complete_bias_linear_regressor
│   │   ├── build.sh
│   │   └── neuron.c
│   ├── MACHINE LEARNING IN C.md
│   └── zero_bias_linear_regressor
│       ├── build.sh
│       └── zero-bias_linear_regressor.c
│
└── 02_double_weight
    ├── or_gate.c
    └── or_gate_model
```

---

## 🧩 Breakdown

### 01 — Single Weight Models

**Goal:** Understand the absolute fundamentals.

#### 🔹 Zero Bias Linear Regressor

* Model: `y = wx`
* Learns a single parameter
* Demonstrates:

  * Cost functions (MSE)
  * Numerical derivatives
  * Basic gradient descent

#### 🔹 Bias Linear Regressor

* Model: `y = wx + b`
* Adds flexibility via bias
* Key insight:

  * Models can now fit data not passing through origin

---

### 02 — Double Weight Models

**Goal:** Move from regression → classification.

#### 🔹 OR Gate Model

* Model: `y = w₁x₁ + w₂x₂ + b`
* Introduces:

  * Multiple inputs
  * Multiple weights
  * Binary classification logic

This is the transition point from:

> “predicting numbers” → “making decisions”

---

## 🛠️ How It Works (Core Loop)

Every model in this repo follows the same fundamental pipeline:

1. **Initialize parameters randomly**
2. **Compute predictions**
3. **Measure error (cost function)**
4. **Estimate gradients (finite differences)**
5. **Update parameters (gradient descent)**
6. **Repeat until convergence**

No magic. Just iteration.

---

## 🧪 Build & Run

Each module contains its own build script.

Example:

```bash
cd 01_single_weight/complete_bias_linear_regressor
chmod +x build.sh
./build.sh
```

---

## 🧠 Key Concepts You’ll See Here

* Mean Squared Error (MSE)
* Numerical differentiation
* Learning rate tuning
* Parameter convergence
* Bias as baseline offset
* Signal weighting via multiplication
* Threshold-based decision making

---

## 🔥 Why C?

Because C forces you to confront reality:

* Memory is explicit
* Computation is visible
* Nothing is hidden

You don’t “call `.fit()`”.

You **build it**.

---

## 🧭 Roadmap

This is just the beginning.

Planned progression:

* [ ] AND / NAND gates
* [ ] Sigmoid activation implementation
* [ ] Multi-layer perceptron (MLP)
* [ ] Matrix abstraction (dynamic memory)
* [ ] Generalized training loops
* [ ] Backpropagation (manual)
* [ ] Mini ML framework in pure C

---

## 🧨 Final Note

Modern models have billions (or trillions) of parameters.

Some of these have one.

Others have two.

But every single one of them follows the same principles you're building here.

---

## 📌 TL;DR

This repo is not about scale.

It’s about **understanding the machinery behind intelligence**.

