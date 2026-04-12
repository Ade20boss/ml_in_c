# 🧠 Machine Learning in C — Study Notes  
### *Episode 1 · Linear regression → multi-input neuron → sigmoid*

---

## 📚 Contents

- 01 Core idea  
- 02 Training loop step-by-step  
- 03 Key concepts  
- 04 1 input → 2 inputs  
- 05 Why that formula exists  
- 06 What sigmoid really does  
- 07 What "toggling" actually means  
- 08 Student thinking analysis  
- 09 Teaching strategy analysis  
- 10 Final mental model  

---

## 01 — Core idea

You are building a machine learning engine from raw C — no NumPy, no PyTorch, no abstraction. The goal is to understand what those libraries are actually doing under the hood by implementing it yourself.

At its simplest, the model is a formula with adjustable numbers (parameters). You feed it input data, it produces an output, and you measure how wrong it is. Then you adjust the parameters to make it less wrong. Repeat until it stops being wrong. That is machine learning.

### The full arc of Episode 1

Single weight, no bias (y = wx) → linear regression with bias (y = wx + b) → two-input neuron (y = w1x1 + w2x2 + b) → sigmoid activation → logic gate classification (OR, AND, NAND)

---

## 02 — Training loop — step by step

### 1  
Initialize parameters randomly. w and b start as random floats (e.g. rand_float() * 10.0f). The model starts knowing nothing — it is just guessing.

### 2  
Run the cost function.  
For every data point, compute the model's prediction, subtract the actual value to get the error, square it (so negatives don't cancel positives and large errors are amplified), then average across all data points. This is MSE — Mean Squared Error. Lower = model is more correct.

### 3  
Compute the numerical derivative for each parameter.  
Nudge w by a tiny amount (changer = 1e-3), recompute the cost, divide by the nudge size. That ratio is the slope — how steeply the error changes when you move w. Do the same for b, holding w fixed.

### 4  
Update rule: subtract the slope.  
w -= rate * dw. The sign of dw tells you which direction is "uphill" on the error surface — so you move in the opposite direction. The magnitude of dw tells you how far you are from the minimum. rate controls the step size.

### 5  
Repeat for N iterations.  
Each loop, the model gets slightly less wrong. After enough iterations, the parameters converge at values where the cost is near zero.

---

## 03 — Key concepts explained

### Weight (w)  
**The importance multiplier**  
Controls how strongly an input affects the output. w = 0 means the input is ignored. A large w means the input dominates. Gradient descent adjusts this automatically.

### Bias (b)  
**The baseline / resting state**  
Without a bias, the model is mathematically locked at 0 when the input is 0. The bias shifts the whole output up or down, decoupled from the input. Like the ₦1,000 Uber pickup fee — it exists regardless of trip length.

### Cost function (MSE)  
**The error meter**  
Measures total model wrongness. Squaring errors does two things: prevents positive and negative errors from cancelling out, and amplifies large mistakes so the model "panics" and corrects them faster.

### Numerical derivative  
**Finite differences**  
Instead of analytical calculus, you estimate the slope by nudging a parameter and observing how the cost changes. dw = (cost(w + ε) - cost(w)) / ε. This is the "feel the slope" approach — no symbolic differentiation needed.

### Learning rate  
**Step size down the error slope**  
Too large: you overshoot the minimum and bounce around or diverge. Too small: training takes forever. Typical starting point: 1e-3. This is a hyperparameter you set, not one the model learns.

### Gradient descent  
**Walking downhill on error**  
The error surface is a bowl in N-dimensional space (one dimension per parameter). Gradient descent moves the parameters in the direction that makes the error go down. The derivative is the compass.

---

## 04 — From 1 input → 2 inputs

The jump from linear regression to a logic gate neuron is smaller than it looks. The architecture is identical — you are just adding a second input wire and its corresponding weight.

**y = wx + b → y = (w1 · x1) + (w2 · x2) + b**


**What changes:** one extra input, one extra weight, one extra derivative

### What changes in your C code:

1. Dataset shape changes from {x, y} to {x1, x2, y}. For an OR gate this is just the truth table: four rows, each with two inputs and one expected output.

2. Parameters go from w, b to w1, w2, b. Your training loop now computes three derivatives: dw1, dw2, db.

3. Sigmoid wraps the output before the cost is computed. Without it, the unbounded raw score makes no sense as a binary 0/1 answer. Sigmoid squishes it into a clean probability.

### Key insight

The training process is doing exactly the same thing as linear regression — it is just looking for three magic numbers (w1, w2, b) that make the arithmetic satisfy the truth table simultaneously for all four inputs.

---

## 05 — Why that formula exists

The formula (w1 · x1) + (w2 · x2) + b is not arbitrary. Each operation does a specific job in the signal processing chain.

### × Multiplication = amplification  
w · x scales the input signal by its learned importance. If x = 0, the signal is completely muted regardless of w. If x = 1, the full weight passes through.

### + Addition = aggregation  
You have two amplified signals. They collapse into one master score. Signals can reinforce or cancel each other.

### b Bias = the default state  
If both inputs are 0, the formula reduces to b. For an OR gate, b becomes negative so the default output is 0.

### Why not use exponents or other math?

Stability. Multiplication and addition have simple derivatives. More complex math causes unstable gradients and breaks training.

---

## 06 — What sigmoid really does

Two jobs. Both are mandatory.

### Job 1 — formatting  
**Squish the raw score into [0, 1]**  
Large positive → ≈1.0  
Large negative → ≈0.0  
Zero → 0.5

### Job 2 — training  
**Provide a slope for gradient descent**  
Without sigmoid, the model has no usable derivative and cannot learn.


**sigmoid(x) = 1 / (1 + e^-x)**


In C: `1.0f / (1.0f + expf(-x))`

---

## 07 — What "toggling" actually means

"Toggling" = repeatedly adjusting parameters using derivatives.

The model starts with random values: w1 = 0.2, w2 = -3.4, b = 1.1


Then asks:

- If I nudge w1, does error go up or down?
- Same for w2
- Same for b

It uses those answers to update all parameters until they converge to values that satisfy all conditions.

---

## 08 — Student thinking analysis

### What you understood well

- Strong Uber bias analogy  
- Clear understanding of derivative (sign + magnitude)  
- Asked deep system-level questions  

### Where you struggled

- Confused input (x) with importance (w)  
- Questioned why ML is needed for OR logic  

### Breakthroughs

- Connected regression → classification  
- Rejected visual explanations → improved clarity  

---

## 09 — Teaching strategy analysis

### Phase 1 — Visual  
Failed (not your thinking style)

### Phase 2 — Logical / threshold  
Worked well

### Phase 3 — Systems / hardware  
Worked best (aligned with C mindset)

---

## 10 — Final mental model

A machine learning model is:

- A formula with adjustable numbers  
- A cost function that measures error  
- A process (gradient descent) that reduces error  

Regression:
- Output = raw number

Classification:
- Output → sigmoid → probability

Same engine.  
Sigmoid is the only structural change.

---