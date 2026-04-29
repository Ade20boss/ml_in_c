# 🧠 How We Model Binary Output — And Why XOR Breaks It

> *Machine Learning in C · Episode 1 Notes · Daniel*
> **Tags:** #machinelearning #C #neuralnetworks #linearalgebra #fundamentals

---

## The Core Idea

A machine learning model is just a formula with adjustable numbers.

You feed it inputs. It does arithmetic. It produces a number. You measure how wrong that number is. You adjust. You repeat — thousands of times — until the formula reliably produces the right answer for every input.

That's it. Everything else is detail.

---

## Part 1 — The Linear Equation

The formula at the heart of a single neuron is:

```
y = (w1 × x1) + (w2 × x2) + b
```

Each piece does a specific job:

| Symbol | Name | Job |
|--------|------|-----|
| `x1`, `x2` | Inputs | Raw data coming in — the state of the world |
| `w1`, `w2` | Weights | How much each input matters |
| `b` | Bias | The baseline output when both inputs are zero |
| `y` | Output | The raw score — unbounded, can be anything |

### Why multiplication?

`w × x` scales the input signal by its learned importance. If `x = 0`, the entire term collapses to zero — the input is completely muted, no matter how large `w` is. If `x = 1`, the full weight passes through. Multiplication is the only basic operation that correctly gates a signal by an importance factor.

### Why addition?

You have two amplified signals. They need to collapse into one number. Addition lets signals reinforce each other (both high → very high score) or partially cancel (one positive, one negative → somewhere in the middle). It's cheap, stable, and its derivative is simple — which matters for training.

### Why bias?

Without a bias, if both inputs are `0`, the formula gives `0`. Always. No matter what the weights are. The bias decouples the output from the inputs — it is the resting state of the neuron.

> **Uber analogy:** If `x` is hours driven and `w` is the per-hour rate, then `b` is the base pickup fee — it exists regardless of trip length. A model trained to predict Uber prices without a bias will always output `₦0` for a zero-hour trip, even though the correct answer is `₦1,000`.

---

## Part 2 — The Problem With Binary Output

The formula `(w1 × x1) + (w2 × x2) + b` is a **linear equation**. It produces a continuous, unbounded number — anything from `-∞` to `+∞`.

But a logic gate only has two possible outputs: `0` or `1`.

How does pure arithmetic produce a binary decision?

**Short answer: it doesn't. Not alone.**

What the linear equation actually does is produce a **raw score**. The binary decision comes from asking one question about that score:

```
Is the score positive or negative?
```

- Positive → output ≈ 1  
- Negative → output ≈ 0  

The **sigmoid function** is what converts the raw score into a clean probability between 0 and 1:

```
sigmoid(x) = 1 / (1 + e^(-x))
```

In C:
```c
float sigmoid(float x) {
    return 1.f / (1.f + expf(-x));
}
```

### What sigmoid actually does

It has two jobs, both mandatory:

**Job 1 — Formatting.** Squishes any raw number into the range `(0.0, 1.0)`. Large positive → ≈1.0. Large negative → ≈0.0. Exactly zero → 0.5 (maximum uncertainty).

**Job 2 — Training.** A hard cutoff (`if score > 0 return 1`) has derivative = 0 everywhere — the model goes blind and can't find which direction to adjust the weights. Sigmoid is a smooth curve with a calculable slope at every point, so gradient descent always has a direction to walk.

> **The load-bearing piece:** The linear equation alone cannot model binary output. Sigmoid is not optional — it is the translator that converts arithmetic into classification.

---

## Part 3 — Why It Works for OR, AND, NAND

For OR, AND, and NAND, there exist values of `w1`, `w2`, `b` that make the arithmetic produce a negative score for "should be 0" inputs and a positive score for "should be 1" inputs.

**OR gate — proof by example:**

After training, the model converges to approximately `w1 = 10, w2 = 10, b = -5`. Verify manually:

```
Input (0,0): (10×0) + (10×0) + (−5) = −5   → negative → sigmoid ≈ 0.007  ✓
Input (1,0): (10×1) + (10×0) + (−5) = +5   → positive → sigmoid ≈ 0.993  ✓
Input (0,1): (10×0) + (10×1) + (−5) = +5   → positive → sigmoid ≈ 0.993  ✓
Input (1,1): (10×1) + (10×1) + (−5) = +15  → positive → sigmoid ≈ 0.999  ✓
```

One set of three numbers satisfies all four cases simultaneously. The training process — gradient descent — finds those numbers by adjusting `w1`, `w2`, `b` thousands of times, each time asking: "does nudging this parameter make the total error go up or down?"

---

## Part 4 — XOR: The Arithmetic Impossibility

XOR truth table:

```
(0, 0) → 0
(0, 1) → 1
(1, 0) → 1
(1, 1) → 0
```

For the model to learn XOR, the arithmetic must satisfy all four conditions simultaneously:

```
b              < 0   ← (0,0) must output 0
w2 + b         > 0   ← (0,1) must output 1
w1 + b         > 0   ← (1,0) must output 1
w1 + w2 + b    < 0   ← (1,1) must output 0
```

### The contradiction — no algebra needed

Look at conditions 2 and 3:

```
w2 + b > 0  →  w2 alone is strong enough to beat the bias
w1 + b > 0  →  w1 alone is strong enough to beat the bias
```

Both weights individually overcome the bias. So when both fire together, they overcome the bias even more. The combined result `w1 + w2 + b` is more positive than either weight alone.

But condition 4 requires `w1 + w2 + b` to be **negative**.

**You cannot have both simultaneously.** If each weight beats the bias alone, they beat it harder together. XOR requires them not to — which is a direct contradiction.

### See it fail with actual numbers

Try `w1 = 1, w2 = 1, b = -0.5`:
```
b           = −0.5          ✓
w2 + b      = 0.5           ✓
w1 + b      = 0.5           ✓
w1 + w2 + b = 1.5           ✗  (needs to be negative)
```

Make `b` more negative to drag `(1,1)` down. Try `b = -2`:
```
b           = −2            ✓
w2 + b      = −1            ✗  (needs to be positive)
w1 + b      = −1            ✗  (needs to be positive)
w1 + w2 + b = 0             ✗
```

Making `b` more negative breaks the middle two cases. Making the weights larger makes `(1,1)` even more positive — the opposite of what you need. Every fix creates a new break. You are chasing your own tail.

> **This is not a training problem.** It is not a matter of more iterations, a different learning rate, or a better epsilon. The solution provably does not exist for a single neuron. You can prove it with arithmetic in five lines.

---

## Part 5 — The Fix: Hidden Layers

Since one formula cannot satisfy all four XOR conditions simultaneously, the solution is two formulas — two neurons — each learning a simpler sub-problem that is actually solvable.

The architecture:

```
         ┌─────────────┐
x1 ──────┤  Neuron A   ├──────┐
x2 ──┬───┤  (e.g. AND) │      ├──── Neuron C (learns XOR from A and B)
     │   └─────────────┘      │
     └───┤  Neuron B   ├──────┘
x1 ──────┤  (e.g. NAND)│
         └─────────────┘
```

- Neuron A and B each solve a sub-problem that a single neuron can handle.  
- Neuron C receives their outputs and learns XOR from those — which is now solvable because the inputs it sees are no longer the raw truth table.

This is a **hidden layer**. The composition of multiple sigmoid-wrapped linear equations becomes a genuinely nonlinear function capable of representing XOR.

---

## Summary — The Mental Model

```
Linear equation   →   raw score (unbounded arithmetic)
      ↓
   Sigmoid        →   probability in [0, 1]
      ↓
   Threshold      →   binary decision (≈0 or ≈1)
```

The linear equation is the engine. Sigmoid is the translator. The binary decision is the output.

For OR, AND, NAND — a single formula can find weights that satisfy all truth table rows. For XOR — no such weights exist. The requirements contradict each other arithmetically. The fix is adding a second layer, which Episode 2 covers.

---

## Key Takeaways

- The linear equation does not model binary output. Sigmoid does.
- Sigmoid's smooth curve is mandatory for training — without it, gradient descent has no slope to follow.
- OR, AND, NAND are solvable by a single neuron. XOR is not — provably, arithmetically.
- The XOR failure is not a training issue. It is a fundamental limitation of a single linear formula.
- Hidden layers are the fix — multiple neurons, each solving a solvable sub-problem, whose outputs a final neuron combines.

---

*Next: Episode 2 — Generalizing to an arbitrary number of layers. The Matrix abstraction. The 4-bit Adder.*
