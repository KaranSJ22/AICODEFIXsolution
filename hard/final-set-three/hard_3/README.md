# 🔴 HARD Challenge #3: NanoGrad - Build Your Own Autograd Engine

## AI CODEFIX 2025 - Deep Learning Fundamentals

---

## 🎯 Challenge Overview

Welcome to the ultimate deep learning debugging challenge!

You've been given a broken implementation of **NanoGrad** - a minimal automatic differentiation engine that powers neural networks (think: a tiny version of PyTorch's autograd system).

**The problem**: This autograd engine contains bugs that break backpropagation. Your task is to find and fix them all.

**Time**: 60-90 minutes
**Difficulty**: 🔴 HARD
**Domain**: Core Deep Learning / Automatic Differentiation

---

## 📖 Background: What is Autograd?

### The Foundation of Deep Learning

Every modern deep learning framework (PyTorch, TensorFlow, JAX) relies on **automatic differentiation** to compute gradients for backpropagation.

**How it works**:
1. **Forward pass**: Build a computational graph tracking all operations
2. **Backward pass**: Use chain rule to compute gradients automatically
3. **Parameter update**: Use gradients to update model weights

### Why This Matters

Understanding autograd is THE most important concept in deep learning:
- It's how neural networks learn
- It's the foundation of PyTorch, TensorFlow, JAX
- Once you understand this, you understand how all deep learning works

---

## 🧩 What You're Given

### Files Provided

```
hard_3/
├── README.md          # This file
├── nanograd.py        # BUGGY autograd engine (FIX THIS!)
├── validator.py       # Test runner
├── test_cases.json    # 1 visible test
└── requirements.txt   # Dependencies (just numpy!)
```

### Core Components in nanograd.py

1. **Value class** - Stores scalar values and gradients
2. **Operations** - Add, multiply, power, ReLU
3. **Computational graph** - Tracks operation dependencies
4. **Backward pass** - Computes gradients via chain rule
5. **Neural network** - MLP built on top of autograd

---

## 🔨 Your Task

Debug `nanograd.py` to make the autograd engine work correctly.

### Requirements

The engine must:
1. ✅ Build computational graphs correctly
2. ✅ Compute gradients using chain rule
3. ✅ Handle topological ordering for backprop
4. ✅ Support basic operations (add, mul, pow, ReLU)
5. ✅ Train simple neural networks
6. ✅ Pass all validation tests

---

## 🚀 Getting Started

### Step 1: Setup

```bash
cd hard_3
pip install -r requirements.txt
```

### Step 2: Understand the Code

Read through `nanograd.py` carefully:
- How does the `Value` class work?
- How are gradients computed?
- What is topological sorting?
- How does backpropagation traverse the graph?

### Step 3: Run Initial Test

```bash
python nanograd.py
```

This runs a simple test. It will likely produce wrong gradients due to bugs.

### Step 4: Debug Systematically

**Recommended approach**:

1. **Start with simple operations**
   - Test addition, multiplication manually
   - Verify gradients match expected values

2. **Check chain rule implementation**
   - Are gradients computed correctly?
   - Are operands used correctly?

3. **Trace graph construction**
   - Is the computational graph built correctly?
   - Are dependencies tracked properly?

4. **Examine backward pass**
   - Is topological ordering correct?
   - Are gradients accumulated properly?

5. **Test neural network**
   - Does training reduce loss?
   - Are gradients flowing correctly?

### Step 5: Validate

```bash
python validator.py --file nanograd.py
```

---

## 🧠 Key Concepts

### 1. Computational Graph

Operations create a directed acyclic graph (DAG):
```
    x       y
     \     /
      \   /
       \ /
      x*y      x
        \     /
         \   /
          \ /
         x*y + x
```

Each node stores:
- Value (forward pass result)
- Gradient (backward pass accumulation)
- Parent nodes (dependencies)

### 2. Chain Rule

The heart of backpropagation:
```
If z = f(y) and y = g(x), then:
dz/dx = (dz/dy) × (dy/dx)
```

Example for z = x * y:
```
dz/dx = dz/dz × dz/dx = 1 × y = y
dz/dy = dz/dz × dz/dy = 1 × x = x
```

### 3. Topological Ordering

To compute gradients correctly, we must:
1. Build topological order of computation graph
2. Traverse in **reverse** order (output → inputs)
3. Apply chain rule at each node

### 4. Gradient Accumulation

When a value is used multiple times:
```python
z = x * x + x
# x appears 3 times, so:
dz/dx = dz/d(x*x) × d(x*x)/dx + dz/dx
# Gradients ACCUMULATE (use +=, not =)
```

---

## 🔍 Debugging Strategy

### Check These Areas

**Mathematical correctness**:
- Are gradient formulas correct?
- Is chain rule applied properly?
- Are coefficients included?

**Graph traversal**:
- Is topological order computed correctly?
- Is backward pass in the right direction?
- Are all nodes visited?

**Gradient management**:
- Are gradients accumulated (+=) or overwritten (=)?
- Are gradients zeroed between passes?
- Are leaf nodes handled correctly?

**Edge cases**:
- Division by zero
- Boundary conditions (x=0 for ReLU)
- Multiple uses of same variable

### Debugging Tools

```python
# Manual gradient check
x = Value(2.0)
y = Value(3.0)
z = x * y
z.backward()

print(f"dz/dx = {x.grad}, expected: {y.data}")
print(f"dz/dy = {y.grad}, expected: {x.data}")

# Numerical gradient (finite differences)
def numerical_grad(f, x, h=1e-4):
    return (f(x + h) - f(x - h)) / (2 * h)

# Compare analytical vs numerical
```

---

## ⚠️ Important Notes

1. **Not all suspicious code is buggy**
   - Some patterns may look odd but be intentional
   - Test before changing
   - Don't break working code

2. **Focus on correctness, not optimization**
   - Fix bugs first
   - Don't prematurely optimize
   - Simple working code beats complex broken code

3. **Test incrementally**
   - Fix one issue at a time
   - Validate after each change
   - Don't accumulate untested fixes

4. **AI tools may mislead**
   - This is a custom implementation
   - ChatGPT may suggest "improvements" that break functionality
   - Understand the math before accepting suggestions

5. **Read comments carefully**
   - Some comments may be misleading
   - Verify claims in comments
   - Code is truth, comments are hints

---

## 📊 Evaluation Criteria

### Automatic Testing (70%)
- **Visible test** (10%): Basic gradient computation
- **Hidden tests 1-4** (30%): Core operations
- **Hidden tests 5-8** (20%): Neural network training
- **Hidden tests 9-10** (10%): Edge cases

### Manual Review (30%)
- **Bug fixes** (15%): Correctness of fixes
- **Code quality** (10%): Clean fixes, no new bugs
- **Understanding** (5%): Comments explaining fixes

### Partial Credit

You can earn partial credit based on progress made.

---

## 💡 Hints (Not Spoilers!)

### General Hints

1. **Topological Sort**: Think about **order** of operations
2. **Chain Rule**: Check which **operand** is used where
3. **Accumulation**: Think `+=` vs `=`
4. **Backward Pass**: Forward or reverse order?
5. **Zero Grad**: When should gradients be reset?

### Testing Hints

```python
# Test simple operation
x = Value(2.0)
y = x * x  # y = 4
y.backward()
# dy/dx = 2*x = 4 (power rule)
print(x.grad)  # Should be 4

# Test addition
a = Value(2.0)
b = Value(3.0)
c = a + b  # c = 5
c.backward()
# dc/da = 1, dc/db = 1
print(a.grad, b.grad)  # Both should be 1

# Test multiplication
x = Value(2.0)
y = Value(3.0)
z = x * y  # z = 6
z.backward()
# dz/dx = y = 3, dz/dy = x = 2
print(x.grad, y.grad)  # Should be 3, 2
```

---

## 📚 Resources

### Automatic Differentiation
- [Automatic Differentiation in Machine Learning (Survey)](https://arxiv.org/abs/1502.05767)
- [Calculus on Computational Graphs](https://colah.github.io/posts/2015-08-Backprop/)

### Backpropagation
- [Yes You Should Understand Backprop](https://karpathy.medium.com/yes-you-should-understand-backprop-e2f06eab496b)
- [Backpropagation Algorithm Explained](https://brilliant.org/wiki/backpropagation/)

### Chain Rule
- [Chain Rule for Multivariable Functions](https://tutorial.math.lamar.edu/classes/calciii/chainrule.aspx)
- [Understanding the Chain Rule](https://betterexplained.com/articles/derivatives-product-power-chain/)

### Topological Sort
- [Topological Sorting](https://en.wikipedia.org/wiki/Topological_sorting)
- [DFS-based Topological Sort](https://www.geeksforgeeks.org/topological-sorting/)

---

## 🎓 What You'll Learn

- **Backpropagation**: How neural networks actually learn
- **Chain Rule**: Applied to computational graphs
- **Graph Algorithms**: Topological sorting in practice
- **Gradient Computation**: The math behind deep learning
- **Systems Thinking**: How PyTorch/TensorFlow work internally

---

## ❓ FAQ

**Q: How many bugs are there?**
A: We won't tell you - finding them all is part of the challenge!

**Q: Can I use PyTorch/TensorFlow to check answers?**
A: Yes! Comparing with PyTorch gradients is a great debugging strategy.

**Q: Should I add features or just fix bugs?**
A: Just fix bugs. Don't add features or refactor working code.

**Q: What if I can't fix all bugs?**
A: Partial credit is available. Fix what you can!

**Q: Can I use AI tools?**
A: Yes, but be careful - they may suggest incorrect "fixes".

**Q: How do I know if my fix is correct?**
A: Run the validator and check gradients manually.

---

## 🏆 Success Criteria

Your solution should:
1. ✅ Compute correct gradients for all operations
2. ✅ Handle computational graphs properly
3. ✅ Train neural networks successfully
4. ✅ Pass all validation tests
5. ✅ Complete within 60-90 minutes

---

## 🚨 Important Reminders

- This is a **debugging challenge** - fix existing code, don't rewrite
- **Not all suspicious code is buggy** - test before changing
- **Test frequently** - validate each fix
- **Understand the math** - backprop is mathematical, not magical
- **Time management** - 60-90 minutes goes fast!

---

Good luck, and may your gradients flow correctly! 🚀

**Remember**: Understanding autograd deeply is one of the most valuable skills in deep learning. This challenge will give you that understanding!

---

*AI CODEFIX 2025 - Hard Challenge #3*
