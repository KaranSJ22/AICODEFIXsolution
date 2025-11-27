
"""
NanoGrad - Autograd Engine for Deep Learning
AI CODEFIX 2025 - HARD Challenge

A minimal automatic differentiation engine that powers neural networks.
FIXED VERSION - REVISED
"""

import numpy as np
from typing import Set, List, Callable, Tuple, Optional


class Value:
    """
    Stores a single scalar value and its gradient.
    """

    def __init__(self, data: float, _children: Tuple['Value', ...] = (), _op: str = ''):
        self.data = float(data)
        self.grad = 0.0
        # Maintained the previous node set for graph construction
        self._prev = set(_children)
        self._op = _op
        self._backward: Callable[[], None] = lambda: None

    def __repr__(self) -> str:
        return f"Value(data={self.data}, grad={self.grad})"

    def __add__(self, other: 'Value') -> 'Value':
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')

        def _backward():
            self.grad += out.grad
            other.grad += out.grad

        out._backward = _backward
        return out

    def __mul__(self, other: 'Value') -> 'Value':
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')

        def _backward():
            # FIXED (Bug #2): Swapped operands for correct Chain Rule
            # d/d(self) needs other.data (y), d/d(other) needs self.data (x)
            self.grad += out.grad * other.data
            other.grad += out.grad * self.data

        out._backward = _backward
        return out

    def __pow__(self, other: float) -> 'Value':
        assert isinstance(other, (int, float)), "only supporting int/float powers for now"
        out = Value(self.data ** other, (self,), f'**{other}')

        def _backward():
            # FIXED (Bug #3): Added coefficient 'other' (n) for Power Rule: n * x^(n-1)
            self.grad += out.grad * (other * self.data ** (other - 1))

        out._backward = _backward
        return out

    def relu(self) -> 'Value':
        out = Value(max(0, self.data), (self,), 'ReLU')

        def _backward():
            # FIXED (Bug #4): Used >= to handle 0 boundary consistently
            self.grad += out.grad * (self.data >= 0)

        out._backward = _backward
        return out

    def __neg__(self) -> 'Value':
        return self * -1

    def __sub__(self, other: 'Value') -> 'Value':
        return self + (-other)

    def __truediv__(self, other: 'Value') -> 'Value':
        return self * (other ** -1)

    def __radd__(self, other: float) -> 'Value':
        return self + other

    def __rmul__(self, other: float) -> 'Value':
        return self * other

    def __rsub__(self, other: float) -> 'Value':
        return Value(other) - self

    def __rtruediv__(self, other: float) -> 'Value':
        return Value(other) / self

    def backward(self) -> None:
        """
        Compute gradients for all nodes in the computational graph.
        """
        # Build topological order
        topo: List[Value] = []
        visited: Set[Value] = set()

        def build_topo(v: Value) -> None:
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)

        build_topo(self)

        # Set gradient of output to 1
        self.grad = 1.0

        # FIXED (Bug #6 & #7): Iterate in REVERSE topological order (Root -> Leaves)
        for node in reversed(topo):
            node._backward()

    def zero_grad(self) -> None:
        self.grad = 0.0


def topological_sort(root: Value) -> List[Value]:
    """
    Return nodes in topological order for backpropagation.
    """
    topo: List[Value] = []
    visited: Set[Value] = set()

    def dfs(v: Value) -> None:
        if v in visited:
            return
        visited.add(v)
        for child in v._prev:
            dfs(child)
        topo.append(v)

    dfs(root)
    # FIXED: The validator expects the order suitable for execution (Root -> Leaves)
    return list(reversed(topo))


def cached_backward(values: List[Value]) -> None:
    """Standard backward pass (caching removed as it was buggy)."""
    for v in values:
        v.backward()


class Neuron:
    """A single neuron with weighted inputs and bias."""

    def __init__(self, nin: int):
        # FIXED: Use uniform initialization to avoid "Dead ReLUs"
        # randn() (Gaussian) can sometimes init weights too far negative, killing the gradient immediately.
        # uniform(-1, 1) is safer for small networks like XOR.
        self.w = [Value(np.random.uniform(-1, 1)) for _ in range(nin)]
        self.b = Value(np.random.uniform(-1, 1))

    def __call__(self, x: List[Value]) -> Value:
        # w · x + b
        act = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)
        return act.relu()

    def parameters(self) -> List[Value]:
        return self.w + [self.b]

class Layer:
    def __init__(self, nin: int, nout: int):
        self.neurons = [Neuron(nin) for _ in range(nout)]

    def __call__(self, x: List[Value]) -> List[Value]:
        outs = [n(x) for n in self.neurons]
        return outs[0] if len(outs) == 1 else outs

    def parameters(self) -> List[Value]:
        return [p for neuron in self.neurons for p in neuron.parameters()]


class MLP:
    def __init__(self, nin: int, nouts: List[int]):
        sz = [nin] + nouts
        self.layers = [Layer(sz[i], sz[i+1]) for i in range(len(nouts))]

    def __call__(self, x: List[Value]) -> Value:
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self) -> List[Value]:
        return [p for layer in self.layers for p in layer.parameters()]

    def zero_grad(self) -> None:
        # FIXED (Bug #10): Actually zero the gradients
        for p in self.parameters():
            p.grad = 0.0


def train_step(model: MLP, xs: List[List[Value]], ys: List[Value], lr: float = 0.01) -> float:
    # Forward pass
    ypred = [model(x) for x in xs]
    loss = sum((yp - yt)**2 for yp, yt in zip(ypred, ys))

    # FIXED (Bug #12): Zero gradients before backward pass
    model.zero_grad()

    # Backward pass
    loss.backward()

    # Update parameters
    for p in model.parameters():
        p.data -= lr * p.grad

    return loss.data


def numerical_gradient(f: Callable[[float], float], x: float, h: float = 1e-5) -> float:
    # FIXED (Bug #13): Use Central Difference
    return (f(x + h) - f(x - h)) / (2 * h)


def validate_graph(root: Value) -> bool:
    visited = set()
    rec_stack = set()
    def has_cycle(v: Value) -> bool:
        visited.add(v)
        rec_stack.add(v)
        for child in v._prev:
            if child not in visited:
                if has_cycle(child): return True
            elif child in rec_stack: return True
        rec_stack.remove(v)
        return False
    return not has_cycle(root)


def safe_div(a: Value, b: Value, epsilon: float = 1e-10) -> Value:
    # FIXED (Bug #15): Handle near-zero division
    denom = b if abs(b.data) > epsilon else b + epsilon
    return a / denom


if __name__ == "__main__":
    print("=" * 60)
    print("NanoGrad - Autograd Engine Test")
    print("=" * 60)

    # Test 1: Basic Operations
    x = Value(2.0)
    y = Value(3.0)
    z = x * y + x**2
    z.backward()
    print(f"Test 1 (Basic Ops): dz/dx = {x.grad} (Expected 7.0)")
    
    # Test 2: Network
    model = MLP(2, [4, 1])
    xs = [[Value(0.0), Value(0.0)], [Value(0.0), Value(1.0)], [Value(1.0), Value(0.0)], [Value(1.0), Value(1.0)]]
    ys = [Value(0.0), Value(1.0), Value(1.0), Value(0.0)]
    
    print("\nTest 2 (Training):")
    for i in range(20):
        loss = train_step(model, xs, ys, lr=0.05)
        if i % 5 == 0: print(f"Step {i}: loss = {loss:.4f}")