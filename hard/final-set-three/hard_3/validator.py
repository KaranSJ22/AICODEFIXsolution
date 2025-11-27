"""
Validator for NanoGrad - Autograd Engine
AI CODEFIX 2025 - HARD Challenge

Tests autograd engine correctness by checking gradient computations.

Usage:
    python validator.py --file nanograd.py
    python validator.py --file nanograd.py --verbose
"""

import argparse
import sys
import importlib.util
import numpy as np
from typing import Dict, List, Tuple, Any, Optional


class GradientValidator:
    """Validator for autograd engine gradient computations."""

    def __init__(self, module_path: str, test_file: str = "test_cases.json", verbose: bool = False):
        """
        Initialize validator.

        Args:
            module_path: Path to the Python file to test
            test_file: Path to test cases (not used in this challenge - tests are hardcoded)
            verbose: Whether to print detailed output
        """
        self.module_path = module_path
        self.verbose = verbose
        self.module = None
        self.tolerance = 1e-4  # Tolerance for gradient comparisons

    def load_module(self) -> bool:
        """
        Dynamically load the Python module.

        Returns:
            True if successful, False otherwise
        """
        try:
            spec = importlib.util.spec_from_file_location("nanograd_module", self.module_path)
            self.module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(self.module)

            # Verify required class exists
            if not hasattr(self.module, 'Value'):
                print("✗ Error: Module must contain 'Value' class")
                return False

            if self.verbose:
                print(f"✓ Successfully loaded module from {self.module_path}")

            return True
        except Exception as e:
            print(f"✗ Error loading module: {e}")
            import traceback
            if self.verbose:
                traceback.print_exc()
            return False

    def numerical_gradient(self, func, x: float, h: float = 1e-5) -> float:
        """
        Compute numerical gradient using central differences.

        Args:
            func: Function to differentiate
            x: Point at which to compute gradient
            h: Step size

        Returns:
            Numerical gradient
        """
        return (func(x + h) - func(x - h)) / (2 * h)

    def test_basic_operations(self) -> Tuple[bool, str]:
        """Test basic arithmetic operations and their gradients."""
        try:
            Value = self.module.Value

            # Test 1: Addition
            a = Value(2.0)
            b = Value(3.0)
            c = a + b
            c.backward()

            if abs(a.grad - 1.0) > self.tolerance or abs(b.grad - 1.0) > self.tolerance:
                return False, f"Addition gradients wrong: da={a.grad}, db={b.grad} (expected: 1.0, 1.0)"

            # Test 2: Multiplication
            x = Value(2.0)
            y = Value(3.0)
            z = x * y
            z.backward()

            if abs(x.grad - 3.0) > self.tolerance or abs(y.grad - 2.0) > self.tolerance:
                return False, f"Multiplication gradients wrong: dx={x.grad}, dy={y.grad} (expected: 3.0, 2.0)"

            # Test 3: Power
            x = Value(3.0)
            y = x ** 2
            y.backward()

            expected_grad = 2 * 3.0  # 2 * x
            if abs(x.grad - expected_grad) > self.tolerance:
                return False, f"Power gradient wrong: dx={x.grad} (expected: {expected_grad})"

            return True, "✓ Basic operations: PASSED"

        except Exception as e:
            return False, f"✗ Basic operations error: {e}"

    def test_chain_rule(self) -> Tuple[bool, str]:
        """Test chain rule in compound expressions."""
        try:
            Value = self.module.Value

            # Test: f(x, y) = x*y + x^2
            x = Value(2.0)
            y = Value(3.0)
            z = x * y + x ** 2

            z.backward()

            # df/dx = y + 2*x = 3 + 4 = 7
            # df/dy = x = 2
            expected_dx = 7.0
            expected_dy = 2.0

            if abs(x.grad - expected_dx) > self.tolerance:
                return False, f"Chain rule dx wrong: {x.grad} (expected: {expected_dx})"

            if abs(y.grad - expected_dy) > self.tolerance:
                return False, f"Chain rule dy wrong: {y.grad} (expected: {expected_dy})"

            return True, "✓ Chain rule: PASSED"

        except Exception as e:
            return False, f"✗ Chain rule error: {e}"

    def test_gradient_accumulation(self) -> Tuple[bool, str]:
        """Test that gradients accumulate when value used multiple times."""
        try:
            Value = self.module.Value

            # Test: f(x) = x*x + x
            x = Value(2.0)
            y = x * x + x
            y.backward()

            # df/dx = 2*x + 1 = 2*2 + 1 = 5
            expected = 5.0

            if abs(x.grad - expected) > self.tolerance:
                return False, f"Gradient accumulation wrong: {x.grad} (expected: {expected})"

            return True, "✓ Gradient accumulation: PASSED"

        except Exception as e:
            return False, f"✗ Gradient accumulation error: {e}"

    def test_relu(self) -> Tuple[bool, str]:
        """Test ReLU activation gradient."""
        try:
            Value = self.module.Value

            # Test 1: Positive input
            x = Value(2.0)
            y = x.relu()
            y.backward()

            if abs(y.data - 2.0) > self.tolerance:
                return False, f"ReLU forward wrong: {y.data} (expected: 2.0)"

            if abs(x.grad - 1.0) > self.tolerance:
                return False, f"ReLU gradient (x>0) wrong: {x.grad} (expected: 1.0)"

            # Test 2: Negative input
            x2 = Value(-1.0)
            y2 = x2.relu()
            y2.backward()

            if abs(y2.data - 0.0) > self.tolerance:
                return False, f"ReLU forward (negative) wrong: {y2.data} (expected: 0.0)"

            if abs(x2.grad - 0.0) > self.tolerance:
                return False, f"ReLU gradient (x<0) wrong: {x2.grad} (expected: 0.0)"

            return True, "✓ ReLU activation: PASSED"

        except Exception as e:
            return False, f"✗ ReLU error: {e}"

    def test_complex_expression(self) -> Tuple[bool, str]:
        """Test complex nested expression."""
        try:
            Value = self.module.Value

            # f(x, y, z) = (x + y) * z + z**2
            x = Value(1.0)
            y = Value(2.0)
            z = Value(3.0)

            f = (x + y) * z + z ** 2
            f.backward()

            # df/dx = z = 3
            # df/dy = z = 3
            # df/dz = (x + y) + 2*z = 3 + 6 = 9

            if abs(x.grad - 3.0) > self.tolerance:
                return False, f"Complex expression dx wrong: {x.grad} (expected: 3.0)"

            if abs(y.grad - 3.0) > self.tolerance:
                return False, f"Complex expression dy wrong: {y.grad} (expected: 3.0)"

            if abs(z.grad - 9.0) > self.tolerance:
                return False, f"Complex expression dz wrong: {z.grad} (expected: 9.0)"

            return True, "✓ Complex expression: PASSED"

        except Exception as e:
            return False, f"✗ Complex expression error: {e}"

    def test_zero_grad(self) -> Tuple[bool, str]:
        """Test that zero_grad correctly resets gradients."""
        try:
            Value = self.module.Value

            x = Value(2.0)
            y = x * x
            y.backward()

            # First backward
            grad1 = x.grad

            # Zero and backward again
            x.zero_grad()
            y = x * x
            y.backward()
            grad2 = x.grad

            # Should be same (not accumulated)
            if abs(grad1 - grad2) > self.tolerance:
                return False, f"zero_grad not working: grad after zero_grad should reset"

            # Check it's actually the right value
            expected = 4.0  # 2*x = 2*2
            if abs(grad2 - expected) > self.tolerance:
                return False, f"Gradient after zero_grad wrong: {grad2} (expected: {expected})"

            return True, "✓ zero_grad: PASSED"

        except Exception as e:
            return False, f"✗ zero_grad error: {e}"

    def test_neural_network(self) -> Tuple[bool, str]:
        """Test that neural network can train."""
        try:
            Value = self.module.Value
            MLP = self.module.MLP

            # Set seed for reproducibility
            np.random.seed(42)

            # Create small network
            model = MLP(2, [4, 1])

            # Simple data: XOR
            xs = [
                [Value(0.0), Value(0.0)],
                [Value(0.0), Value(1.0)],
                [Value(1.0), Value(0.0)],
                [Value(1.0), Value(1.0)],
            ]
            ys = [Value(0.0), Value(1.0), Value(1.0), Value(0.0)]

            # Train for a few steps
            initial_loss = None
            final_loss = None

            for i in range(20):
                # Forward
                ypred = [model(x) for x in xs]
                loss = sum((yp - yt)**2 for yp, yt in zip(ypred, ys))

                if i == 0:
                    initial_loss = loss.data

                # Backward
                model.zero_grad()
                loss.backward()

                # Update
                for p in model.parameters():
                    p.data -= 0.01 * p.grad

                if i == 19:
                    final_loss = loss.data

            # Loss should decrease
            if final_loss >= initial_loss:
                return False, f"Network not learning: initial_loss={initial_loss:.4f}, final_loss={final_loss:.4f}"

            return True, f"✓ Neural network training: PASSED (loss: {initial_loss:.4f} → {final_loss:.4f})"

        except Exception as e:
            return False, f"✗ Neural network error: {e}"

    def test_topological_sort(self) -> Tuple[bool, str]:
        """Test that backprop traverses graph in correct order."""
        try:
            Value = self.module.Value

            # Create a graph where order matters
            x = Value(2.0)
            y = x + x  # y depends on x
            z = y * y  # z depends on y
            w = z + x  # w depends on z and x

            w.backward()

            # Check all gradients computed
            if x.grad == 0.0:
                return False, "Topological order wrong: x.grad is 0 (not computed)"

            if y.grad == 0.0:
                return False, "Topological order wrong: y.grad is 0 (not computed)"

            if z.grad == 0.0:
                return False, "Topological order wrong: z.grad is 0 (not computed)"

            # Verify gradient values
            # dw/dz = 1, dz/dy = 2*y, dy/dx = 2
            # dw/dx = dw/dz * dz/dy * dy/dx + dw/dx = 1 * 2*4 * 2 + 1 = 17
            expected_dx = 17.0

            if abs(x.grad - expected_dx) > self.tolerance:
                return False, f"Topological order produces wrong gradient: {x.grad} (expected: {expected_dx})"

            return True, "✓ Topological sort: PASSED"

        except Exception as e:
            return False, f"✗ Topological sort error: {e}"

    def run_all_tests(self) -> Tuple[int, int, List[str]]:
        """Run all test cases."""
        tests = [
            ("Basic Operations", self.test_basic_operations),
            ("Chain Rule", self.test_chain_rule),
            ("Gradient Accumulation", self.test_gradient_accumulation),
            ("ReLU Activation", self.test_relu),
            ("Complex Expression", self.test_complex_expression),
            ("Zero Grad", self.test_zero_grad),
            ("Topological Sort", self.test_topological_sort),
            ("Neural Network", self.test_neural_network),
        ]

        passed = 0
        total = len(tests)
        messages = []

        for name, test_func in tests:
            success, message = test_func()
            messages.append(message)

            if success:
                passed += 1

        return passed, total, messages

    def validate(self) -> bool:
        """Run complete validation."""
        print("=" * 70)
        print("NanoGrad Autograd Engine - Validator")
        print("=" * 70)

        # Load module
        if not self.load_module():
            return False

        print(f"\nRunning tests...\n")

        # Run tests
        passed, total, messages = self.run_all_tests()

        # Print results
        for msg in messages:
            print(msg)

        print("\n" + "=" * 70)
        print(f"Results: {passed}/{total} tests passed")
        print("=" * 70)

        if passed == total:
            print("✓ All tests passed! Excellent work!")
            return True
        else:
            print(f"✗ {total - passed} test(s) failed. Keep debugging!")
            return False


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Validate NanoGrad autograd engine implementation"
    )
    parser.add_argument(
        '--file',
        type=str,
        default='nanograd.py',
        help='Path to the Python file to validate (default: nanograd.py)'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Print detailed output'
    )

    args = parser.parse_args()

    # Create validator
    validator = GradientValidator(
        module_path=args.file,
        verbose=args.verbose
    )

    # Run validation
    success = validator.validate()

    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
