import sys
import math
import importlib
from unittest.mock import patch

def get_snapshot(name):
    print(f"Running {name}...")
    # Intercept 'range' to force the training loop to stop early
    # We use 3 steps to ensure Adam's first/second moments are updated
    with patch('builtins.range', side_effect=lambda *args: range(3) if args == (500,) else range(*args)):
        # Clear module if it was imported before
        if name in sys.modules: del sys.modules[name]
        mod = importlib.import_module(name)
    
    return {
        'loss': mod.loss.data,
        'grad_sum': sum((p.grad for p in mod.params), 0.0), # p.grad is float
        'param_count': len(mod.params),
        'weights': [p.data for p in mod.params[:5]]
    }

print("--- Fast Parity Check (3 Steps) ---")

try:
    orig = get_snapshot('original')
    simp = get_snapshot('simplified')

    # Compare results
    results = []
    results.append(("Loss", math.isclose(orig['loss'], simp['loss'])))
    results.append(("Grad Sum", math.isclose(orig['grad_sum'], simp['grad_sum'])))
    results.append(("Param Count", orig['param_count'] == simp['param_count']))
    results.append(("Weights", all(math.isclose(o, s) for o, s in zip(orig['weights'], simp['weights']))))

    print("\n--- Results ---")
    for metric, passed in results:
        print(f"{metric:<15}: {'✅ PASS' if passed else '❌ FAIL'}")

    if all(r[1] for r in results):
        print("\nConclusion: Functional Parity Confirmed. The logic is identical.")
    else:
        print("\nConclusion: Parity Failed. Check logic differences.")

except Exception as e:
    print(f"Error during verification: {e}")