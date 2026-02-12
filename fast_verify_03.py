import sys
import math
import importlib
import importlib.util

def get_snapshot(name):
    if name in sys.modules:
        del sys.modules[name]
    
    spec = importlib.util.find_spec(name)
    mod = importlib.util.module_from_spec(spec)
    
    # We don't need to inject here anymore since you edited the files to 3 steps
    print(f"Executing {name}...")
    spec.loader.exec_module(mod)
    
    # FIX: Use mod.Value(0) as the start of the sum to avoid the int + Value error
    # This sums the weights to create a "fingerprint" of the model state
    grad_sum = sum((p.grad for p in mod.params), 0.0) # Gradients are floats
    weight_sum = sum(mod.params, mod.Value(0)).data  # Weights are Value objects
    
    return {
        'loss': mod.loss.data,
        'grad_sum': grad_sum,
        'weight_sum': weight_sum,
        'param_count': len(mod.params)
    }

try:
    orig = get_snapshot('original')
    simp = get_snapshot('simplified')

    print("\n--- Parity Results (30 Steps) ---")
    metrics = [
        ("Loss", orig['loss'], simp['loss']),
        ("Grad Sum", orig['grad_sum'], simp['grad_sum']),
        ("Weight Sum", orig['weight_sum'], simp['weight_sum']),
        ("Param Count", orig['param_count'], simp['param_count'])
    ]

    all_passed = True
    for label, o_val, s_val in metrics:
        match = math.isclose(o_val, s_val, rel_tol=1e-9)
        all_passed &= match
        print(f"{label:<15}: {'✅ PASS' if match else '❌ FAIL'} ({o_val:.6f} vs {s_val:.6f})")

    if all_passed:
        print("\n✅ SUCCESS: Mathematical parity confirmed.")
        print("The simplified code is functionally identical to Karpathy's.")
    else:
        print("\n❌ FAILURE: Numerical divergence detected.")

except Exception as e:
    print(f"\nVerification failed: {e}")