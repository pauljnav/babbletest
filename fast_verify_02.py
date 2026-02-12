import sys
import math
import importlib
import importlib.util  # Explicitly import the util submodule

def get_snapshot(name):
    # Force a fresh import if it already exists
    if name in sys.modules:
        del sys.modules[name]
    
    # 1. Load the spec and create the module object
    spec = importlib.util.find_spec(name)
    if spec is None:
        raise ImportError(f"Could not find {name}.py in the current directory.")
    mod = importlib.util.module_from_spec(spec)
    
    # 2. Inject a small num_steps before execution
    # This prevents the full 500-step run
    mod.num_steps = 3 
    
    print(f"Executing {name} for 3 steps...")
    spec.loader.exec_module(mod)
    
    # 3. Capture mathematical state
    grad_sum = sum(p.grad for p in mod.params)
    
    return {
        'loss': mod.loss.data,
        'grad_sum': grad_sum,
        'param_count': len(mod.params),
        'weights': [p.data for p in mod.params[:5]]
    }

try:
    orig = get_snapshot('original')
    simp = get_snapshot('simplified')

    print("\n--- Parity Results ---")
    metrics = [
        ("Loss", orig['loss'], simp['loss']),
        ("Grad Sum", orig['grad_sum'], simp['grad_sum']),
        ("Param Count", orig['param_count'], simp['param_count'])
    ]

    all_passed = True
    for label, o_val, s_val in metrics:
        # Use a tight tolerance for mathematical exactness
        match = math.isclose(o_val, s_val, rel_tol=1e-9)
        all_passed &= match
        print(f"{label:<15}: {'✅ PASS' if match else '❌ FAIL'} ({o_val:.8f} vs {s_val:.8f})")

    if all_passed:
        print("\n✅ SUCCESS: Mathematical parity confirmed.")
    else:
        print("\n❌ FAILURE: The simplified version has a logic divergence.")

except Exception as e:
    print(f"\nVerification failed: {e}")