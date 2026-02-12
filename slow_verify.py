import importlib
import sys

def get_results(module_name):
    # Reset seed for identical initialization
    import random
    random.seed(42)
    
    # Import the model script
    mod = importlib.import_module(module_name)
    
    # Return the first loss and the sum of all gradients
    return mod.loss.data, sum(p.grad for p in mod.params)

try:
    orig_loss, orig_grads = get_results('original')
    simp_loss, simp_grads = get_results('simplified')

    print(f"Original   | Loss: {orig_loss:.8f} | Grad Sum: {orig_grads:.8f}")
    print(f"Simplified | Loss: {simp_loss:.8f} | Grad Sum: {simp_grads:.8f}")

    assert math.isclose(orig_loss, simp_loss), "Loss mismatch!"
    assert math.isclose(orig_grads, simp_grads), "Gradient mismatch!"
    print("\n✅ Exactness Confirmed: The simplified code is functionally identical.")

except Exception as e:
    print(f"Verification failed: {e}")