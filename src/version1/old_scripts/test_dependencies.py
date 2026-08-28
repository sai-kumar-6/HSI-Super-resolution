"""Test that all dependent files can import the optimized VMamba-Pansharp"""

import sys
import importlib

print("Testing dependencies...")
print("=" * 80)

files_to_test = [
    ('train_vmamba_pansharp', 'Training script'),
    ('evaluate_vmamba_pansharp', 'Evaluation script'),
    ('compare_models', 'Model comparison script'),
    ('visualize_model_output', 'Visualization script'),
]

failed = []
passed = []

for module_name, description in files_to_test:
    try:
        print(f"\n[TESTING] {description} ({module_name}.py)")

        # Try to import the module
        module = importlib.import_module(module_name)

        print(f"[OK] {description} imported successfully")
        passed.append(description)

    except Exception as e:
        print(f"[FAIL] {description} - Error: {e}")
        failed.append((description, str(e)))

print("\n" + "=" * 80)
print(f"\nResults: {len(passed)} passed, {len(failed)} failed")

if passed:
    print("\n[PASSED]:")
    for p in passed:
        print(f"  - {p}")

if failed:
    print("\n[FAILED]:")
    for desc, error in failed:
        print(f"  - {desc}: {error}")
    sys.exit(1)
else:
    print("\n[SUCCESS] All dependencies verified!")
