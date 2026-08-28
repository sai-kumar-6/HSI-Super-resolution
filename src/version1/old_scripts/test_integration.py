"""Simple integration test for optimized VMamba-Pansharp"""

print("Testing import...")
try:
    from vmamba_pansharp import VMambaPansharp
    print("[OK] VMambaPansharp imported successfully")
except Exception as e:
    print(f"[FAIL] Import error: {e}")
    exit(1)

print("\nTesting model creation...")
try:
    model = VMambaPansharp(in_channels=102, out_channels=102, d_model=32, scale=4)
    print(f"[OK] Model created successfully")

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"[OK] Total parameters: {total_params:,}")
except Exception as e:
    print(f"[FAIL] Model creation error: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

print("\n[SUCCESS] Integration test passed!")
