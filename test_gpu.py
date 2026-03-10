import cupy as cp
try:
    count = cp.cuda.runtime.getDeviceCount()
    name = cp.cuda.runtime.getDeviceProperties(0)['name'].decode('utf-8')
    print(f"SUCCESS: Found {count} GPU(s).")
    print(f"GPU 0: {name}")
    
    # Quick math test
    x = cp.array([1, 2, 3])
    print(f"Math Test: {x * 2}")
except Exception as e:
    print(f"FAILURE: {e}")
