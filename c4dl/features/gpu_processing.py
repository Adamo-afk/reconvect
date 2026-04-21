import numpy as np
import time
try:
    import cupy as cp
    from cupyx.scipy import ndimage
    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False
    print("CuPy not installed. GPU comparison will be skipped.")
    print("To install: pip install cupy-cuda11x  (replace with your CUDA version)")


# Create a large 2D array representing meteorological data (e.g., temperature grid)
# This simulates a high-resolution grid covering a large area
SIZE = 4000
print(f"Creating {SIZE}x{SIZE} matrix (~{SIZE*SIZE*8/1e6:.1f} MB)...")
cpu_data = np.random.random((SIZE, SIZE)).astype(np.float32)


def cpu_spatial_average(data, window_size=5):
    """Compute spatial moving average on CPU"""
    result = np.zeros_like(data)
    half_window = window_size // 2
    
    for i in range(half_window, data.shape[0] - half_window):
        for j in range(half_window, data.shape[1] - half_window):
            # Extract window and compute average
            window = data[i-half_window:i+half_window+1, j-half_window:j+half_window+1]
            result[i, j] = np.mean(window)
    
    return result


def gpu_spatial_average(data, window_size=5):
    """Compute spatial moving average on GPU using CuPy"""
    if not HAS_CUPY:
        return None
        
    # Transfer data to GPU
    gpu_data = cp.asarray(data)
    
    # Create a simple convolution kernel for averaging
    kernel_size = window_size
    kernel = cp.ones((kernel_size, kernel_size), dtype=cp.float32) / (kernel_size * kernel_size)
    
    # For 2D convolution with CuPy, we can use the ndimage module
    result = ndimage.convolve(gpu_data, kernel, mode='constant', cval=0.0)
    
    # Transfer result back to CPU
    return cp.asnumpy(result)


# Run CPU version
print("\nRunning CPU version...")
cpu_start = time.time()
cpu_result = cpu_spatial_average(cpu_data)
cpu_end = time.time()
cpu_time = cpu_end - cpu_start
print(f"CPU time: {cpu_time:.4f} seconds")

# Run GPU version if CuPy is available
if HAS_CUPY:
    print("\nRunning GPU version...")
    # Warm up the GPU (first CUDA call has overhead)
    _ = cp.array([1, 2, 3])
    
    gpu_start = time.time()
    gpu_result = gpu_spatial_average(cpu_data)
    gpu_end = time.time()
    gpu_time = gpu_end - gpu_start
    print(f"GPU time: {gpu_time:.4f} seconds")
    
    # Calculate speedup
    speedup = cpu_time / gpu_time
    print(f"\nGPU speedup: {speedup:.2f}x faster")
    
    # Verify results are similar
    if gpu_result is not None:
        diff = np.abs(cpu_result - gpu_result).mean()
        print(f"Average difference between CPU and GPU results: {diff:.8f}")
else:
    print("\nSkipping GPU comparison as CuPy is not installed.")
    print("To enable GPU comparison, install CuPy with: pip install cupy-cuda11x")

print("\nNote: For best GPU performance, you should use more sophisticated libraries")
print("like Numba or specialized atmospheric science packages that support CUDA.")
