import numpy as np
import pandas as pd
import time

# Try to import cuDF, handle case when it's not installed
try:
    import cudf
    import cupy as cp
    HAS_RAPIDS = True
except ImportError:
    HAS_RAPIDS = False
    print("RAPIDS (cuDF/cuPy) not installed. GPU comparison will be skipped.")
    print("To install: conda install -c rapidsai -c conda-forge cudf python=3.9 cudatoolkit=11.5")

# Create sample meteorological data
# Let's simulate a dataset with lat, lon, temperature, humidity, and pressure
# This could represent weather station readings or model output


def create_sample_data(n_rows=1000000):
    """Create sample meteorological data for benchmarking"""
    print(f"Creating sample data with {n_rows:,} rows...")
    
    # Generate synthetic meteorological data
    np.random.seed(42)  # For reproducibility
    
    lat = np.random.uniform(20, 60, n_rows)  # Latitude range
    lon = np.random.uniform(-130, -60, n_rows)  # Longitude range (North America)
    
    # Temperature with spatial correlation (lower at higher latitudes)
    base_temp = 30 - 0.5 * (lat - 20)  # Base temperature decreases with latitude
    temp = base_temp + np.random.normal(0, 5, n_rows)  # Add random variations
    
    # Humidity with some correlation to temperature
    humidity = 70 - 0.3 * (temp - 15) + np.random.normal(0, 10, n_rows)
    humidity = np.clip(humidity, 0, 100)  # Clip to valid range
    
    # Pressure with slight spatial correlation
    pressure = 1013 + 0.1 * lat + np.random.normal(0, 5, n_rows)
    
    # Create pandas DataFrame
    df = pd.DataFrame({
        'lat': lat,
        'lon': lon,
        'temperature': temp,
        'humidity': humidity,
        'pressure': pressure
    })
    
    return df


# CPU version using pandas
def cpu_process_meteorological_data(df):
    """Process meteorological data using CPU (pandas)"""
    # Calculate temperature anomalies (deviation from latitudinal average)
    lat_temp_avg = df.groupby(pd.cut(df['lat'], bins=40))['temperature'].transform('mean')
    df_result = df.copy()
    df_result['temp_anomaly'] = df['temperature'] - lat_temp_avg
    
    # Calculate heat index (a derived quantity based on temperature and humidity)
    # Simplified version of the heat index formula
    temp_f = df['temperature'] * 9/5 + 32  # Convert to Fahrenheit for the formula
    rh = df['humidity']
    df_result['heat_index'] = -42.379 + 2.04901523*temp_f + 10.14333127*rh - \
                             0.22475541*temp_f*rh - 0.00683783*temp_f*temp_f - \
                             0.05481717*rh*rh + 0.00122874*temp_f*temp_f*rh + \
                             0.00085282*temp_f*rh*rh - 0.00000199*temp_f*temp_f*rh*rh
    df_result['heat_index'] = (df_result['heat_index'] - 32) * 5/9  # Back to Celsius
    
    # Find regions of high temperature and low pressure (potential storm indicators)
    df_result['storm_indicator'] = ((df_result['temp_anomaly'] > 5) & 
                                   (df_result['pressure'] < 1005)).astype(int)
    
    # Calculate summary statistics by latitude bands
    lat_bands = pd.cut(df_result['lat'], bins=10)
    summary = df_result.groupby(lat_bands).agg({
        'temperature': ['mean', 'std', 'min', 'max'],
        'humidity': ['mean', 'min', 'max'],
        'pressure': ['mean', 'min', 'max'],
        'storm_indicator': 'sum'
    })
    
    return df_result, summary


# GPU version using cuDF
def gpu_process_meteorological_data(df):
    """Process meteorological data using GPU (cuDF)"""
    if not HAS_RAPIDS:
        return None, None
    
    # Convert pandas DataFrame to cuDF DataFrame
    gdf = cudf.DataFrame.from_pandas(df)
    
    # Calculate temperature anomalies (deviation from latitudinal average)
    lat_bins = cudf.cut(gdf['lat'], bins=40)
    lat_temp_avg = gdf.groupby(lat_bins)['temperature'].transform('mean')
    gdf_result = gdf.copy()
    gdf_result['temp_anomaly'] = gdf['temperature'] - lat_temp_avg
    
    # Calculate heat index (a derived quantity based on temperature and humidity)
    # Simplified version of the heat index formula
    temp_f = gdf['temperature'] * 9/5 + 32  # Convert to Fahrenheit for the formula
    rh = gdf['humidity']
    gdf_result['heat_index'] = -42.379 + 2.04901523*temp_f + 10.14333127*rh - \
                              0.22475541*temp_f*rh - 0.00683783*temp_f*temp_f - \
                              0.05481717*rh*rh + 0.00122874*temp_f*temp_f*rh + \
                              0.00085282*temp_f*rh*rh - 0.00000199*temp_f*temp_f*rh*rh
    gdf_result['heat_index'] = (gdf_result['heat_index'] - 32) * 5/9  # Back to Celsius
    
    # Find regions of high temperature and low pressure (potential storm indicators)
    gdf_result['storm_indicator'] = ((gdf_result['temp_anomaly'] > 5) & 
                                    (gdf_result['pressure'] < 1005)).astype('int32')
    
    # Calculate summary statistics by latitude bands
    lat_bands = cudf.cut(gdf_result['lat'], bins=10)
    gsummary = gdf_result.groupby(lat_bands).agg({
        'temperature': ['mean', 'std', 'min', 'max'],
        'humidity': ['mean', 'min', 'max'],
        'pressure': ['mean', 'min', 'max'],
        'storm_indicator': 'sum'
    })
    
    # Convert back to pandas for comparison
    df_result = gdf_result.to_pandas()
    summary = gsummary.to_pandas()
    
    return df_result, summary


# Main benchmarking code
if __name__ == "__main__":
    # Set the number of rows for the test
    n_rows = 5000000  # 5 million rows - adjust based on your system's memory
    
    # Create sample data
    df = create_sample_data(n_rows)
    print(f"Created dataframe with {len(df):,} rows and {len(df.columns)} columns")
    
    # Run CPU version
    print("\nRunning CPU version (pandas)...")
    cpu_start = time.time()
    cpu_result, cpu_summary = cpu_process_meteorological_data(df)
    cpu_end = time.time()
    cpu_time = cpu_end - cpu_start
    print(f"CPU time: {cpu_time:.4f} seconds")
    
    # Print a small sample of CPU results
    print("\nCPU Results Sample:")
    print(cpu_result.head(3))
    print("\nCPU Summary Sample:")
    print(cpu_summary.head(3))
    
    # Run GPU version if RAPIDS is available
    if HAS_RAPIDS:
        print("\nRunning GPU version (cuDF)...")
        # Warm up GPU
        _ = cp.array([1, 2, 3])
        
        gpu_start = time.time()
        gpu_result, gpu_summary = gpu_process_meteorological_data(df)
        gpu_end = time.time()
        gpu_time = gpu_end - gpu_start
        print(f"GPU time: {gpu_time:.4f} seconds")
        
        # Calculate speedup
        speedup = cpu_time / gpu_time
        print(f"\nGPU speedup: {speedup:.2f}x faster")
        
        # Print a sample of GPU results
        print("\nGPU Results Sample:")
        print(gpu_result.head(3) if gpu_result is not None else "No results")
        
        # Verify results match (approximately)
        if gpu_result is not None:
            print("\nVerifying results...")
            temp_diff = abs(cpu_result['temperature'].mean() - gpu_result['temperature'].mean())
            anomaly_diff = abs(cpu_result['temp_anomaly'].mean() - gpu_result['temp_anomaly'].mean())
            heat_diff = abs(cpu_result['heat_index'].mean() - gpu_result['heat_index'].mean())
            
            print(f"Average temperature difference: {temp_diff:.8f}")
            print(f"Average anomaly difference: {anomaly_diff:.8f}")
            print(f"Average heat index difference: {heat_diff:.8f}")
    else:
        print("\nSkipping GPU comparison as RAPIDS (cuDF) is not installed.")
        print("To enable GPU comparison, install RAPIDS:")
        print("conda install -c rapidsai -c conda-forge cudf python=3.9 cudatoolkit=11.5")
        
    print("\nNote: For meteorological data processing, RAPIDS provides significant speedups")
    print("for common operations. For more complex spatial or temporal operations,")
    print("consider combining with other GPU libraries like cupy-xarray or using dask-cudf")
    print("for larger-than-memory workloads.")
