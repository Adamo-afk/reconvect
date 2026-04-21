import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from scipy.spatial.distance import pdist
from scipy import stats


######################## PATCH VALIDATION AND EXPLAINABILITY FUNCTIONS ########################
def validate_patch_processing(original_patches, processed_patches, pooling_factor=4):
    """
    Validate that processed patches match expected transformations of original patches
    
    Parameters:
    -----------
    original_patches : np.array
        Shape: (n_patches, 32, 32) - Original 32x32 patches
    processed_patches : np.array  
        Shape: (n_patches, 8, 8) - Processed 8x8 patches
    pooling_factor : int
        Factor used for average pooling (32/8 = 4)
    """
    
    def manual_average_pool(patch, factor):
        """Manually compute average pooling to compare"""
        h, w = patch.shape
        new_h, new_w = h // factor, w // factor
        pooled = np.zeros((new_h, new_w))
        
        for i in range(new_h):
            for j in range(new_w):
                pooled[i, j] = np.mean(patch[i*factor:(i+1)*factor, j*factor:(j+1)*factor])
        return pooled
    
    validation_results = {
        'patch_differences': [],
        'correlation_scores': [],
        'mse_scores': [],
        'exact_matches': 0,
        'close_matches': 0  # within tolerance
    }
    
    tolerance = 1e-6
    
    for i, (orig, proc) in enumerate(zip(original_patches, processed_patches)):
        # Compute expected processed patch
        expected = manual_average_pool(orig, pooling_factor)
        
        # Calculate differences
        diff = np.abs(expected - proc)
        mse = np.mean(diff**2)
        correlation = np.corrcoef(expected.flatten(), proc.flatten())[0, 1]
        
        validation_results['patch_differences'].append(diff)
        validation_results['mse_scores'].append(mse)
        validation_results['correlation_scores'].append(correlation)
        
        # Check for exact or close matches
        if np.allclose(expected, proc, atol=tolerance):
            if np.array_equal(expected, proc):
                validation_results['exact_matches'] += 1
            else:
                validation_results['close_matches'] += 1
    
    # Summary statistics
    validation_results['mean_mse'] = np.mean(validation_results['mse_scores'])
    validation_results['mean_correlation'] = np.mean(validation_results['correlation_scores'])
    validation_results['total_patches'] = len(original_patches)
    
    return validation_results

def extract_patch_statistics(patches):
    """
    Extract comprehensive statistics from patches for explainability
    
    Parameters:
    -----------
    patches : np.array
        Shape: (n_patches, height, width) - Input patches
    """
    
    stats_dict = {}
    
    for i, patch in enumerate(patches):
        patch_stats = {
            'mean': np.mean(patch),
            'std': np.std(patch),
            'min': np.min(patch),
            'max': np.max(patch),
            'median': np.median(patch),
            'skewness': stats.skew(patch.flatten()),
            'kurtosis': stats.kurtosis(patch.flatten()),
            'range': np.max(patch) - np.min(patch),
            'q25': np.percentile(patch, 25),
            'q75': np.percentile(patch, 75),
            'iqr': np.percentile(patch, 75) - np.percentile(patch, 25),
            'non_zero_count': np.count_nonzero(patch),
            'zero_count': np.count_nonzero(patch == 0),
            'negative_count': np.count_nonzero(patch < 0),
            'positive_count': np.count_nonzero(patch > 0)
        }
        
        # Spatial gradients
        grad_y, grad_x = np.gradient(patch)
        patch_stats['gradient_magnitude_mean'] = np.mean(np.sqrt(grad_x**2 + grad_y**2))
        patch_stats['gradient_magnitude_std'] = np.std(np.sqrt(grad_x**2 + grad_y**2))
        
        # Texture measures (using local standard deviation)
        from scipy.ndimage import uniform_filter
        patch_stats['texture_measure'] = np.std(uniform_filter(patch, size=3))
        
        stats_dict[f'patch_{i}'] = patch_stats
    
    return stats_dict

def visualize_patch_comparison(original_patch, processed_patch, patch_id=0):
    """
    Visualize comparison between original and processed patches
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Original patch
    im1 = axes[0, 0].imshow(original_patch, cmap='viridis')
    axes[0, 0].set_title(f'Original 32x32 Patch {patch_id}')
    axes[0, 0].set_xlabel('X dimension')
    axes[0, 0].set_ylabel('Y dimension')
    plt.colorbar(im1, ax=axes[0, 0])
    
    # Processed patch
    im2 = axes[0, 1].imshow(processed_patch, cmap='viridis')
    axes[0, 1].set_title(f'Processed 8x8 Patch {patch_id}')
    axes[0, 1].set_xlabel('X dimension')
    axes[0, 1].set_ylabel('Y dimension')
    plt.colorbar(im2, ax=axes[0, 1])
    
    # Expected processed patch (manual pooling)
    def manual_pool(patch, factor=4):
        h, w = patch.shape
        new_h, new_w = h // factor, w // factor
        pooled = np.zeros((new_h, new_w))
        for i in range(new_h):
            for j in range(new_w):
                pooled[i, j] = np.mean(patch[i*factor:(i+1)*factor, j*factor:(j+1)*factor])
        return pooled
    
    expected = manual_pool(original_patch)
    im3 = axes[0, 2].imshow(expected, cmap='viridis')
    axes[0, 2].set_title(f'Expected Processed Patch {patch_id}')
    axes[0, 2].set_xlabel('X dimension')
    axes[0, 2].set_ylabel('Y dimension')
    plt.colorbar(im3, ax=axes[0, 2])
    
    # Difference map
    diff = np.abs(expected - processed_patch)
    im4 = axes[1, 0].imshow(diff, cmap='Reds')
    axes[1, 0].set_title(f'Absolute Difference\nMSE: {np.mean(diff**2):.6f}')
    axes[1, 0].set_xlabel('X dimension')
    axes[1, 0].set_ylabel('Y dimension')
    plt.colorbar(im4, ax=axes[1, 0])
    
    # Value distribution comparison
    axes[1, 1].hist(original_patch.flatten(), bins=30, alpha=0.7, label='Original', density=True)
    axes[1, 1].hist(processed_patch.flatten(), bins=30, alpha=0.7, label='Processed', density=True)
    axes[1, 1].hist(expected.flatten(), bins=30, alpha=0.7, label='Expected', density=True)
    axes[1, 1].set_title('Value Distributions')
    axes[1, 1].set_xlabel('Pixel Values')
    axes[1, 1].set_ylabel('Density')
    axes[1, 1].legend()
    
    # Correlation plot
    axes[1, 2].scatter(expected.flatten(), processed_patch.flatten(), alpha=0.6)
    axes[1, 2].plot([expected.min(), expected.max()], [expected.min(), expected.max()], 'r--', lw=2)
    correlation = np.corrcoef(expected.flatten(), processed_patch.flatten())[0, 1]
    axes[1, 2].set_title(f'Expected vs Processed\nCorrelation: {correlation:.4f}')
    axes[1, 2].set_xlabel('Expected Values')
    axes[1, 2].set_ylabel('Processed Values')
    
    plt.tight_layout()
    return fig

def analyze_patch_grid_structure(patches_grid):
    """
    Analyze the structure of the patches as shown in your visualization
    
    Parameters:
    -----------
    patches_grid : np.array
        2D array representing the patches grid visualization
    """
    
    analysis = {
        'grid_shape': patches_grid.shape,
        'value_range': (patches_grid.min(), patches_grid.max()),
        'mean_value': patches_grid.mean(),
        'std_value': patches_grid.std(),
        'unique_values': len(np.unique(patches_grid))
    }
    
    # Analyze horizontal and vertical patterns
    horizontal_variation = np.std(patches_grid, axis=1)  # Variation across each row
    vertical_variation = np.std(patches_grid, axis=0)    # Variation across each column
    
    analysis['horizontal_variation'] = {
        'mean': np.mean(horizontal_variation),
        'std': np.std(horizontal_variation),
        'max_row': np.argmax(horizontal_variation),
        'min_row': np.argmin(horizontal_variation)
    }
    
    analysis['vertical_variation'] = {
        'mean': np.mean(vertical_variation),
        'std': np.std(vertical_variation),
        'max_col': np.argmax(vertical_variation),
        'min_col': np.argmin(vertical_variation)
    }
    
    return analysis

def create_explainability_report(original_patches, processed_patches, output_path=None):
    """
    Generate comprehensive explainability report
    """
    
    print("=== PATCH VALIDATION AND EXPLAINABILITY REPORT ===\n")
    
    # 1. Validation
    print("1. VALIDATION RESULTS:")
    validation = validate_patch_processing(original_patches, processed_patches)
    print(f"   Total patches analyzed: {validation['total_patches']}")
    print(f"   Exact matches: {validation['exact_matches']}")
    print(f"   Close matches: {validation['close_matches']}")
    print(f"   Mean MSE: {validation['mean_mse']:.8f}")
    print(f"   Mean correlation: {validation['mean_correlation']:.6f}")
    
    # 2. Statistical analysis
    print("\n2. STATISTICAL ANALYSIS:")
    orig_stats = extract_patch_statistics(original_patches[:5])  # Sample first 5
    proc_stats = extract_patch_statistics(processed_patches[:5])
    
    print("   Original patches (sample):")
    for i in range(min(3, len(orig_stats))):
        stats = orig_stats[f'patch_{i}']
        print(f"     Patch {i}: mean={stats['mean']:.3f}, std={stats['std']:.3f}, range={stats['range']:.3f}")
    
    print("   Processed patches (sample):")
    for i in range(min(3, len(proc_stats))):
        stats = proc_stats[f'patch_{i}']
        print(f"     Patch {i}: mean={stats['mean']:.3f}, std={stats['std']:.3f}, range={stats['range']:.3f}")
    
    # 3. Recommendations
    print("\n3. RECOMMENDATIONS:")
    if validation['mean_correlation'] > 0.99:
        print("   ✓ Strong correlation between expected and processed patches")
    elif validation['mean_correlation'] > 0.95:
        print("   ⚠ Good correlation, but check for systematic differences")
    else:
        print("   ✗ Low correlation - investigate processing pipeline")
    
    if validation['mean_mse'] < 1e-6:
        print("   ✓ Very low MSE - processing appears correct")
    elif validation['mean_mse'] < 1e-3:
        print("   ⚠ Moderate MSE - acceptable but monitor")
    else:
        print("   ✗ High MSE - significant differences detected")
    
    return validation

############################## PATCH ANALYSIS FUNCTIONS ##############################
def analyze_patch_information_content(patches):
    """
    Analyze information content and diversity in patches
    
    Parameters:
    -----------
    patches : np.array
        Shape: (n_patches, height, width) - Input patches
    """
    
    # Flatten patches for analysis
    patches_flat = patches.reshape(patches.shape[0], -1)
    
    analysis = {}
    
    # 1. Information content metrics
    analysis['entropy'] = []
    for patch in patches_flat:
        # Discretize values for entropy calculation
        hist, _ = np.histogram(patch, bins=50, density=True)
        hist = hist[hist > 0]  # Remove zero bins
        entropy = -np.sum(hist * np.log2(hist + 1e-10))
        analysis['entropy'].append(entropy)
    
    # 2. Patch diversity using pairwise distances
    distances = pdist(patches_flat, metric='euclidean')
    analysis['pairwise_distances'] = distances
    analysis['mean_distance'] = np.mean(distances)
    analysis['std_distance'] = np.std(distances)
    
    # 3. PCA analysis for dimensionality
    pca = PCA()
    pca_result = pca.fit_transform(patches_flat)
    
    # Find number of components for 95% variance
    cumsum_var = np.cumsum(pca.explained_variance_ratio_)
    n_components_95 = np.argmax(cumsum_var >= 0.95) + 1
    
    analysis['pca'] = {
        'explained_variance_ratio': pca.explained_variance_ratio_,
        'cumulative_variance': cumsum_var,
        'n_components_95_variance': n_components_95,
        'effective_rank': np.sum(pca.explained_variance_ratio_ > 0.01)
    }
    
    # 4. Clustering analysis
    optimal_k = min(10, len(patches) // 5)  # Reasonable upper bound
    if optimal_k > 1:
        kmeans = KMeans(n_clusters=optimal_k, random_state=42)
        cluster_labels = kmeans.fit_predict(patches_flat)
        analysis['clustering'] = {
            'labels': cluster_labels,
            'cluster_centers': kmeans.cluster_centers_,
            'inertia': kmeans.inertia_,
            'n_clusters': optimal_k
        }
    
    return analysis

def detect_patch_anomalies(patches, contamination=0.1):
    """
    Detect anomalous patches that might indicate data quality issues
    """
    from sklearn.ensemble import IsolationForest
    
    patches_flat = patches.reshape(patches.shape[0], -1)
    
    # Isolation Forest for anomaly detection
    iso_forest = IsolationForest(contamination=contamination, random_state=42)
    anomaly_labels = iso_forest.fit_predict(patches_flat)
    
    # Statistical anomalies
    patch_means = np.mean(patches_flat, axis=1)
    patch_stds = np.std(patches_flat, axis=1)
    
    # Z-score based anomalies
    mean_z_scores = np.abs((patch_means - np.mean(patch_means)) / np.std(patch_means))
    std_z_scores = np.abs((patch_stds - np.mean(patch_stds)) / np.std(patch_stds))
    
    statistical_anomalies = (mean_z_scores > 3) | (std_z_scores > 3)
    
    return {
        'isolation_forest_anomalies': anomaly_labels == -1,
        'statistical_anomalies': statistical_anomalies,
        'anomaly_scores': iso_forest.decision_function(patches_flat),
        'mean_z_scores': mean_z_scores,
        'std_z_scores': std_z_scores
    }

def compare_original_vs_processed_information(original_patches, processed_patches):
    """
    Compare information content between original and processed patches
    """
    
    print("=== INFORMATION CONTENT COMPARISON ===\n")
    
    # Analyze both sets
    orig_analysis = analyze_patch_information_content(original_patches)
    proc_analysis = analyze_patch_information_content(processed_patches)
    
    # Compare entropy
    orig_entropy = np.mean(orig_analysis['entropy'])
    proc_entropy = np.mean(proc_analysis['entropy'])
    entropy_retention = proc_entropy / orig_entropy
    
    print(f"1. ENTROPY ANALYSIS:")
    print(f"   Original patches mean entropy: {orig_entropy:.3f}")
    print(f"   Processed patches mean entropy: {proc_entropy:.3f}")
    print(f"   Information retention: {entropy_retention:.3f} ({entropy_retention*100:.1f}%)")
    
    # Compare diversity
    orig_diversity = orig_analysis['mean_distance']
    proc_diversity = proc_analysis['mean_distance']
    diversity_retention = proc_diversity / orig_diversity
    
    print(f"\n2. DIVERSITY ANALYSIS:")
    print(f"   Original patches mean distance: {orig_diversity:.3f}")
    print(f"   Processed patches mean distance: {proc_diversity:.3f}")
    print(f"   Diversity retention: {diversity_retention:.3f} ({diversity_retention*100:.1f}%)")
    
    # Compare effective dimensionality
    orig_effective_rank = orig_analysis['pca']['effective_rank']
    proc_effective_rank = proc_analysis['pca']['effective_rank']
    
    print(f"\n3. DIMENSIONALITY ANALYSIS:")
    print(f"   Original patches effective rank: {orig_effective_rank}")
    print(f"   Processed patches effective rank: {proc_effective_rank}")
    print(f"   Rank retention: {proc_effective_rank/orig_effective_rank:.3f}")
    
    return {
        'entropy_retention': entropy_retention,
        'diversity_retention': diversity_retention,
        'rank_retention': proc_effective_rank/orig_effective_rank,
        'original_analysis': orig_analysis,
        'processed_analysis': proc_analysis
    }

def visualize_patch_grid_interpretation(patches_grid, patch_positions=None):
    """
    Create interpretable visualization of the patch grid
    
    Parameters:
    -----------
    patches_grid : np.array
        2D grid showing patches (like in your image)
    patch_positions : list of tuples
        Optional: (row, col) positions corresponding to geographic locations
    """
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 1. Original grid
    im1 = axes[0, 0].imshow(patches_grid, cmap='viridis', aspect='auto')
    axes[0, 0].set_title('Patches Grid Visualization')
    axes[0, 0].set_xlabel('Height Dimension')
    axes[0, 0].set_ylabel('Patch Dimension')
    plt.colorbar(im1, ax=axes[0, 0])
    
    # 2. Row-wise analysis (each row represents one patch)
    row_means = np.mean(patches_grid, axis=1)
    row_stds = np.std(patches_grid, axis=1)
    
    axes[0, 1].plot(row_means, 'b-', label='Mean', linewidth=2)
    axes[0, 1].fill_between(range(len(row_means)), 
                           row_means - row_stds, 
                           row_means + row_stds, 
                           alpha=0.3, label='±1 std')
    axes[0, 1].set_title('Per-Patch Statistics')
    axes[0, 1].set_xlabel('Patch Index')
    axes[0, 1].set_ylabel('Value')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Column-wise analysis (height/level dimension)
    col_means = np.mean(patches_grid, axis=0)
    col_stds = np.std(patches_grid, axis=0)
    
    axes[0, 2].plot(col_means, 'r-', label='Mean', linewidth=2)
    axes[0, 2].fill_between(range(len(col_means)), 
                           col_means - col_stds, 
                           col_means + col_stds, 
                           alpha=0.3, label='±1 std')
    axes[0, 2].set_title('Per-Height Statistics')
    axes[0, 2].set_xlabel('Height/Level Index')
    axes[0, 2].set_ylabel('Value')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)
    
    # 4. Heatmap with annotations for extreme values
    masked_grid = np.ma.masked_where(
        (patches_grid < np.percentile(patches_grid, 5)) | 
        (patches_grid > np.percentile(patches_grid, 95)), 
        patches_grid
    )
    
    im4 = axes[1, 0].imshow(masked_grid, cmap='RdYlBu_r', aspect='auto')
    axes[1, 0].set_title('Extreme Values Highlighted\n(5th and 95th percentiles masked)')
    axes[1, 0].set_xlabel('Height Dimension')
    axes[1, 0].set_ylabel('Patch Dimension')
    plt.colorbar(im4, ax=axes[1, 0])
    
    # 5. Value distribution
    axes[1, 1].hist(patches_grid.flatten(), bins=50, alpha=0.7, density=True)
    axes[1, 1].axvline(np.mean(patches_grid), color='red', linestyle='--', label=f'Mean: {np.mean(patches_grid):.2f}')
    axes[1, 1].axvline(np.median(patches_grid), color='green', linestyle='--', label=f'Median: {np.median(patches_grid):.2f}')
    axes[1, 1].set_title('Value Distribution')
    axes[1, 1].set_xlabel('Pixel Values')
    axes[1, 1].set_ylabel('Density')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    # 6. Correlation structure
    correlation_matrix = np.corrcoef(patches_grid)
    im6 = axes[1, 2].imshow(correlation_matrix, cmap='coolwarm', vmin=-1, vmax=1)
    axes[1, 2].set_title('Inter-Patch Correlations')
    axes[1, 2].set_xlabel('Patch Index')
    axes[1, 2].set_ylabel('Patch Index')
    plt.colorbar(im6, ax=axes[1, 2])
    
    plt.tight_layout()
    return fig

def generate_training_data_quality_report(original_patches, processed_patches):
    """
    Comprehensive quality assessment for neural network training data
    """
    
    print("=== TRAINING DATA QUALITY REPORT ===\n")
    
    # 1. Basic validation
    validation = validate_patch_processing(original_patches, processed_patches)
    
    # 2. Information content analysis
    info_comparison = compare_original_vs_processed_information(original_patches, processed_patches)
    
    # 3. Anomaly detection
    orig_anomalies = detect_patch_anomalies(original_patches)
    proc_anomalies = detect_patch_anomalies(processed_patches)
    
    print(f"\n4. ANOMALY DETECTION:")
    print(f"   Original patches:")
    print(f"     Isolation Forest anomalies: {np.sum(orig_anomalies['isolation_forest_anomalies'])}")
    print(f"     Statistical anomalies: {np.sum(orig_anomalies['statistical_anomalies'])}")
    print(f"   Processed patches:")
    print(f"     Isolation Forest anomalies: {np.sum(proc_anomalies['isolation_forest_anomalies'])}")
    print(f"     Statistical anomalies: {np.sum(proc_anomalies['statistical_anomalies'])}")
    
    # 5. Training suitability assessment
    print(f"\n5. TRAINING SUITABILITY ASSESSMENT:")
    
    # Check for sufficient diversity
    proc_diversity = info_comparison['processed_analysis']['mean_distance']
    if proc_diversity > np.std(processed_patches.flatten()):
        print("   ✓ Sufficient patch diversity for training")
    else:
        print("   ⚠ Limited patch diversity - consider data augmentation")
    
    # Check for information retention
    if info_comparison['entropy_retention'] > 0.7:
        print("   ✓ Good information retention during processing")
    else:
        print("   ⚠ Significant information loss during processing")
    
    # Check for anomalies
    anomaly_rate = np.sum(proc_anomalies['isolation_forest_anomalies']) / len(processed_patches)
    if anomaly_rate < 0.05:
        print("   ✓ Low anomaly rate in processed data")
    else:
        print(f"   ⚠ High anomaly rate ({anomaly_rate:.1%}) - review data quality")
    
    return {
        'validation': validation,
        'information_analysis': info_comparison,
        'anomalies': {'original': orig_anomalies, 'processed': proc_anomalies}
    }


def load_and_analyze_patches(original_path, processed_path):
    """
    Example function to load and analyze patches from saved files
    """
    # Load your patches (adapt to your file format)
    # original_patches = np.load(original_path)  # Shape: (n_patches, 32, 32)
    # processed_patches = np.load(processed_path)  # Shape: (n_patches, 8, 8)
    
    # Generate comprehensive report
    # quality_report = generate_training_data_quality_report(original_patches, processed_patches)
    
    # Create visualizations
    # fig1 = visualize_patch_comparison(original_patches[0], processed_patches[0])
    # fig2 = visualize_patch_grid_interpretation(processed_patches.reshape(-1, processed_patches.shape[-1]))