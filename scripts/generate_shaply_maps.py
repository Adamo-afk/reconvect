import numpy as np
import tensorflow as tf
import shap
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import pandas as pd
from scipy.ndimage import zoom
import gc

# ============================================================================
# 1. COMPUTE SHAP VALUES FOR METEOROLOGICAL MODEL
# ============================================================================

def compute_shap_values_per_input(model, background_data, test_data, 
                                  input_names, max_samples=50):
    """
    Compute SHAP values for each input variable.
    
    Args:
        model: Trained Keras model
        background_data: Background dataset for SHAP (dict or list)
        test_data: Test samples to explain (dict or list)
        input_names: List of input variable names
        max_samples: Max samples to use for background (memory limit)
        
    Returns:
        Dictionary with SHAP values per input
    """
    print("\n" + "="*70)
    print("COMPUTING SHAP VALUES")
    print("="*70)
    
    # Prepare background data (subsample for efficiency)
    if isinstance(background_data, dict):
        background_subset = {
            k: v[:max_samples] for k, v in background_data.items()
        }
    elif isinstance(background_data, list):
        background_subset = [x[:max_samples] for x in background_data]
    else:
        background_subset = background_data[:max_samples]
    
    # Prepare test data (use small sample)
    if isinstance(test_data, dict):
        test_subset = {
            k: v[:4] for k, v in test_data.items()
        }
    elif isinstance(test_data, list):
        test_subset = [x[:4] for x in test_data]
    else:
        test_subset = test_data[:4]
    
    print(f"Background samples: {max_samples}")
    print(f"Test samples: 4")
    
    # Create SHAP explainer
    print("\nInitializing SHAP DeepExplainer...")
    try:
        explainer = shap.DeepExplainer(model, background_subset)
    except Exception as e:
        print(f"DeepExplainer failed: {e}")
        print("Trying GradientExplainer instead...")
        explainer = shap.GradientExplainer(model, background_subset)
    
    # Compute SHAP values
    print("Computing SHAP values (this may take several minutes)...")
    shap_values = explainer.shap_values(test_subset)
    
    # SHAP values structure depends on model output
    # For multi-output: list of arrays, one per output
    # For single output: single array
    if isinstance(shap_values, list):
        # Average across all outputs
        shap_values = [np.mean([sv for sv in shap_values], axis=0)]
    
    print("SHAP computation complete!")
    
    # Process SHAP values per input
    shap_results = {}
    
    if isinstance(test_subset, dict):
        # Dictionary input
        for i, (input_name, input_key) in enumerate(zip(input_names, test_subset.keys())):
            print(f"  Processing {input_name}...")
            
            # Get SHAP values for this input
            if isinstance(shap_values, list) and len(shap_values) > 0:
                input_shap = shap_values[0][i] if i < len(shap_values[0]) else shap_values[0]
            else:
                input_shap = shap_values
            
            # Aggregate: average across samples and time/space
            if len(input_shap.shape) == 5:  # (samples, time, h, w, channels)
                # Average across samples and time
                spatial_importance = np.mean(np.abs(input_shap), axis=(0, 1, 4))  # (h, w)
                global_importance = np.mean(np.abs(input_shap))
            elif len(input_shap.shape) == 4:  # (samples, h, w, channels)
                spatial_importance = np.mean(np.abs(input_shap), axis=(0, 3))
                global_importance = np.mean(np.abs(input_shap))
            else:
                spatial_importance = np.mean(np.abs(input_shap), axis=0)
                global_importance = np.mean(np.abs(input_shap))
            
            shap_results[input_name] = {
                'spatial_importance': spatial_importance,
                'global_importance': global_importance,
                'raw_shap': input_shap
            }
    
    elif isinstance(test_subset, list):
        # List input
        for i, input_name in enumerate(input_names):
            if i >= len(test_subset):
                break
            
            print(f"  Processing {input_name}...")
            
            # SHAP values are returned as a list matching input structure
            if isinstance(shap_values, list) and i < len(shap_values):
                input_shap = shap_values[i]
            else:
                # Fallback: try to extract from combined output
                continue
            
            # Aggregate
            if len(input_shap.shape) == 5:
                spatial_importance = np.mean(np.abs(input_shap), axis=(0, 1, 4))
                global_importance = np.mean(np.abs(input_shap))
            elif len(input_shap.shape) == 4:
                spatial_importance = np.mean(np.abs(input_shap), axis=(0, 3))
                global_importance = np.mean(np.abs(input_shap))
            else:
                spatial_importance = np.mean(np.abs(input_shap), axis=0)
                global_importance = np.mean(np.abs(input_shap))
            
            shap_results[input_name] = {
                'spatial_importance': spatial_importance,
                'global_importance': global_importance,
                'raw_shap': input_shap
            }
    
    return shap_results


# ============================================================================
# 2. VISUALIZE SHAP IMPORTANCE MAPS
# ============================================================================

def visualize_shap_spatial_maps(shap_results, top_n=10, save_path='shap_spatial_maps.png'):
    """
    Visualize spatial importance maps from SHAP values.
    
    Args:
        shap_results: Dictionary from compute_shap_values_per_input
        top_n: Number of top inputs to visualize
        save_path: Path to save figure
    """
    # Sort by global importance
    sorted_inputs = sorted(shap_results.items(), 
                          key=lambda x: x[1]['global_importance'], 
                          reverse=True)
    
    top_inputs = sorted_inputs[:top_n]
    
    # Create subplots
    n_cols = 5
    n_rows = (top_n + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 4*n_rows))
    axes = axes.flatten() if n_rows > 1 else [axes] if n_cols == 1 else axes
    
    for i, (input_name, data) in enumerate(top_inputs):
        spatial_map = data['spatial_importance']
        global_imp = data['global_importance']
        
        im = axes[i].imshow(spatial_map, cmap='Reds')
        axes[i].set_title(f'{input_name}\nSHAP: {global_imp:.4f}', 
                         fontsize=10, fontweight='bold')
        axes[i].axis('off')
        plt.colorbar(im, ax=axes[i], fraction=0.046)
    
    # Hide empty subplots
    for i in range(len(top_inputs), len(axes)):
        axes[i].axis('off')
    
    plt.suptitle('SHAP Spatial Importance Maps (Top Inputs)', 
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved: {save_path}")
    plt.close()


def create_shap_importance_bar_chart(shap_results):
    """
    Create interactive bar chart of SHAP importance values.
    """
    # Prepare data
    inputs = []
    importance = []
    
    for input_name, data in shap_results.items():
        inputs.append(input_name)
        importance.append(data['global_importance'])
    
    # Sort by importance
    sorted_indices = np.argsort(importance)[::-1]
    inputs = [inputs[i] for i in sorted_indices]
    importance = [importance[i] for i in sorted_indices]
    
    # Create bar chart
    fig = go.Figure(data=[
        go.Bar(
            x=inputs,
            y=importance,
            marker_color='crimson',
            text=[f'{v:.4f}' for v in importance],
            textposition='auto'
        )
    ])
    
    fig.update_layout(
        title='Global SHAP Importance (Mean Absolute SHAP Values)',
        xaxis_title='Input Variable',
        yaxis_title='SHAP Importance',
        height=600,
        width=1600,
        showlegend=False
    )
    fig.update_xaxes(tickangle=45)
    
    return fig


# ============================================================================
# 3. COMPARE SHAP WITH GRAD-CAM + XI
# ============================================================================

def compare_shap_with_gradcam(shap_results, gradcam_df, save_dir='comparison'):
    """
    Compare SHAP importance with Grad-CAM + Xi correlations.
    
    Args:
        shap_results: Dictionary from compute_shap_values_per_input
        gradcam_df: DataFrame with Xi correlations from Grad-CAM analysis
        save_dir: Directory to save comparison results
    """
    import os
    os.makedirs(save_dir, exist_ok=True)
    
    print("\n" + "="*70)
    print("COMPARING SHAP vs GRAD-CAM + XI")
    print("="*70)
    
    # Create comparison dataframe
    comparison_data = []
    
    for input_name in shap_results.keys():
        shap_importance = shap_results[input_name]['global_importance']
        
        # Find corresponding Xi value
        xi_row = gradcam_df[gradcam_df['Input'] == input_name]
        xi_value = xi_row['Xi'].values[0] if len(xi_row) > 0 else np.nan
        
        comparison_data.append({
            'Input': input_name,
            'SHAP_Importance': shap_importance,
            'Xi_Correlation': xi_value
        })
    
    df_comparison = pd.DataFrame(comparison_data)
    df_comparison = df_comparison.sort_values('SHAP_Importance', ascending=False)
    
    # Save comparison table
    df_comparison.to_csv(f'{save_dir}/shap_vs_gradcam_comparison.csv', index=False)
    
    print("\nComparison Table (Top 15):")
    print(df_comparison.head(15).to_string(index=False))
    
    # Compute correlation between SHAP and Xi
    if not df_comparison['Xi_Correlation'].isna().all():
        from scipy.stats import spearmanr
        valid_mask = ~df_comparison['Xi_Correlation'].isna()
        shap_vals = df_comparison[valid_mask]['SHAP_Importance'].values
        xi_vals = df_comparison[valid_mask]['Xi_Correlation'].values
        
        correlation, p_value = spearmanr(shap_vals, xi_vals)
        print(f"\nSpearman Correlation (SHAP vs Xi): {correlation:.4f} (p={p_value:.4e})")
    
    # Create scatter plot comparing the two methods
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=df_comparison['SHAP_Importance'],
        y=df_comparison['Xi_Correlation'],
        mode='markers+text',
        marker=dict(size=12, color='steelblue', opacity=0.7),
        text=df_comparison['Input'],
        textposition='top center',
        textfont=dict(size=8)
    ))
    
    fig.update_layout(
        title='SHAP Importance vs Grad-CAM Xi Correlation',
        xaxis_title='SHAP Importance (Mean |SHAP|)',
        yaxis_title='Xi Correlation (Grad-CAM)',
        height=700,
        width=900,
        showlegend=False
    )
    
    # Add diagonal reference line
    max_val = max(df_comparison['SHAP_Importance'].max(), 
                  df_comparison['Xi_Correlation'].max() if not df_comparison['Xi_Correlation'].isna().all() else 1)
    fig.add_shape(type="line", line=dict(dash="dash", color="gray"),
                  x0=0, y0=0, x1=max_val, y1=max_val)
    
    fig.write_html(f'{save_dir}/shap_vs_gradcam_scatter.html')
    print(f"\nSaved: {save_dir}/shap_vs_gradcam_scatter.html")
    
    # Create side-by-side bar chart
    fig_bars = go.Figure()
    
    # Normalize both to [0, 1] for comparison
    shap_norm = df_comparison['SHAP_Importance'] / df_comparison['SHAP_Importance'].max()
    xi_norm = df_comparison['Xi_Correlation'] / df_comparison['Xi_Correlation'].max() \
              if not df_comparison['Xi_Correlation'].isna().all() else np.zeros(len(df_comparison))
    
    fig_bars.add_trace(go.Bar(
        name='SHAP',
        x=df_comparison['Input'],
        y=shap_norm,
        marker_color='crimson'
    ))
    
    fig_bars.add_trace(go.Bar(
        name='Xi (Grad-CAM)',
        x=df_comparison['Input'],
        y=xi_norm,
        marker_color='steelblue'
    ))
    
    fig_bars.update_layout(
        title='Normalized Importance: SHAP vs Grad-CAM Xi',
        xaxis_title='Input Variable',
        yaxis_title='Normalized Importance',
        barmode='group',
        height=600,
        width=1600
    )
    fig_bars.update_xaxes(tickangle=45)
    
    fig_bars.write_html(f'{save_dir}/shap_vs_gradcam_bars.html')
    print(f"Saved: {save_dir}/shap_vs_gradcam_bars.html")
    
    return df_comparison


# ============================================================================
# 4. VISUALIZE SHAP VS GRAD-CAM SPATIAL MAPS SIDE-BY-SIDE
# ============================================================================

def compare_shap_gradcam_spatial(shap_results, gradcam_results, 
                                input_name, save_path='shap_gradcam_spatial_comparison.png'):
    """
    Compare SHAP spatial map with Grad-CAM spatial map for a specific input.
    
    Args:
        shap_results: Dictionary from compute_shap_values_per_input
        gradcam_results: Results dictionary from Grad-CAM analysis
        input_name: Name of input to visualize
        save_path: Path to save figure
    """
    if input_name not in shap_results:
        print(f"Input {input_name} not found in SHAP results")
        return
    
    if input_name not in gradcam_results['input_names']:
        print(f"Input {input_name} not found in Grad-CAM results")
        return
    
    # Get SHAP spatial map
    shap_map = shap_results[input_name]['spatial_importance']
    shap_importance = shap_results[input_name]['global_importance']
    
    # Get Grad-CAM spatial map
    input_idx = gradcam_results['input_names'].index(input_name)
    gradcam_map = gradcam_results['input_gradcams'][input_idx]
    xi_avg = gradcam_results['averaged_xi'][input_idx]
    
    # Resize to same resolution
    if shap_map.shape != gradcam_map.shape:
        scale_factors = (gradcam_map.shape[0] / shap_map.shape[0],
                        gradcam_map.shape[1] / shap_map.shape[1])
        shap_map_resized = zoom(shap_map, scale_factors, order=1)
    else:
        shap_map_resized = shap_map
    
    # Create comparison plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # SHAP map
    im0 = axes[0].imshow(shap_map_resized, cmap='Reds')
    axes[0].set_title(f'SHAP Spatial Importance\n{input_name}\nSHAP: {shap_importance:.4f}', 
                     fontsize=12, fontweight='bold')
    axes[0].axis('off')
    plt.colorbar(im0, ax=axes[0], fraction=0.046, label='|SHAP|')
    
    # Grad-CAM map
    im1 = axes[1].imshow(gradcam_map, cmap='jet')
    axes[1].set_title(f'Grad-CAM Attention\n{input_name}\nXi: {xi_avg:.4f}', 
                     fontsize=12, fontweight='bold')
    axes[1].axis('off')
    plt.colorbar(im1, ax=axes[1], fraction=0.046, label='Attention')
    
    # Overlay
    axes[2].imshow(shap_map_resized, cmap='Reds', alpha=0.5)
    axes[2].imshow(gradcam_map, cmap='Blues', alpha=0.5)
    axes[2].set_title(f'Overlay\n(Red=SHAP, Blue=Grad-CAM)', 
                     fontsize=12, fontweight='bold')
    axes[2].axis('off')
    
    plt.suptitle(f'SHAP vs Grad-CAM Spatial Comparison: {input_name}', 
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()


# ============================================================================
# 5. COMPLETE SHAP ANALYSIS WORKFLOW
# ============================================================================

def run_shap_analysis(model, background_data, test_data, input_names,
                     gradcam_df=None, gradcam_results=None,
                     output_dir='shap_results'):
    """
    Complete SHAP analysis workflow with comparison to Grad-CAM.
    
    Args:
        model: Trained Keras model
        background_data: Background dataset for SHAP
        test_data: Test samples to explain
        input_names: List of input variable names
        gradcam_df: Optional DataFrame with Grad-CAM Xi results
        gradcam_results: Optional results dict from Grad-CAM analysis
        output_dir: Directory to save results
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    # Compute SHAP values
    shap_results = compute_shap_values_per_input(
        model, background_data, test_data, input_names, max_samples=50
    )
    
    # Visualize SHAP spatial maps
    visualize_shap_spatial_maps(shap_results, top_n=15, 
                               save_path=f'{output_dir}/shap_spatial_maps.png')
    
    # Create SHAP importance bar chart
    fig_shap_bars = create_shap_importance_bar_chart(shap_results)
    fig_shap_bars.write_html(f'{output_dir}/shap_importance_bars.html')
    print(f"Saved: {output_dir}/shap_importance_bars.html")
    
    # Compare with Grad-CAM if available
    if gradcam_df is not None:
        comparison_dir = f'{output_dir}/comparison'
        df_comparison = compare_shap_with_gradcam(shap_results, gradcam_df, comparison_dir)
        
        # Create spatial comparisons for top 5 inputs
        if gradcam_results is not None:
            print("\nGenerating SHAP vs Grad-CAM spatial comparisons...")
            top_5 = df_comparison.head(5)['Input'].tolist()
            
            for rank, input_name in enumerate(top_5, 1):
                compare_shap_gradcam_spatial(
                    shap_results, gradcam_results, input_name,
                    save_path=f'{comparison_dir}/spatial_comparison_rank{rank}_{input_name}.png'
                )
    
    print("\n" + "="*70)
    print("SHAP ANALYSIS COMPLETE")
    print("="*70)
    print(f"Results saved to: {output_dir}/")
    
    return shap_results
