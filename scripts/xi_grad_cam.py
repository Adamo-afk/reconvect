# import numpy as np
# import tensorflow as tf
# from tensorflow.keras.models import Model
# import matplotlib.pyplot as plt
# import plotly.graph_objects as go
# import plotly.express as px
# import pandas as pd
# from scipy.ndimage import zoom
# import gc

# # ============================================================================
# # 1. XI CORRELATION COEFFICIENT
# # ============================================================================

# def compute_xi_coefficient(x, y):
#     """Compute Xi coefficient between two 1D arrays"""
#     n = len(x)
#     if n < 2:
#         return 0.0
    
#     sorted_indices = np.argsort(x)
#     y_sorted = y[sorted_indices]
#     y_ranks = np.argsort(np.argsort(y_sorted))
    
#     numerator = np.sum(np.abs(np.diff(y_ranks)))
#     denominator = 2 * n**2 / 3
    
#     if denominator == 0:
#         return 0.0
    
#     xi = 1 - (numerator / denominator)
#     return max(0.0, min(1.0, xi))


# # ============================================================================
# # 2. IMPROVED GRAD-CAM FOR SPECIFIC INPUT CHANNELS
# # ============================================================================

# def compute_gradcam_for_concatenated_inputs(model, input_data_single, 
#                                            concat_layer_name, resblock_name,
#                                            input_channel_ranges):
#     """
#     Compute Grad-CAM for each input group separately.
    
#     Args:
#         model: Keras model
#         input_data_single: Single sample input (batch_size=1)
#         concat_layer_name: Name of concatenation layer
#         resblock_name: Name of ResBlock layer
#         input_channel_ranges: List of (start, end) channel indices for each input
        
#     Returns:
#         List of Grad-CAM heatmaps, one per input
#     """
#     # Get the concatenated layer and ResBlock output
#     concat_layer = model.get_layer(concat_layer_name)
#     resblock_layer = model.get_layer(resblock_name)
    
#     # Create model that outputs concatenated features, ResBlock output, and final prediction
#     intermediate_model = Model(
#         inputs=model.input,
#         outputs=[concat_layer.output, resblock_layer.output, model.output]
#     )
    
#     gradcams = []
    
#     # Process each input channel group
#     for input_idx, (start_ch, end_ch) in enumerate(input_channel_ranges):
#         with tf.GradientTape() as tape:
#             concat_output, resblock_output, predictions = intermediate_model(input_data_single)
            
#             # Watch the concatenated output
#             tape.watch(concat_output)
            
#             # Average predictions across time and spatial dimensions
#             pred_mean = tf.reduce_mean(predictions)
        
#         # Get gradients w.r.t. concatenated output
#         grads = tape.gradient(pred_mean, concat_output)
        
#         if grads is None:
#             print(f"WARNING: No gradients for input {input_idx}")
#             # Create dummy heatmap
#             dummy_shape = resblock_output.shape[2:4]
#             gradcams.append(np.zeros(dummy_shape))
#             continue
        
#         # Extract gradients and activations for this input's channels
#         input_grads = grads[..., start_ch:end_ch]
#         input_activations = concat_output[..., start_ch:end_ch]
        
#         # Compute importance weights (average gradients globally)
#         if len(input_grads.shape) == 5:  # (batch, time, h, w, channels)
#             weights = tf.reduce_mean(input_grads, axis=[0, 1, 2, 3])
#             activations = tf.reduce_mean(input_activations, axis=1)  # Average over time
#         else:  # (batch, h, w, channels)
#             weights = tf.reduce_mean(input_grads, axis=[0, 1, 2])
#             activations = input_activations[0]
        
#         # Weight activations by importance
#         weighted_activations = activations * weights
        
#         # Sum across channels to get heatmap
#         heatmap = tf.reduce_sum(weighted_activations, axis=-1)
        
#         # Apply ReLU and normalize
#         heatmap = tf.maximum(heatmap, 0)
#         heatmap_max = tf.reduce_max(heatmap)
#         if heatmap_max > 1e-10:
#             heatmap = heatmap / heatmap_max
        
#         gradcams.append(heatmap.numpy()[0] if len(heatmap.shape) > 2 else heatmap.numpy())
    
#     return gradcams


# # ============================================================================
# # 3. BATCH PROCESSING WITH AVERAGING
# # ============================================================================

# def process_batch_samples(model, input_data_batch, concat_layer_name, 
#                          resblock_name, input_channel_ranges, num_samples=None):
#     """
#     Process multiple samples from batch and average Grad-CAM results.
    
#     Args:
#         model: Keras model
#         input_data_batch: Batch of inputs
#         concat_layer_name: Concatenation layer name
#         resblock_name: ResBlock name
#         input_channel_ranges: Channel ranges for each input
#         num_samples: Number of samples to process (None = all)
        
#     Returns:
#         averaged_gradcams: List of averaged Grad-CAM heatmaps
#         all_gradcams: List of lists (per sample, per input)
#     """
#     # Determine batch size
#     if isinstance(input_data_batch, dict):
#         batch_size = list(input_data_batch.values())[0].shape[0]
#     elif isinstance(input_data_batch, list):
#         batch_size = input_data_batch[0].shape[0]
#     else:
#         batch_size = input_data_batch.shape[0]
    
#     if num_samples is None:
#         num_samples = batch_size
#     num_samples = min(num_samples, batch_size)
    
#     print(f"\nProcessing {num_samples} samples from batch...")
    
#     all_gradcams = []  # [sample][input]
    
#     for sample_idx in range(num_samples):
#         print(f"  Sample {sample_idx + 1}/{num_samples}...", end='\r')
        
#         # Extract single sample
#         if isinstance(input_data_batch, dict):
#             single_sample = {k: v[sample_idx:sample_idx+1] for k, v in input_data_batch.items()}
#         elif isinstance(input_data_batch, list):
#             single_sample = [x[sample_idx:sample_idx+1] for x in input_data_batch]
#         else:
#             single_sample = input_data_batch[sample_idx:sample_idx+1]
        
#         # Compute Grad-CAM for this sample
#         gradcams = compute_gradcam_for_concatenated_inputs(
#             model, single_sample, concat_layer_name, 
#             resblock_name, input_channel_ranges
#         )
        
#         all_gradcams.append(gradcams)
        
#         # Clear memory
#         tf.keras.backend.clear_session()
#         gc.collect()
    
#     print(f"\n  Completed {num_samples} samples")
    
#     # Average across samples
#     num_inputs = len(all_gradcams[0])
#     averaged_gradcams = []
    
#     for input_idx in range(num_inputs):
#         # Stack all samples for this input
#         stacked = np.stack([all_gradcams[s][input_idx] for s in range(num_samples)])
#         # Average
#         averaged = np.mean(stacked, axis=0)
#         averaged_gradcams.append(averaged)
    
#     return averaged_gradcams, all_gradcams


# # ============================================================================
# # 4. COMPUTE CORRELATIONS WITH ALL 12 OUTPUT TIMESTEPS
# # ============================================================================

# def compute_correlations_with_outputs(gradcam_heatmap, output_predictions):
#     """
#     Compute Xi correlation between Grad-CAM and each of 12 output timesteps.
    
#     Args:
#         gradcam_heatmap: 2D array (H, W)
#         output_predictions: 3D or 4D array (time, H, W) or (time, H, W, 1)
        
#     Returns:
#         List of Xi correlation values for each timestep
#     """
#     if len(output_predictions.shape) == 4:
#         output_predictions = output_predictions[..., 0]  # Remove channel dim
    
#     num_timesteps = output_predictions.shape[0]
#     xi_correlations = []
    
#     for t in range(num_timesteps):
#         output_t = output_predictions[t]
        
#         # Resize Grad-CAM to match output resolution
#         if gradcam_heatmap.shape != output_t.shape:
#             scale_factors = (output_t.shape[0] / gradcam_heatmap.shape[0],
#                            output_t.shape[1] / gradcam_heatmap.shape[1])
#             gradcam_resized = zoom(gradcam_heatmap, scale_factors, order=1)
#         else:
#             gradcam_resized = gradcam_heatmap
        
#         # Flatten
#         x = gradcam_resized.flatten()
#         y = output_t.flatten()
        
#         # Remove NaN/Inf
#         mask = np.isfinite(x) & np.isfinite(y)
#         x = x[mask]
#         y = y[mask]
        
#         if len(x) < 2 or np.std(x) == 0 or np.std(y) == 0:
#             xi_correlations.append(np.nan)
#             continue
        
#         # Compute Xi correlation only
#         xi_correlations.append(compute_xi_coefficient(x, y))
    
#     return xi_correlations


# # ============================================================================
# # 5. MAIN ANALYSIS WITH PROPER BATCH PROCESSING
# # ============================================================================

# def analyze_inputs_comprehensive(model, input_data_batch, concat_layer_name,
#                                  resblock_name, input_names, 
#                                  num_samples=8, max_samples_memory=4):
#     """
#     Complete analysis with batch processing and per-timestep correlations.
#     """
#     # Get model predictions for the batch
#     print("\nComputing model predictions...")
#     predictions = model.predict(input_data_batch, batch_size=1, verbose=0)
    
#     # Determine input channel ranges from concatenation layer
#     concat_layer = model.get_layer(concat_layer_name)
#     total_channels = concat_layer.output.shape[-1]
#     num_inputs = len(input_names)
    
#     # Assume equal channel distribution (adjust if needed)
#     channels_per_input = total_channels // num_inputs
#     input_channel_ranges = [(i * channels_per_input, 
#                             (i + 1) * channels_per_input if i < num_inputs - 1 else total_channels)
#                            for i in range(num_inputs)]
    
#     print(f"Total channels: {total_channels}")
#     print(f"Inputs: {num_inputs}")
#     print(f"Channels per input: ~{channels_per_input}")
    
#     # Process batch samples with memory management
#     num_samples = min(num_samples, max_samples_memory)
#     averaged_gradcams, all_gradcams = process_batch_samples(
#         model, input_data_batch, concat_layer_name, resblock_name,
#         input_channel_ranges, num_samples=num_samples
#     )
    
#     # Compute correlations for each input against all 12 timesteps
#     print("\nComputing Xi correlations across timesteps...")
    
#     results = {
#         'input_names': input_names,
#         'averaged_gradcams': averaged_gradcams,
#         'xi_per_timestep': [],  # [input][timestep]
#         'averaged_xi': [],  # [input] - averaged across timesteps
#     }
    
#     for input_idx, input_name in enumerate(input_names):
#         print(f"  Processing {input_name}...")
        
#         # Average Grad-CAM for this input across samples
#         avg_gradcam = averaged_gradcams[input_idx]
        
#         # Compute Xi correlations with each of the 12 output timesteps
#         # Use first sample's predictions for consistency
#         sample_predictions = predictions[0]  # (12, H, W, 1)
        
#         xi_per_timestep = compute_correlations_with_outputs(avg_gradcam, sample_predictions)
#         results['xi_per_timestep'].append(xi_per_timestep)
        
#         # Average across timesteps
#         avg_xi = np.nanmean(xi_per_timestep)
#         results['averaged_xi'].append(avg_xi)
        
#         print(f"    Avg Xi: {avg_xi:.4f}")
    
#     return results


# # ============================================================================
# # 6. CREATE SUMMARY TABLES
# # ============================================================================

# def create_summary_tables(results):
#     """Create DataFrames with Xi correlation results"""
    
#     # Table 1: Averaged Xi across all timesteps for each input
#     df_averaged = pd.DataFrame({
#         'Input': results['input_names'],
#         'Xi': results['averaged_xi']
#     })
#     df_averaged = df_averaged.sort_values('Xi', ascending=False)
    
#     # Table 2: Xi per timestep for each input
#     timestep_data = []
#     for input_idx, input_name in enumerate(results['input_names']):
#         xi_values = results['xi_per_timestep'][input_idx]
#         for t in range(12):
#             timestep_data.append({
#                 'Input': input_name,
#                 'Timestep': t + 1,
#                 'Leadtime_min': (t + 1) * 5,
#                 'Xi': xi_values[t] if t < len(xi_values) else np.nan
#             })
    
#     df_timesteps = pd.DataFrame(timestep_data)
    
#     return df_averaged, df_timesteps


# # ============================================================================
# # 7. PLOTLY VISUALIZATIONS
# # ============================================================================

# def create_interactive_heatmap(df_timesteps, metric='Xi'):
#     """Create interactive heatmap of correlations across inputs and timesteps"""
    
#     # Pivot data
#     pivot_data = df_timesteps.pivot(index='Input', columns='Timestep', values=metric)
    
#     fig = go.Figure(data=go.Heatmap(
#         z=pivot_data.values,
#         x=[f'T{i}' for i in pivot_data.columns],
#         y=pivot_data.index,
#         colorscale='RdYlBu',
#         zmid=0.5 if metric == 'Xi' else 0,
#         colorbar=dict(title=metric),
#         text=np.round(pivot_data.values, 3),
#         texttemplate='%{text}',
#         textfont={"size": 8}
#     ))
    
#     fig.update_layout(
#         title=f'Interactive Meteorological Variables Heatmap - {metric} Correlation',
#         xaxis_title='Timestep (5-min intervals)',
#         yaxis_title='Variable',
#         height=800,
#         width=1200
#     )
    
#     return fig


# def create_clustered_bar_chart(df_timesteps, top_n=10):
#     """Create clustered bar chart showing top N inputs across timesteps"""
    
#     # Get top N inputs by average Xi
#     top_inputs = (df_timesteps.groupby('Input')['Xi']
#                   .mean()
#                   .nlargest(top_n)
#                   .index.tolist())
    
#     df_top = df_timesteps[df_timesteps['Input'].isin(top_inputs)]
    
#     fig = go.Figure()
    
#     for input_name in top_inputs:
#         df_input = df_top[df_top['Input'] == input_name]
#         fig.add_trace(go.Bar(
#             x=df_input['Leadtime_min'],
#             y=df_input['Xi'],
#             name=input_name,
#             text=np.round(df_input['Xi'].values, 3),
#             textposition='auto'
#         ))
    
#     fig.update_layout(
#         title=f'Clustered Bar Chart - Meteorological Variables (Top {top_n})',
#         xaxis_title='Lead Time (minutes)',
#         yaxis_title='Xi Correlation',
#         barmode='group',
#         height=600,
#         width=1400,
#         legend=dict(orientation="v", yanchor="top", y=1, xanchor="left", x=1.02)
#     )
    
#     return fig


# def create_box_plots(df_timesteps):
#     """Create box plots showing distribution of correlations per input"""
    
#     fig = go.Figure()
    
#     for input_name in df_timesteps['Input'].unique():
#         df_input = df_timesteps[df_timesteps['Input'] == input_name]
#         fig.add_trace(go.Box(
#             y=df_input['Xi'],
#             name=input_name,
#             boxmean='sd'
#         ))
    
#     fig.update_layout(
#         title='Box Plots - Distribution of Xi Correlation Across 12 Timesteps',
#         yaxis_title='Xi Correlation',
#         xaxis_title='Variable',
#         height=600,
#         width=1600,
#         showlegend=False
#     )
    
#     fig.update_xaxes(tickangle=45)
    
#     return fig


# # ============================================================================
# # 8. COMPLETE WORKFLOW
# # ============================================================================

# def run_complete_analysis(model, input_data_batch, concat_layer_name,
#                          resblock_name, input_names, output_dir='results'):
#     """Run complete analysis and save all outputs"""
    
#     import os
#     os.makedirs(output_dir, exist_ok=True)
    
#     # Run analysis
#     results = analyze_inputs_comprehensive(
#         model, input_data_batch, concat_layer_name,
#         resblock_name, input_names, 
#         num_samples=min(8, input_data_batch[0].shape[0] if isinstance(input_data_batch, list) 
#                        else list(input_data_batch.values())[0].shape[0]),
#         max_samples_memory=4
#     )
    
#     # Create tables
#     df_averaged, df_timesteps = create_summary_tables(results)
    
#     # Save tables
#     df_averaged.to_csv(f'{output_dir}/averaged_correlations.csv', index=False)
#     df_timesteps.to_csv(f'{output_dir}/timestep_correlations.csv', index=False)
    
#     print(f"\n\nSaved tables to {output_dir}/")
#     print("\nTop 10 Inputs by Xi Correlation:")
#     print(df_averaged.head(10))
    
#     # Create visualizations
#     fig1 = create_interactive_heatmap(df_timesteps, metric='Xi')
#     fig1.write_html(f'{output_dir}/heatmap_xi.html')
#     fig1.show()
    
#     fig2 = create_clustered_bar_chart(df_timesteps, top_n=10)
#     fig2.write_html(f'{output_dir}/bar_chart_top10.html')
#     fig2.show()
    
#     fig3 = create_box_plots(df_timesteps)
#     fig3.write_html(f'{output_dir}/box_plots.html')
#     fig3.show()
    
#     print(f"\nSaved interactive plots to {output_dir}/")
    
#     return results, df_averaged, df_timesteps

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from scipy.ndimage import zoom
import gc

# ============================================================================
# 1. XI CORRELATION COEFFICIENT
# ============================================================================

def compute_xi_coefficient(x, y):
    """Compute Xi coefficient between two 1D arrays"""
    n = len(x)
    if n < 2:
        return 0.0
    
    sorted_indices = np.argsort(x)
    y_sorted = y[sorted_indices]
    y_ranks = np.argsort(np.argsort(y_sorted))
    
    numerator = np.sum(np.abs(np.diff(y_ranks)))
    denominator = 2 * n**2 / 3
    
    if denominator == 0:
        return 0.0
    
    xi = 1 - (numerator / denominator)
    return max(0.0, min(1.0, xi))


# ============================================================================
# 2. COMPUTE GRAD-CAM FOR OUTPUT LAYER
# ============================================================================

def compute_output_gradcam(model, input_data_single):
    """
    Compute Grad-CAM for the OUTPUT layer using proper Grad-CAM methodology.
    
    Args:
        model: Keras model
        input_data_single: Single sample input (batch_size=1)
        
    Returns:
        List of Grad-CAM heatmaps for each of 12 output timesteps
    """
    # Get the output layer (last layer in the model)
    output_layer = model.layers[-1]
    print(f"Output layer found: {output_layer.name} (type: {type(output_layer).__name__})")
    
    # Find the ResBlock that feeds into the output layer
    target_layer = None
    for layer in reversed(model.layers[:-1]):
        if 'res_block' in layer.name:
            target_layer = layer
            break
    
    if target_layer is None:
        raise ValueError("Could not find ResBlock before output for Grad-CAM")
    
    print(f"Computing output Grad-CAM from layer: {target_layer.name}")
    
    # Create model that outputs convolution activations and final predictions
    grad_model = Model(
        inputs=model.input,
        outputs=[target_layer.output, output_layer.output]
    )
    
    output_gradcams = []
    
    # Compute Grad-CAM for each of the 12 output timesteps
    for t in range(12):
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(input_data_single)
            
            # Watch the conv outputs
            tape.watch(conv_outputs)
            
            # Target: maximize the mean prediction at this timestep
            if len(predictions.shape) == 5:  # (batch, time, h, w, channels)
                target_output = predictions[0, t, :, :, :]  # Shape: (h, w, c)
            else:  # (batch, h, w, channels)
                target_output = predictions[0, :, :, :]
            
            # Loss: mean activation (we want to see what causes high predictions)
            loss = tf.reduce_mean(target_output)
        
        # Compute gradients of loss w.r.t. conv_outputs
        grads = tape.gradient(loss, conv_outputs)
        
        if grads is None:
            print(f"WARNING: No gradients for output timestep {t}")
            output_gradcams.append(np.zeros((256, 256)))
            continue
        
        # Standard Grad-CAM: Global average pooling of gradients
        if len(grads.shape) == 5:  # (batch, time, h, w, channels)
            # Get the corresponding timestep
            if t < grads.shape[1]:
                grads_t = grads[0, t, :, :, :]  # (h, w, channels)
                conv_t = conv_outputs[0, t, :, :, :]
            else:
                # Use last available timestep
                grads_t = grads[0, -1, :, :, :]
                conv_t = conv_outputs[0, -1, :, :, :]
            
            # Compute importance weights per channel
            weights = tf.reduce_mean(grads_t, axis=(0, 1))  # (channels,)
            
        else:  # (batch, h, w, channels)
            grads_t = grads[0, :, :, :]
            conv_t = conv_outputs[0, :, :, :]
            weights = tf.reduce_mean(grads_t, axis=(0, 1))
        
        # Weighted combination of forward activation maps
        cam = tf.zeros(conv_t.shape[:-1])  # (h, w)
        for i, w in enumerate(weights):
            cam += w * conv_t[:, :, i]
        
        # Apply ReLU (only positive influences)
        cam = tf.nn.relu(cam)
        
        # Normalize to [0, 1]
        cam_max = tf.reduce_max(cam)
        if cam_max > 1e-8:
            cam = cam / cam_max
        
        output_gradcams.append(cam.numpy())
    
    return output_gradcams


# ============================================================================
# 3. COMPUTE GRAD-CAM FOR INPUT LAYER
# ============================================================================

def compute_input_gradcam(model, input_data_single, concat_layer_name, 
                         resblock_name, input_channel_range):
    """
    Compute Grad-CAM for a specific input channel range using proper Grad-CAM methodology.
    
    Args:
        model: Keras model
        input_data_single: Single sample input (batch_size=1)
        concat_layer_name: Concatenation layer name
        resblock_name: ResBlock layer name
        input_channel_range: (start, end) channel indices for this input
        
    Returns:
        Grad-CAM heatmap (2D array)
    """
    start_ch, end_ch = input_channel_range
    
    # Get layers
    concat_layer = model.get_layer(concat_layer_name)
    resblock_layer = model.get_layer(resblock_name)
    
    # Create model that outputs concat features, resblock features, and final predictions
    grad_model = Model(
        inputs=model.input,
        outputs=[concat_layer.output, resblock_layer.output, model.output]
    )
    
    with tf.GradientTape() as tape:
        concat_output, resblock_output, predictions = grad_model(input_data_single)
        
        # Watch the ResBlock output (the convolutional features we want Grad-CAM for)
        tape.watch(resblock_output)
        
        # Loss: maximize mean prediction across all outputs
        loss = tf.reduce_mean(predictions)
    
    # Get gradients w.r.t ResBlock output
    grads = tape.gradient(loss, resblock_output)
    
    if grads is None:
        print(f"WARNING: No gradients for channels {start_ch}:{end_ch}")
        return np.zeros((128, 128))
    
    # Standard Grad-CAM formula:
    # 1. Global average pooling of gradients to get importance weights
    # 2. Weighted combination of feature maps
    # 3. ReLU
    # 4. Normalize
    
    if len(grads.shape) == 5:  # (batch, time, h, w, channels)
        # Average across time dimension
        grads_avg = tf.reduce_mean(grads[0], axis=0)  # (h, w, channels)
        conv_avg = tf.reduce_mean(resblock_output[0], axis=0)  # (h, w, channels)
    else:  # (batch, h, w, channels)
        grads_avg = grads[0]
        conv_avg = resblock_output[0]
    
    # Extract only the channels corresponding to this input
    grads_input = grads_avg[:, :, start_ch:end_ch]  # (h, w, input_channels)
    conv_input = conv_avg[:, :, start_ch:end_ch]
    
    # Compute importance weights: global average pooling of gradients
    weights = tf.reduce_mean(grads_input, axis=(0, 1))  # (input_channels,)
    
    # Weighted combination of feature maps
    cam = tf.zeros(conv_input.shape[:-1])  # (h, w)
    for i, w in enumerate(weights):
        cam += w * conv_input[:, :, i]
    
    # Apply ReLU
    cam = tf.nn.relu(cam)
    
    # Normalize
    cam_max = tf.reduce_max(cam)
    if cam_max > 1e-8:
        cam = cam / cam_max
    
    return cam.numpy()


# ============================================================================
# 4. BATCH PROCESSING
# ============================================================================

def process_batch_samples(model, input_data_batch, concat_layer_name,
                         resblock_name, input_channel_ranges, num_samples=4):
    """
    Process multiple samples and average Grad-CAM results.
    Also returns raw data for visualization.
    """
    # Determine batch size
    if isinstance(input_data_batch, dict):
        batch_size = list(input_data_batch.values())[0].shape[0]
    elif isinstance(input_data_batch, list):
        batch_size = input_data_batch[0].shape[0]
    else:
        batch_size = input_data_batch.shape[0]
    
    num_samples = min(num_samples, batch_size)
    print(f"Processing {num_samples} samples...")
    
    all_input_gradcams = []  # [sample][input]
    all_output_gradcams = []  # [sample][timestep]
    
    # Store first sample's data for visualization
    sample_data = None
    sample_predictions = None
    
    for sample_idx in range(num_samples):
        print(f"  Sample {sample_idx + 1}/{num_samples}...", end='\r')
        
        # Extract single sample
        if isinstance(input_data_batch, dict):
            single_sample = {k: v[sample_idx:sample_idx+1] for k, v in input_data_batch.items()}
        elif isinstance(input_data_batch, list):
            single_sample = [x[sample_idx:sample_idx+1] for x in input_data_batch]
        else:
            single_sample = input_data_batch[sample_idx:sample_idx+1]
        
        # Store first sample for visualization
        if sample_idx == 0:
            sample_data = single_sample
            # Get predictions for visualization
            sample_predictions = model.predict(single_sample, verbose=0)
        
        # Compute input Grad-CAMs
        input_gradcams = []
        for input_idx, ch_range in enumerate(input_channel_ranges):
            gradcam = compute_input_gradcam(
                model, single_sample, concat_layer_name, 
                resblock_name, ch_range
            )
            input_gradcams.append(gradcam)
        
        all_input_gradcams.append(input_gradcams)
        
        # Compute output Grad-CAMs (12 timesteps)
        output_gradcams = compute_output_gradcam(model, single_sample)
        all_output_gradcams.append(output_gradcams)
        
        # Clear memory
        tf.keras.backend.clear_session()
        gc.collect()
    
    print(f"\n  Completed {num_samples} samples")
    
    # Average across samples
    num_inputs = len(all_input_gradcams[0])
    averaged_input_gradcams = []
    for input_idx in range(num_inputs):
        stacked = np.stack([all_input_gradcams[s][input_idx] for s in range(num_samples)])
        averaged_input_gradcams.append(np.mean(stacked, axis=0))
    
    # Average output Grad-CAMs across samples
    averaged_output_gradcams = []
    for t in range(12):
        stacked = np.stack([all_output_gradcams[s][t] for s in range(num_samples)])
        averaged_output_gradcams.append(np.mean(stacked, axis=0))
    
    return averaged_input_gradcams, averaged_output_gradcams, sample_data, sample_predictions


# ============================================================================
# 5. COMPUTE XI CORRELATIONS BETWEEN INPUT AND OUTPUT GRAD-CAMS
# ============================================================================

def compute_xi_correlations(input_gradcam, output_gradcams):
    """
    Compute Xi correlation between input Grad-CAM and each output Grad-CAM.
    
    Args:
        input_gradcam: 2D array - Grad-CAM from input
        output_gradcams: List of 2D arrays - Grad-CAMs from output (12 timesteps)
        
    Returns:
        List of Xi values, one per timestep
    """
    xi_values = []
    
    for t, output_gradcam in enumerate(output_gradcams):
        # Resize to same resolution
        if input_gradcam.shape != output_gradcam.shape:
            scale_factors = (output_gradcam.shape[0] / input_gradcam.shape[0],
                           output_gradcam.shape[1] / input_gradcam.shape[1])
            input_resized = zoom(input_gradcam, scale_factors, order=1)
        else:
            input_resized = input_gradcam
        
        # Flatten
        x = input_resized.flatten()
        y = output_gradcam.flatten()
        
        # Remove NaN/Inf
        mask = np.isfinite(x) & np.isfinite(y)
        x = x[mask]
        y = y[mask]
        
        if len(x) < 2 or np.std(x) < 1e-10 or np.std(y) < 1e-10:
            xi_values.append(np.nan)
            continue
        
        # Compute Xi
        xi = compute_xi_coefficient(x, y)
        xi_values.append(xi)
    
    return xi_values


# ============================================================================
# 6. MAIN ANALYSIS FUNCTION
# ============================================================================

def analyze_inputs_comprehensive(model, input_data_batch, concat_layer_name,
                                 resblock_name, input_names, num_samples=4):
    """
    Complete analysis: compute input Grad-CAMs, output Grad-CAMs, and Xi correlations.
    """
    # Determine input channel ranges
    concat_layer = model.get_layer(concat_layer_name)
    total_channels = concat_layer.output.shape[-1]
    num_inputs = len(input_names)
    
    channels_per_input = total_channels // num_inputs
    input_channel_ranges = [(i * channels_per_input, 
                            (i + 1) * channels_per_input if i < num_inputs - 1 else total_channels)
                           for i in range(num_inputs)]
    
    print(f"\nTotal channels: {total_channels}")
    print(f"Inputs: {num_inputs}")
    print(f"Channels per input: ~{channels_per_input}")
    
    # Process batch samples
    averaged_input_gradcams, averaged_output_gradcams, sample_data, sample_predictions = process_batch_samples(
        model, input_data_batch, concat_layer_name, resblock_name,
        input_channel_ranges, num_samples=num_samples
    )
    
    # Compute Xi correlations
    print("\nComputing Xi correlations between input and output Grad-CAMs...")
    
    results = {
        'input_names': input_names,
        'input_gradcams': averaged_input_gradcams,
        'output_gradcams': averaged_output_gradcams,
        'xi_per_timestep': [],
        'averaged_xi': [],
        'sample_data': sample_data,  # Store for visualization
        'sample_predictions': sample_predictions  # Store for visualization
    }
    
    for input_idx, input_name in enumerate(input_names):
        print(f"  {input_name}...", end=' ')
        
        input_gradcam = averaged_input_gradcams[input_idx]
        xi_values = compute_xi_correlations(input_gradcam, averaged_output_gradcams)
        
        results['xi_per_timestep'].append(xi_values)
        avg_xi = np.nanmean(xi_values)
        results['averaged_xi'].append(avg_xi)
        
        print(f"Xi: {avg_xi:.4f}")
    
    return results


# ============================================================================
# 7. CREATE SUMMARY TABLES
# ============================================================================

def create_summary_tables(results):
    """Create DataFrames with Xi correlation results"""
    
    df_averaged = pd.DataFrame({
        'Input': results['input_names'],
        'Xi': results['averaged_xi']
    })
    df_averaged = df_averaged.sort_values('Xi', ascending=False)
    
    timestep_data = []
    for input_idx, input_name in enumerate(results['input_names']):
        xi_values = results['xi_per_timestep'][input_idx]
        for t in range(12):
            timestep_data.append({
                'Input': input_name,
                'Timestep': t + 1,
                'Leadtime_min': (t + 1) * 5,
                'Xi': xi_values[t] if t < len(xi_values) else np.nan
            })
    
    df_timesteps = pd.DataFrame(timestep_data)
    
    return df_averaged, df_timesteps


# ============================================================================
# 8. VISUALIZATIONS
# ============================================================================

def create_interactive_heatmap(df_timesteps):
    """Interactive heatmap of Xi correlations"""
    pivot_data = df_timesteps.pivot(index='Input', columns='Timestep', values='Xi')
    
    fig = go.Figure(data=go.Heatmap(
        z=pivot_data.values,
        x=[f'T{i}' for i in pivot_data.columns],
        y=pivot_data.index,
        colorscale='RdYlBu',
        zmid=0.5,
        zmin=0,
        zmax=1,
        colorbar=dict(title='Xi'),
        text=np.round(pivot_data.values, 3),
        texttemplate='%{text}',
        textfont={"size": 8}
    ))
    
    fig.update_layout(
        title='Xi Correlation: Input Grad-CAM vs Output Grad-CAM',
        xaxis_title='Output Timestep (5-min intervals)',
        yaxis_title='Input Variable',
        height=800,
        width=1200
    )
    
    return fig


def create_clustered_bar_chart(df_timesteps, top_n=10):
    """Create clustered bar chart showing top N inputs across timesteps"""
    
    # Get top N inputs by average Xi
    top_inputs = (df_timesteps.groupby('Input')['Xi']
                  .mean()
                  .nlargest(top_n)
                  .index.tolist())
    
    df_top = df_timesteps[df_timesteps['Input'].isin(top_inputs)]
    
    fig = go.Figure()
    
    for input_name in top_inputs:
        df_input = df_top[df_top['Input'] == input_name]
        fig.add_trace(go.Bar(
            x=df_input['Leadtime_min'],
            y=df_input['Xi'],
            name=input_name,
            text=np.round(df_input['Xi'].values, 3),
            textposition='auto'
        ))
    
    fig.update_layout(
        title=f'Clustered Bar Chart - Meteorological Variables (Top {top_n})',
        xaxis_title='Lead Time (minutes)',
        yaxis_title='Xi Correlation (Input vs Output Grad-CAM)',
        barmode='group',
        height=600,
        width=1400,
        legend=dict(orientation="v", yanchor="top", y=1, xanchor="left", x=1.02)
    )
    
    return fig


def create_box_plots(df_timesteps):
    """Create box plots showing distribution of Xi correlations per input"""
    
    fig = go.Figure()
    
    for input_name in df_timesteps['Input'].unique():
        df_input = df_timesteps[df_timesteps['Input'] == input_name]
        fig.add_trace(go.Box(
            y=df_input['Xi'],
            name=input_name,
            boxmean='sd'
        ))
    
    fig.update_layout(
        title='Box Plots - Distribution of Xi Correlation Across 12 Output Timesteps',
        yaxis_title='Xi Correlation (Input vs Output Grad-CAM)',
        xaxis_title='Variable',
        height=600,
        width=1600,
        showlegend=False
    )
    
    fig.update_xaxes(tickangle=45)
    
    return fig


def create_gradcam_comparison_plot(results, input_idx, output_timestep, input_data_single, predictions):
    """
    Visualize: raw input, input Grad-CAM, raw output, output Grad-CAM, overlay.
    
    Args:
        results: Results dictionary
        input_idx: Index of the input to visualize
        output_timestep: Which output timestep to show (0-11)
        input_data_single: The actual input data (for showing raw inputs)
        predictions: The actual model predictions (for showing raw outputs)
    """
    input_name = results['input_names'][input_idx]
    input_gradcam = results['input_gradcams'][input_idx]
    output_gradcam = results['output_gradcams'][output_timestep]
    xi_value = results['xi_per_timestep'][input_idx][output_timestep]
    
    # Extract raw input data - this is tricky since we need to find the right input
    # We'll try to get it from the input data
    if isinstance(input_data_single, dict) and input_name in input_data_single:
        raw_input = input_data_single[input_name][0]  # (time, h, w, c)
        if len(raw_input.shape) == 4:
            # Average across time and take first channel
            raw_input = np.mean(raw_input, axis=0)[:, :, 0]
        else:
            raw_input = raw_input[:, :, 0]
    elif isinstance(input_data_single, list):
        # Try to find by index
        if input_idx < len(input_data_single):
            raw_input = input_data_single[input_idx][0]
            if len(raw_input.shape) == 4:
                raw_input = np.mean(raw_input, axis=0)[:, :, 0]
            else:
                raw_input = raw_input[:, :, 0]
        else:
            raw_input = None
    else:
        raw_input = None
    
    # Extract raw output
    if len(predictions.shape) == 5:  # (batch, time, h, w, c)
        raw_output = predictions[0, output_timestep, :, :, 0]
    elif len(predictions.shape) == 4:  # (batch, h, w, c)
        raw_output = predictions[0, :, :, 0]
    else:
        raw_output = None
    
    # Create figure with 5 subplots
    fig, axes = plt.subplots(1, 5, figsize=(25, 5))
    
    # 1. Raw input
    if raw_input is not None:
        im0 = axes[0].imshow(raw_input, cmap='viridis')
        axes[0].set_title(f'Raw Input\n{input_name}', fontsize=12, fontweight='bold')
        axes[0].axis('off')
        plt.colorbar(im0, ax=axes[0], fraction=0.046)
    else:
        axes[0].text(0.5, 0.5, 'Raw Input\nNot Available', 
                    ha='center', va='center', transform=axes[0].transAxes)
        axes[0].axis('off')
    
    # 2. Input Grad-CAM
    im1 = axes[1].imshow(input_gradcam, cmap='jet', vmin=0, vmax=1)
    axes[1].set_title(f'Input Grad-CAM\n{input_name}', fontsize=12, fontweight='bold')
    axes[1].axis('off')
    plt.colorbar(im1, ax=axes[1], fraction=0.046, label='Attention')
    
    # 3. Raw output
    if raw_output is not None:
        im2 = axes[2].imshow(raw_output, cmap='viridis')
        axes[2].set_title(f'Raw Output\nTimestep {output_timestep+1} ({(output_timestep+1)*5} min)', 
                         fontsize=12, fontweight='bold')
        axes[2].axis('off')
        plt.colorbar(im2, ax=axes[2], fraction=0.046)
    else:
        axes[2].text(0.5, 0.5, 'Raw Output\nNot Available', 
                    ha='center', va='center', transform=axes[2].transAxes)
        axes[2].axis('off')
    
    # 4. Output Grad-CAM
    im3 = axes[3].imshow(output_gradcam, cmap='jet', vmin=0, vmax=1)
    axes[3].set_title(f'Output Grad-CAM\nTimestep {output_timestep+1}', 
                     fontsize=12, fontweight='bold')
    axes[3].axis('off')
    plt.colorbar(im3, ax=axes[3], fraction=0.046, label='Attention')
    
    # 5. Overlay comparison
    # Resize input Grad-CAM to match output resolution
    if input_gradcam.shape != output_gradcam.shape:
        from scipy.ndimage import zoom
        scale_factors = (output_gradcam.shape[0] / input_gradcam.shape[0],
                        output_gradcam.shape[1] / input_gradcam.shape[1])
        input_gradcam_resized = zoom(input_gradcam, scale_factors, order=1)
    else:
        input_gradcam_resized = input_gradcam
    
    axes[4].imshow(input_gradcam_resized, cmap='Blues', alpha=0.6, vmin=0, vmax=1)
    axes[4].imshow(output_gradcam, cmap='Reds', alpha=0.6, vmin=0, vmax=1)
    axes[4].set_title(f'Overlay\n(Blue=Input, Red=Output)\nXi: {xi_value:.4f}', 
                     fontsize=12, fontweight='bold')
    axes[4].axis('off')
    
    plt.suptitle(f'Grad-CAM Analysis: {input_name} vs Output at {(output_timestep+1)*5} min', 
                fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    return fig


# ============================================================================
# 9. COMPLETE WORKFLOW
# ============================================================================

def run_complete_analysis(model, input_data_batch, concat_layer_name,
                         resblock_name, input_names, output_dir='results'):
    """Run complete analysis and save outputs"""
    
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    # Run analysis
    results = analyze_inputs_comprehensive(
        model, input_data_batch, concat_layer_name,
        resblock_name, input_names, num_samples=4
    )
    
    # Create tables
    df_averaged, df_timesteps = create_summary_tables(results)
    
    # Save tables
    df_averaged.to_csv(f'{output_dir}/averaged_xi.csv', index=False)
    df_timesteps.to_csv(f'{output_dir}/timestep_xi.csv', index=False)
    
    print(f"\n{'='*70}")
    print(f"RESULTS FOR {resblock_name}")
    print(f"{'='*70}")
    print(f"\nTop 10 Inputs by Xi Correlation:")
    print(df_averaged.head(10).to_string(index=False))
    
    # Create interactive heatmap
    fig_heatmap = create_interactive_heatmap(df_timesteps)
    fig_heatmap.write_html(f'{output_dir}/heatmap_xi.html')
    print(f"\nSaved: {output_dir}/heatmap_xi.html")
    
    # Create clustered bar chart
    fig_bar = create_clustered_bar_chart(df_timesteps, top_n=10)
    fig_bar.write_html(f'{output_dir}/bar_chart_top10.html')
    print(f"Saved: {output_dir}/bar_chart_top10.html")
    
    # Create box plots
    fig_box = create_box_plots(df_timesteps)
    fig_box.write_html(f'{output_dir}/box_plots.html')
    print(f"Saved: {output_dir}/box_plots.html")
    
    # Save Grad-CAM comparison plots for top 5 inputs
    print(f"\nGenerating Grad-CAM comparison plots (with raw inputs and outputs)...")
    
    # Get sample data and predictions for visualization
    sample_data = results['sample_data']
    sample_predictions = results['sample_predictions']
    
    for rank in range(min(5, len(input_names))):
        input_name = df_averaged.iloc[rank]['Input']
        actual_idx = input_names.index(input_name)
        
        # Create comparison for multiple timesteps
        for timestep in [2, 5, 8, 11]:  # 15min, 30min, 45min, 60min
            fig = create_gradcam_comparison_plot(
                results, actual_idx, timestep, 
                sample_data, sample_predictions
            )
            fig.savefig(f'{output_dir}/gradcam_comparison_rank{rank+1}_t{timestep+1}.png', 
                       dpi=150, bbox_inches='tight')
            plt.close(fig)
    
    print(f"Saved Grad-CAM comparison plots to {output_dir}/")
    print(f"  - Includes: raw input, input Grad-CAM, raw output, output Grad-CAM, overlay")
    
    return results, df_averaged, df_timesteps
