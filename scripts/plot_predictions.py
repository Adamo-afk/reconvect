import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import ot
from netCDF4 import Dataset
from matplotlib.colors import ListedColormap, BoundaryNorm
from pathlib import Path
import matplotlib.animation as animation
from matplotlib.gridspec import GridSpec
from datetime import datetime, timedelta
import os


def visualize_lightning_predictions(predictions, targets, sample_idx=0, save_dir='lightning_plots', 
                                   figsize=(20, 12), dpi=150, show_difference=True):
    """
    Visualize lightning nowcasting predictions vs targets with temporal alignment.
    
    Parameters:
    -----------
    predictions : numpy.ndarray
        Shape (batch, 12, 256, 256, 1) - Predicted lightning occurrence
    targets : numpy.ndarray  
        Shape (batch, 12, 256, 256, 1) - Target lightning occurrence
    sample_idx : int
        Which sample from batch to visualize (0-7)
    save_dir : str
        Directory to save plots
    figsize : tuple
        Figure size
    dpi : int
        Plot resolution
    show_difference : bool
        Whether to show difference maps
    """
    
    # Create save directory
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    # Extract single sample
    pred_sample = predictions[sample_idx, :, :, :, 0]  # (12, 256, 256)
    target_sample = targets[sample_idx, :, :, :, 0]    # (12, 256, 256)

    # Binary target and prediction, also reversed items for better visualization
    pred_sample = list(map(lambda x: (x > 0.5).astype(int), pred_sample))[::-1]
    target_sample = list(map(lambda x: (x > 0.5).astype(int), target_sample))[::-1]
    print(f"Pred binary shape {np.array(pred_sample).shape}, Target binary shape {np.array(target_sample).shape}")
    print(f"Pred binary unique values {np.unique(np.array(pred_sample))}, Target binary unique values {np.unique(np.array(target_sample))}")

    # Time labels (5-minute intervals)
    # time_labels = [f't+{(i+1)*5}min' for i in range(12)]
    time_labels = [f't+{(12-i)*5}min' for i in range(12)]
    
    # Create comprehensive visualization
    fig = plt.figure(figsize=figsize)
    
    if show_difference:
        # 4 rows: Targets, Predictions, Difference, Metrics
        gs = GridSpec(4, 13, figure=fig, height_ratios=[1, 1, 1, 0.8], 
                     width_ratios=[1]*12 + [0.1])  # Extra column for colorbar
    else:
        # 3 rows: Targets, Predictions, Metrics
        gs = GridSpec(3, 13, figure=fig, height_ratios=[1, 1, 0.8], 
                     width_ratios=[1]*12 + [0.1])
    
    # Lightning colormap (white for no lightning, red/yellow for lightning)
    colors = ['white', 'red']
    n_bins = len(colors)
    cmap = ListedColormap(colors)
    norm = BoundaryNorm(np.linspace(0, 1, n_bins), cmap.N)
    
    # Plot targets (row 1)
    for t in range(12):
        ax = fig.add_subplot(gs[0, t])
        im = ax.imshow(target_sample[t], cmap=cmap, norm=norm, aspect='equal')
        ax.set_title(f'Target\n{time_labels[t]}', fontsize=10, fontweight='bold')
        ax.set_xticks([])
        ax.set_yticks([])
        if t == 0:
            ax.set_ylabel('Ground Truth', fontsize=12, fontweight='bold')
    
    # Plot predictions (row 2) 
    for t in range(12):
        ax = fig.add_subplot(gs[1, t])
        im = ax.imshow(pred_sample[t], cmap=cmap, norm=norm, aspect='equal')
        ax.set_title(f'Predicted\n{time_labels[t]}', fontsize=10, fontweight='bold')
        ax.set_xticks([])
        ax.set_yticks([])
        if t == 0:
            ax.set_ylabel('Predictions', fontsize=12, fontweight='bold')
    
    # Add colorbar
    cbar_ax = fig.add_subplot(gs[0:2, 12])
    cbar = plt.colorbar(im, cax=cbar_ax)
    cbar.set_label('Lightning Probability', fontsize=12)
    cbar.set_ticks([0, 0.33, 0.66, 1.0])
    cbar.set_ticklabels(['No Lightning', 'Low', 'Medium', 'High'])
    
    if show_difference:
        # Plot differences (row 3)
        diff_cmap = plt.cm.RdBu_r
        for t in range(12):
            ax = fig.add_subplot(gs[2, t])
            diff = pred_sample[t] - target_sample[t]
            im_diff = ax.imshow(diff, cmap=diff_cmap, vmin=-1, vmax=1, aspect='equal')
            ax.set_title(f'Diff\n{time_labels[t]}', fontsize=10, fontweight='bold')
            ax.set_xticks([])
            ax.set_yticks([])
            if t == 0:
                ax.set_ylabel('Prediction - Target', fontsize=12, fontweight='bold')
        
        # Add difference colorbar
        diff_cbar_ax = fig.add_subplot(gs[2, 12])
        diff_cbar = plt.colorbar(im_diff, cax=diff_cbar_ax)
        diff_cbar.set_label('Difference', fontsize=12)
        
        metrics_row = 3
    else:
        metrics_row = 2
    
    # Plot temporal metrics (bottom row)
    metrics_ax = fig.add_subplot(gs[metrics_row, :12])
    
    # Calculate metrics over time
    iou_scores = []
    accuracies = []
    lightning_coverage_pred = []
    lightning_coverage_target = []

    # Convert to numpy arrays
    pred_sample = np.array(pred_sample)
    target_sample = np.array(target_sample)
    
    for t in range(12):
        # # Binary predictions (threshold at 0.5)
        # pred_binary = (pred_sample[t] > 0.5).astype(int)
        # target_binary = (target_sample[t] > 0.5).astype(int)
        
        # IoU calculation
        intersection = np.sum(pred_sample * target_sample)
        union = np.sum((pred_sample + target_sample) > 0)
        iou = intersection / (union + 1e-8)
        iou_scores.append(iou)
        
        # Accuracy
        accuracy = np.mean(pred_sample == target_sample)
        accuracies.append(accuracy)
        
        # Lightning coverage (percentage of pixels with lightning)
        lightning_coverage_pred.append(np.mean(pred_sample) * 100)
        lightning_coverage_target.append(np.mean(target_sample) * 100)
    
    # Plot metrics
    time_minutes = np.arange(0, 60, 5)
    metrics_ax.plot(time_minutes, iou_scores, 'b-o', linewidth=2, markersize=6, label='IoU Score')
    metrics_ax.plot(time_minutes, accuracies, 'g-s', linewidth=2, markersize=6, label='Accuracy')
    
    # Secondary y-axis for coverage
    ax2 = metrics_ax.twinx()
    ax2.plot(time_minutes, lightning_coverage_pred, 'r--^', linewidth=2, markersize=6, label='Predicted Coverage %')
    ax2.plot(time_minutes, lightning_coverage_target, 'orange', linestyle='--', marker='v', 
             linewidth=2, markersize=6, label='Target Coverage %')
    
    # Format metrics plot
    metrics_ax.set_xlabel('Forecast Time (minutes)', fontsize=12)
    metrics_ax.set_ylabel('IoU Score / Accuracy', fontsize=12, color='blue')
    metrics_ax.tick_params(axis='y', labelcolor='blue')
    metrics_ax.set_xlim(0, 55)
    metrics_ax.set_ylim(0, 1)
    metrics_ax.grid(True, alpha=0.3)
    
    ax2.set_ylabel('Lightning Coverage (%)', fontsize=12, color='red')
    ax2.tick_params(axis='y', labelcolor='red')
    
    # Combined legend
    lines1, labels1 = metrics_ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    metrics_ax.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=10)
    
    plt.suptitle(f'Lightning Nowcasting: Sample {sample_idx} - Temporal Evolution (1 Hour Forecast)', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    # Save plot
    plot_path = save_path / f'lightning_nowcasting_sample_{sample_idx}.png'
    plt.savefig(plot_path, dpi=dpi, bbox_inches='tight')
    print(f"Saved comprehensive plot: {plot_path}")
    plt.close()

def create_temporal_summary_plot(predictions, targets, save_dir='lightning_plots', figsize=(15, 10)):
    """
    Create summary plots showing performance across all samples and timesteps.
    """
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    batch_size, timesteps = predictions.shape[:2]
    
    # Calculate metrics for all samples and timesteps
    all_iou = np.zeros((batch_size, timesteps))
    all_accuracy = np.zeros((batch_size, timesteps))
    all_lightning_pred = np.zeros((batch_size, timesteps))
    all_lightning_target = np.zeros((batch_size, timesteps))
    
    for b in range(batch_size):
        for t in range(timesteps):
            pred_binary = (predictions[b, t, :, :, 0] > 0.5).astype(int)
            target_binary = (targets[b, t, :, :, 0] > 0.5).astype(int)
            
            # IoU
            intersection = np.sum(pred_binary * target_binary)
            union = np.sum((pred_binary + target_binary) > 0)
            all_iou[b, t] = intersection / (union + 1e-8)
            
            # Accuracy
            all_accuracy[b, t] = np.mean(pred_binary == target_binary)
            
            # Lightning coverage
            all_lightning_pred[b, t] = np.mean(pred_binary) * 100
            all_lightning_target[b, t] = np.mean(target_binary) * 100
    
    # Create summary plots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=figsize)
    
    time_minutes = np.arange(0, 60, 5)
    
    # IoU over time
    mean_iou = np.mean(all_iou, axis=0)
    std_iou = np.std(all_iou, axis=0)
    ax1.plot(time_minutes, mean_iou, 'b-o', linewidth=2, markersize=6)
    ax1.fill_between(time_minutes, mean_iou - std_iou, mean_iou + std_iou, alpha=0.3)
    ax1.set_title('IoU Score Over Time', fontweight='bold')
    ax1.set_xlabel('Forecast Time (minutes)')
    ax1.set_ylabel('IoU Score')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 1)
    
    # Accuracy over time
    mean_acc = np.mean(all_accuracy, axis=0)
    std_acc = np.std(all_accuracy, axis=0)
    ax2.plot(time_minutes, mean_acc, 'g-s', linewidth=2, markersize=6)
    ax2.fill_between(time_minutes, mean_acc - std_acc, mean_acc + std_acc, alpha=0.3)
    ax2.set_title('Accuracy Over Time', fontweight='bold')
    ax2.set_xlabel('Forecast Time (minutes)')
    ax2.set_ylabel('Accuracy')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1)
    
    # Lightning coverage comparison
    mean_pred_cov = np.mean(all_lightning_pred, axis=0)
    mean_target_cov = np.mean(all_lightning_target, axis=0)
    ax3.plot(time_minutes, mean_pred_cov, 'r--^', linewidth=2, markersize=6, label='Predicted')
    ax3.plot(time_minutes, mean_target_cov, 'orange', linestyle='--', marker='v', 
             linewidth=2, markersize=6, label='Target')
    ax3.set_title('Lightning Coverage Comparison', fontweight='bold')
    ax3.set_xlabel('Forecast Time (minutes)')
    ax3.set_ylabel('Lightning Coverage (%)')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
    # Performance heatmap across samples
    im = ax4.imshow(all_iou, aspect='equal', cmap='viridis', vmin=0, vmax=1)
    ax4.set_title('IoU Score Heatmap (Samples vs Time)', fontweight='bold')
    ax4.set_xlabel('Timestep')
    ax4.set_ylabel('Sample')
    ax4.set_xticks(range(0, 12, 2))
    ax4.set_xticklabels([f't+{i*5}min' for i in range(0, 12, 2)])
    plt.colorbar(im, ax=ax4, label='IoU Score')
    
    plt.tight_layout()
    
    summary_path = save_path / 'lightning_summary_analysis.png'
    plt.savefig(summary_path, dpi=150, bbox_inches='tight')
    print(f"Saved summary analysis: {summary_path}")
    plt.close()

def create_animation(predictions, targets, sample_idx=0, save_dir='lightning_plots', 
                    interval=500, figsize=(12, 6)):
    """
    Create animated visualization showing temporal evolution.
    """
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    pred_sample = predictions[sample_idx, :, :, :, 0]
    target_sample = targets[sample_idx, :, :, :, 0]

    # Binary target and prediction, also reversed items for better visualization
    pred_sample = list(map(lambda x: (x > 0.5).astype(int), pred_sample))[::-1]
    target_sample = list(map(lambda x: (x > 0.5).astype(int), target_sample))[::-1]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Lightning colormap
    colors = ['white', 'red']
    cmap = ListedColormap(colors)
    norm = BoundaryNorm(np.linspace(0, 1, len(colors)), cmap.N)
    
    # Initial plots
    im1 = ax1.imshow(target_sample[0], cmap=cmap, norm=norm)
    ax1.set_xticks([])
    ax1.set_yticks([])
    
    im2 = ax2.imshow(pred_sample[0], cmap=cmap, norm=norm)
    ax2.set_xticks([])
    ax2.set_yticks([])
    
    # Add colorbar
    cbar = plt.colorbar(im1, ax=[ax1, ax2], fraction=0.046, pad=0.04)
    cbar.set_label('Lightning Probability')
    
    # time_text = fig.suptitle('t+0min', fontsize=14, fontweight='bold')
    # Time labels (5-minute intervals)
    # time_labels = [f't+{(i+1)*5}min' for i in range(12)]
    time_labels = [f't+{(12-i)*5}min' for i in range(12)]
    
    def animate(frame):
        im1.set_array(target_sample[frame])
        ax1.set_title(f'Ground Truth - {time_labels[frame]}')
        im2.set_array(pred_sample[frame])
        ax2.set_title(f'Predictions - {time_labels[frame]}')
        # time_text.set_text(f't+{frame*5}min')
        return [im1, im2]
    
    anim = animation.FuncAnimation(fig, animate, frames=12, interval=interval, 
                                 blit=True, repeat=True)
    
    # Save animation
    anim_path = save_path / f'lightning_animation_sample_{sample_idx}.gif'
    anim.save(anim_path, writer='pillow', fps=2)
    print(f"Saved animation: {anim_path}")
    plt.close()


def visualize_lightning_nowcasting_complete(predictions, targets, save_dir='lightning_plots'):
    """
    Complete visualization pipeline for lightning nowcasting results.
    """
    print("Creating comprehensive lightning nowcasting visualizations...")
    
    # Visualize all samples
    for sample_idx in range(predictions.shape[0]):
        visualize_lightning_predictions(predictions, targets, sample_idx=sample_idx, 
                                       save_dir=save_dir, show_difference=True)
        
        # Create animation for first sample only (to save space)
        create_animation(predictions, targets, sample_idx=sample_idx, save_dir=save_dir)
    
    # Create summary analysis
    create_temporal_summary_plot(predictions, targets, save_dir=save_dir)
    
    print(f"\nAll visualizations saved to: {Path(save_dir).absolute()}")


# def plot_encoder_decoder_sample(data_inputs, target, sample_idx=0, 
#                                   input_names=None, figsize=None, 
#                                   cmap='viridis', vmin=None, vmax=None):
#     """
#     Plot all input product maps and target maps for a specific sample.
    
#     Parameters:
#     -----------
#     data_inputs : list of np.ndarray
#         List of N input arrays, each with shape (batch, time, height, width, channels)
#     target : np.ndarray
#         Target array with shape (batch, time, height, width, channels)
#     sample_idx : int
#         Index of the sample to plot from the batch (default: 0)
#     input_names : list of str, optional
#         Names for each input product (default: "Input 0", "Input 1", etc.)
#     figsize : tuple, optional
#         Figure size (width, height). Auto-calculated if None
#     cmap : str
#         Colormap for the plots (default: 'viridis')
#     vmin, vmax : float, optional
#         Min and max values for colorbar normalization
    
#     Returns:
#     --------
#     fig, axes : matplotlib figure and axes objects
#     """
    
#     n_inputs = len(data_inputs)
    
#     # Generate default input names if not provided
#     if input_names is None:
#         input_names = [f"Input {i}" for i in range(n_inputs)]
    
#     # Extract the specific sample and average channels if needed
#     processed_inputs = []
#     n_timesteps_per_input = []
    
#     for data in data_inputs:
#         # Extract sample
#         sample_data = data[sample_idx]  # Shape: (time, height, width, channels)
        
#         # Average channels if more than 1
#         if sample_data.shape[-1] > 1:
#             sample_data = np.mean(sample_data, axis=-1)  # Shape: (time, height, width)
#         else:
#             sample_data = sample_data[..., 0]  # Shape: (time, height, width)
        
#         processed_inputs.append(sample_data)
#         n_timesteps_per_input.append(sample_data.shape[0])
    
#     # Process target
#     target_sample = target[sample_idx]  # Shape: (time, height, width, channels)
#     if target_sample.shape[-1] > 1:
#         target_sample = np.mean(target_sample, axis=-1)
#     else:
#         target_sample = target_sample[..., 0]
    
#     n_target_timesteps = target_sample.shape[0]
    
#     # Calculate total number of plots
#     total_input_timesteps = sum(n_timesteps_per_input)
#     total_plots = total_input_timesteps + n_target_timesteps
    
#     # Determine grid layout (aim for roughly square layout)
#     n_cols = int(np.ceil(np.sqrt(total_plots)))
#     n_rows = int(np.ceil(total_plots / n_cols))
    
#     # Auto-calculate figure size if not provided
#     if figsize is None:
#         figsize = (n_cols * 3, n_rows * 3)
    
#     # Create figure and subplots
#     fig = plt.figure(figsize=figsize)
#     gs = GridSpec(n_rows, n_cols, figure=fig, hspace=0.4, wspace=0.3)
    
#     plot_idx = 0
    
#     # Plot all inputs
#     for input_idx, (input_data, input_name) in enumerate(zip(processed_inputs, input_names)):
#         n_times = input_data.shape[0]
#         resolution = f"{input_data.shape[1]}x{input_data.shape[2]}"
        
#         # Determine title color based on number of timesteps
#         if n_times == 6:
#             title_color = 'orange'  # Past timestamps
#         elif n_times == 12:
#             title_color = 'darkblue'  # Future timestamps
#         else:
#             title_color = 'black'  # Default for other cases
        
#         for t in range(n_times):
#             row = plot_idx // n_cols
#             col = plot_idx % n_cols
#             ax = fig.add_subplot(gs[row, col])
            
#             im = ax.imshow(input_data[t], cmap=cmap, vmin=vmin, vmax=vmax, 
#                           origin='lower', aspect='auto')
#             ax.set_title(f"{input_name}\nT={t+1} ({resolution})", fontsize=9, 
#                         color=title_color)
#             ax.axis('off')
            
#             # Add colorbar for first timestep of each input
#             if t == 0:
#                 plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            
#             plot_idx += 1
    
#     # Plot target
#     target_resolution = f"{target_sample.shape[1]}x{target_sample.shape[2]}"
#     for t in range(n_target_timesteps):
#         row = plot_idx // n_cols
#         col = plot_idx % n_cols
#         ax = fig.add_subplot(gs[row, col])
        
#         im = ax.imshow(target_sample[t], cmap=cmap, vmin=vmin, vmax=vmax,
#                       origin='lower', aspect='auto')
#         ax.set_title(f"Target\nT={t+1} ({target_resolution})", fontsize=9, 
#                     fontweight='bold', color='red')
#         ax.axis('off')
        
#         # Add colorbar for first target timestep
#         if t == 0:
#             plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        
#         plot_idx += 1
    
#     # Add main title
#     fig.suptitle(f'Encoder-Decoder Data Visualization (Sample {sample_idx})', 
#                  fontsize=14, fontweight='bold', y=0.995)
    
#     return fig, gs


def plot_encoder_decoder_sample(data_inputs, target, sample_idx=0, 
                                  input_names=None, figsize=None, 
                                  cmap='viridis', vmin=None, vmax=None,
                                  output_path='encoder_decoder_animation.gif',
                                  sample_datetime=None, duration_per_frame=0.5):
    """
    Create animated visualization with one subplot per input showing all timesteps.
    
    Parameters:
    -----------
    data_inputs : list of np.ndarray
        List of N input arrays, each with shape (batch, time, height, width, channels)
    target : np.ndarray
        Target array with shape (batch, time, height, width, channels)
    sample_idx : int
        Index of the sample to plot from the batch (default: 0)
    input_names : list of str, optional
        Names for each input product (default: "Input 0", "Input 1", etc.)
    figsize : tuple, optional
        Figure size (width, height). Auto-calculated if None
    cmap : str
        Colormap for the plots (default: 'viridis')
    vmin, vmax : float, optional
        Min and max values for colorbar normalization
    output_path : str
        Path where the animation will be saved (default: 'encoder_decoder_animation.gif')
    sample_datetime : datetime, optional
        Datetime object for the sample (t=0). If None, not displayed
    duration_per_frame : float
        Duration in seconds for each frame (default: 0.5)
    
    Returns:
    --------
    str : Path to the saved animation file
    """
    
    n_inputs = len(data_inputs)
    
    # Generate default input names if not provided
    if input_names is None:
        input_names = [f"Input {i}" for i in range(n_inputs)]
    
    # Extract the specific sample and average channels if needed
    processed_inputs = []
    n_timesteps_per_input = []
    
    for data in data_inputs:
        # Extract sample
        sample_data = data[sample_idx]  # Shape: (time, height, width, channels)
        
        # Average channels if more than 1
        if sample_data.shape[-1] > 1:
            sample_data = np.mean(sample_data, axis=-1)  # Shape: (time, height, width)
        else:
            sample_data = sample_data[..., 0]  # Shape: (time, height, width)
        
        processed_inputs.append(sample_data)
        n_timesteps_per_input.append(sample_data.shape[0])
    
    # Process target
    target_sample = target[sample_idx]  # Shape: (time, height, width, channels)
    if target_sample.shape[-1] > 1:
        target_sample = np.mean(target_sample, axis=-1)
    else:
        target_sample = target_sample[..., 0]
    
    n_target_timesteps = target_sample.shape[0]
    
    # Total number of products (inputs + target)
    total_products = n_inputs + 1
    
    # Find maximum number of timesteps for animation
    max_timesteps = max(n_timesteps_per_input + [n_target_timesteps])
    
    # Determine grid layout (aim for roughly square layout)
    n_cols = int(np.ceil(np.sqrt(total_products)))
    n_rows = int(np.ceil(total_products / n_cols))
    
    # Auto-calculate figure size if not provided
    if figsize is None:
        figsize = (n_cols * 4, n_rows * 4)
    
    # Create figure and subplots
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(n_rows, n_cols, figure=fig, hspace=0.5, wspace=0.3)
    
    # Store subplot info
    subplots = []
    images = []
    colorbars = []
    
    # Create subplots for inputs
    for input_idx, (input_data, input_name) in enumerate(zip(processed_inputs, input_names)):
        row = input_idx // n_cols
        col = input_idx % n_cols
        ax = fig.add_subplot(gs[row, col])
        
        n_times = input_data.shape[0]
        resolution = f"{input_data.shape[1]}x{input_data.shape[2]}"
        
        # Determine title color based on number of timesteps
        if n_times == 6:
            title_color = 'orange'  # Past timestamps
        elif n_times == 12:
            title_color = 'darkblue'  # Future timestamps
        else:
            title_color = 'black'
        
        # Initial plot (first timestep)
        im = ax.imshow(input_data[0], cmap=cmap, vmin=vmin, vmax=vmax, aspect='equal')
        ax.set_title(f"{input_name}\nT=1/{n_times} ({resolution})", 
                    fontsize=10, color=title_color, fontweight='bold')
        ax.axis('off')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        
        subplots.append({
            'ax': ax,
            'data': input_data,
            'name': input_name,
            'n_times': n_times,
            'resolution': resolution,
            'color': title_color,
            'is_target': False
        })
        images.append(im)
        colorbars.append(cbar)
    
    # Create subplot for target
    target_idx = n_inputs
    row = target_idx // n_cols
    col = target_idx % n_cols
    ax = fig.add_subplot(gs[row, col])
    
    target_resolution = f"{target_sample.shape[1]}x{target_sample.shape[2]}"
    
    # Initial plot (first timestep)
    im = ax.imshow(target_sample[0], cmap=cmap, vmin=vmin, vmax=vmax, aspect='equal')
    ax.set_title(f"Target\nT=1/{n_target_timesteps} ({target_resolution})", 
                fontsize=10, fontweight='bold', color='red')
    ax.axis('off')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    subplots.append({
        'ax': ax,
        'data': target_sample,
        'name': 'Target',
        'n_times': n_target_timesteps,
        'resolution': target_resolution,
        'color': 'red',
        'is_target': True
    })
    images.append(im)
    colorbars.append(cbar)
    
    # Add main title with timestamp if provided
    if sample_datetime:
        main_title = f'Encoder-Decoder Sample {sample_idx}\n{sample_datetime.strftime("%Y-%m-%d %H:%M:%S")}'
    else:
        main_title = f'Encoder-Decoder Sample {sample_idx}'
    
    fig.suptitle(main_title, fontsize=14, fontweight='bold', y=0.98)
    
    print(f"\nCreating animation for {total_products} products:")
    print(f"  - Max timesteps: {max_timesteps}")
    print(f"  - Output: {output_path}")
    
    # Animation update function
    def update_frame(frame):
        """Update all subplots for the current frame"""
        for idx, (subplot_info, im) in enumerate(zip(subplots, images)):
            data = subplot_info['data']
            n_times = subplot_info['n_times']
            
            # Use modulo to cycle through timesteps if this product has fewer timesteps
            t = frame % n_times
            
            # Update image data
            im.set_array(data[t])
            
            # Update title
            title = f"{subplot_info['name']}\nT={t+1}/{n_times} ({subplot_info['resolution']})"
            subplot_info['ax'].set_title(title, fontsize=10, 
                                        color=subplot_info['color'],
                                        fontweight='bold')
        
        return images
    
    # Create animation
    print("Generating animation frames...")
    fps = 1.0 / duration_per_frame
    anim = animation.FuncAnimation(fig, update_frame, frames=max_timesteps,
                                   interval=duration_per_frame * 1000,
                                   blit=False, repeat=True)
    
    # Save animation
    print(f"Saving animation to {output_path}...")
    try:
        # Try Pillow for GIF
        Writer = animation.PillowWriter(fps=fps)
        anim.save(output_path, writer=Writer, dpi=100)
        print(f"✓ Animation saved successfully!")
    except Exception as e:
        print(f"✗ Error saving animation: {e}")
        print("Trying alternative: saving as MP4...")
        
        try:
            # Try ffmpeg for MP4
            mp4_path = output_path.replace('.gif', '.mp4')
            Writer = animation.writers['ffmpeg']
            writer = Writer(fps=fps, bitrate=1800)
            anim.save(mp4_path, writer=writer, dpi=100)
            print(f"✓ Video saved successfully to {mp4_path}!")
            output_path = mp4_path
        except Exception as e2:
            print(f"✗ Error saving video: {e2}")
            print("Saving as static PNG instead...")
            png_path = output_path.replace('.gif', '.png')
            plt.savefig(png_path, dpi=150, bbox_inches='tight')
            print(f"✓ Static image saved to {png_path}")
            output_path = png_path
    
    plt.close(fig)


# def plot_input_distributions(data_inputs, sample_idx=0, input_names=None, 
#                              figsize=None, n_cols=3):
#     """
#     Plot value distributions for all input data products.
    
#     For each input, displays a histogram showing the distribution of all pixel 
#     values across all timesteps with 5 equal bins from 0 to max value.
    
#     Parameters:
#     -----------
#     data_inputs : list of np.ndarray
#         List of N input arrays, each with shape (batch, time, height, width, channels)
#     sample_idx : int
#         Index of the sample to plot from the batch (default: 0)
#     input_names : list of str, optional
#         Names for each input product (default: "Input 0", "Input 1", etc.)
#     figsize : tuple, optional
#         Figure size (width, height). Auto-calculated if None
#     n_cols : int
#         Number of columns in the grid layout (default: 3)
    
#     Returns:
#     --------
#     fig, axes : matplotlib figure and axes objects
#     """
    
#     n_inputs = len(data_inputs)
    
#     # Generate default input names if not provided
#     if input_names is None:
#         input_names = [f"Input {i}" for i in range(n_inputs)]
    
#     # Calculate grid layout
#     n_rows = int(np.ceil(n_inputs / n_cols))
    
#     # Auto-calculate figure size if not provided
#     if figsize is None:
#         figsize = (n_cols * 5, n_rows * 4)
    
#     # Create figure and subplots
#     fig = plt.figure(figsize=figsize)
#     gs = GridSpec(n_rows, n_cols, figure=fig, hspace=2, wspace=0.4)
    
#     for input_idx, (data, input_name) in enumerate(zip(data_inputs, input_names)):
#         # Extract sample
#         sample_data = data[sample_idx]  # Shape: (time, height, width, channels)
        
#         # Average channels if more than 1
#         if sample_data.shape[-1] > 1:
#             sample_data = np.mean(sample_data, axis=-1)  # Shape: (time, height, width)
#         else:
#             sample_data = sample_data[..., 0]  # Shape: (time, height, width)
        
#         # Get all pixel values across all timesteps
#         all_values = sample_data.flatten()
        
#         # Get data statistics
#         n_timesteps = sample_data.shape[0]
#         resolution = f"{sample_data.shape[1]}x{sample_data.shape[2]}"
#         min_val = 0  # Always start from 0
#         max_val = all_values.max()
        
#         # Handle case where all values are zero or max_val == min_val
#         has_warning = False
#         warning_text = ""
#         if max_val <= min_val:
#             if max_val == 0:
#                 warning_text = "⚠ ALL ZERO VALUES"
#                 print(f"WARNING: {input_name} (Input {input_idx}) has all zero values!")
#             else:
#                 warning_text = f"⚠ ALL CONSTANT ({max_val:.2f})"
#                 print(f"WARNING: {input_name} (Input {input_idx}) has all constant values: {max_val:.2f}")
#             has_warning = True
#             max_val = min_val + 1  # Set a minimum range for plotting
        
#         # Create 5 equal-width bins
#         bin_edges = np.linspace(min_val, max_val, 6)  # 6 edges = 5 bins
        
#         # Create bin labels
#         bin_labels = []
#         for i in range(5):
#             bin_labels.append(f'{bin_edges[i]:.2f}-{bin_edges[i+1]:.2f}')
        
#         # Calculate histogram
#         hist, _ = np.histogram(all_values, bins=bin_edges)
        
#         # Determine title color based on number of timesteps
#         if n_timesteps == 6:
#             title_color = 'orange'  # Past timestamps
#         elif n_timesteps == 12:
#             title_color = 'darkblue'  # Future timestamps
#         else:
#             title_color = 'black'  # Default
        
#         # Create subplot
#         row = input_idx // n_cols
#         col = input_idx % n_cols
#         ax = fig.add_subplot(gs[row, col])
        
#         # Create bar plot
#         colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#6A994E']
#         bars = ax.bar(range(len(hist)), hist, color=colors, 
#                      edgecolor='black', linewidth=0.5)
        
#         ax.set_xticks(range(len(bin_labels)))
#         ax.set_xticklabels(bin_labels, rotation=45, ha='right', fontsize=9)
#         ax.set_ylabel('Pixel Count', fontsize=10)
        
#         # Create title with warning if needed
#         title_text = f'{input_name}\n{n_timesteps}T × {resolution} | Range: [{min_val:.2f}, {max_val:.2f}]'
#         if has_warning:
#             title_text = f'{input_name}\n{warning_text}\n{n_timesteps}T × {resolution}'
        
#         ax.set_title(title_text, fontsize=10, fontweight='bold', color=title_color)
#         ax.grid(axis='y', alpha=0.3, linestyle='--')
        
#         # Add value labels on bars
#         for bar in bars:
#             height = bar.get_height()
#             if height > 0:
#                 ax.text(bar.get_x() + bar.get_width()/2., height,
#                        f'{int(height)}',
#                        ha='center', va='bottom', fontsize=8)
    
#     # Hide empty subplots if n_inputs doesn't fill the grid
#     for idx in range(n_inputs, n_rows * n_cols):
#         row = idx // n_cols
#         col = idx % n_cols
#         ax = fig.add_subplot(gs[row, col])
#         ax.axis('off')
    
#     # Add main title
#     fig.suptitle(f'Input Data Distribution Analysis (Sample {sample_idx})', 
#                 fontsize=14, fontweight='bold', y=0.998)
    
#     return fig, gs

def plot_input_distributions(data_inputs, sample_idx=0, input_names=None, 
                             figsize=None, n_cols=3):
    """
    Plot value distributions for all input data products.
    
    For each input, displays a histogram showing the distribution of all pixel 
    values across all timesteps with 5 equal bins from 0 to max value.
    
    Parameters:
    -----------
    data_inputs : list of np.ndarray
        List of N input arrays, each with shape (batch, time, height, width, channels)
    sample_idx : int
        Index of the sample to plot from the batch (default: 0)
    input_names : list of str, optional
        Names for each input product (default: "Input 0", "Input 1", etc.)
    figsize : tuple, optional
        Figure size (width, height). Auto-calculated if None
    n_cols : int
        Number of columns in the grid layout (default: 3)
    
    Returns:
    --------
    fig, axes : matplotlib figure and axes objects
    """
    
    n_inputs = len(data_inputs)
    
    # Generate default input names if not provided
    if input_names is None:
        input_names = [f"Input {i}" for i in range(n_inputs)]
    
    # Calculate grid layout
    n_rows = int(np.ceil(n_inputs / n_cols))
    
    # Auto-calculate figure size if not provided
    if figsize is None:
        figsize = (n_cols * 5, n_rows * 4)
    
    # Create figure and subplots
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(n_rows, n_cols, figure=fig, hspace=0.4, wspace=0.3)
    
    for input_idx, (data, input_name) in enumerate(zip(data_inputs, input_names)):
        # Extract sample
        sample_data = data[sample_idx]  # Shape: (time, height, width, channels)
        
        # Average channels if more than 1
        if sample_data.shape[-1] > 1:
            sample_data = np.mean(sample_data, axis=-1)  # Shape: (time, height, width)
        else:
            sample_data = sample_data[..., 0]  # Shape: (time, height, width)
        
        # Get all pixel values across all timesteps
        all_values = sample_data.flatten()
        
        # Get data statistics
        n_timesteps = sample_data.shape[0]
        resolution = f"{sample_data.shape[1]}x{sample_data.shape[2]}"
        min_val = all_values.min()  # Actual minimum from data
        max_val = all_values.max()  # Actual maximum from data
        
        # Find most common value (mode) - round to 2 decimals for meaningful grouping
        rounded_values = np.round(all_values, 2)
        unique_vals, counts = np.unique(rounded_values, return_counts=True)
        most_common_val = unique_vals[np.argmax(counts)]
        most_common_count = counts[np.argmax(counts)]
        most_common_pct = (most_common_count / len(all_values)) * 100
        
        # Handle case where all values are the same
        has_warning = False
        warning_text = ""
        if max_val <= min_val:
            warning_text = f"⚠ ALL CONSTANT ({min_val:.3f})"
            print(f"WARNING: {input_name} (Input {input_idx}) has all constant values: {min_val:.3f}")
            has_warning = True
            max_val = min_val + 1e-6  # Set a tiny range for plotting
        
        # Create 5 equal-width bins
        bin_edges = np.linspace(min_val, max_val, 6)  # 6 edges = 5 bins
        
        # Create bin labels
        bin_labels = []
        for i in range(5):
            bin_labels.append(f'{bin_edges[i]:.2f}-{bin_edges[i+1]:.2f}')
        
        # Calculate histogram
        hist, _ = np.histogram(all_values, bins=bin_edges)
        
        # Determine title color based on number of timesteps
        if n_timesteps == 6:
            title_color = 'orange'  # Past timestamps
        elif n_timesteps == 12:
            title_color = 'darkblue'  # Future timestamps
        else:
            title_color = 'black'  # Default
        
        # Create subplot
        row = input_idx // n_cols
        col = input_idx % n_cols
        ax = fig.add_subplot(gs[row, col])
        
        # Create bar plot
        colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#6A994E']
        bars = ax.bar(range(len(hist)), hist, color=colors, 
                     edgecolor='black', linewidth=0.5)
        
        ax.set_xticks(range(len(bin_labels)))
        ax.set_xticklabels(bin_labels, rotation=45, ha='right', fontsize=9)
        ax.set_ylabel('Pixel Count', fontsize=10)
        
        # Create title with warning if needed
        title_text = f'{input_name}\n{n_timesteps}T × {resolution}'
        if has_warning:
            title_text += f'\n{warning_text}'
        else:
            # Add min/max/mode on separate lines
            title_text += f'\nMin: {min_val:.3f} | Max: {max_val:.3f}'
            title_text += f'\nMost Common: {most_common_val:.2f} ({most_common_pct:.1f}%)'
        
        ax.set_title(title_text, fontsize=10, fontweight='bold', color=title_color)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{int(height)}',
                       ha='center', va='bottom', fontsize=8)
    
    # Hide empty subplots if n_inputs doesn't fill the grid
    for idx in range(n_inputs, n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        ax = fig.add_subplot(gs[row, col])
        ax.axis('off')
    
    # Add main title
    fig.suptitle(f'Input Data Distribution Analysis (Sample {sample_idx})', 
                fontsize=14, fontweight='bold', y=0.998)


def plot_prediction_analysis(predictions, sample_idx=0, normalize=True, 
                             threshold=0.9, cmap='viridis', figsize=None):
    """
    Plot prediction maps with masked versions and pixel value distributions.
    
    For each timestep, displays:
    1. Original prediction map
    2. Masked prediction (only values > threshold visible)
    3. Distribution histogram with 5 bins (0-0.2, 0.2-0.4, 0.4-0.6, 0.6-0.8, 0.8-1.0)
    
    Parameters:
    -----------
    predictions : np.ndarray
        Prediction array with shape (batch, time, height, width, channels)
    sample_idx : int
        Index of the sample to plot from the batch (default: 0)
    normalize : bool
        If True, normalize predictions to [0, 1] range (default: True)
    threshold : float
        Threshold value for masking (default: 0.9)
    cmap : str
        Colormap for the plots (default: 'viridis')
    figsize : tuple, optional
        Figure size (width, height). Auto-calculated if None
    
    Returns:
    --------
    fig, axes : matplotlib figure and axes objects
    """
    
    # Extract sample and handle channels
    pred_sample = predictions[sample_idx]  # Shape: (time, height, width, channels)
    
    if pred_sample.shape[-1] > 1:
        pred_sample = np.mean(pred_sample, axis=-1)  # Average channels
    else:
        pred_sample = pred_sample[..., 0]  # Remove channel dimension
    
    n_timesteps = pred_sample.shape[0]
    
    # Normalize to [0, 1] if requested
    if normalize:
        pred_min = pred_sample.min()
        pred_max = pred_sample.max()
        if pred_max > pred_min:
            pred_sample = (pred_sample - pred_min) / (pred_max - pred_min)
        else:
            pred_sample = np.zeros_like(pred_sample)
    
    # Auto-calculate figure size if not provided
    if figsize is None:
        figsize = (15, n_timesteps * 2.5)
    
    # Create figure with GridSpec (3 columns per row)
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(n_timesteps, 3, figure=fig, hspace=0.6, wspace=0.5,
                  width_ratios=[1, 1, 0.8])
    
    # Define bin edges for distribution
    bin_edges = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    bin_labels = ['0-0.2', '0.2-0.4', '0.4-0.6', '0.6-0.8', '0.8-1.0']
    
    for t in range(n_timesteps):
        pred_map = pred_sample[t]
        # timestamp = n_timesteps - t # Reversed timestamp
        timestamp = t + 1 
        
        # Column 1: Original prediction map
        ax1 = fig.add_subplot(gs[t, 0])
        im1 = ax1.imshow(pred_map, cmap=cmap, vmin=0, vmax=1, aspect='equal')
        ax1.set_title(f'Prediction T={timestamp}', fontsize=10, fontweight='bold')
        ax1.axis('off')
        plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
        
        # Column 2: Masked prediction (only values > threshold)
        ax2 = fig.add_subplot(gs[t, 1])
        masked_pred = np.ma.masked_where(pred_map <= threshold, pred_map)
        im2 = ax2.imshow(masked_pred, cmap=cmap, vmin=0, vmax=1, aspect='equal')
        ax2.set_title(f'Masked (>{threshold}) T={timestamp}', fontsize=10, 
                     fontweight='bold', color='darkred')
        ax2.axis('off')
        
        # Add text showing percentage of pixels above threshold
        pct_above = (pred_map > threshold).sum() / pred_map.size * 100
        ax2.text(0.02, 0.98, f'{pct_above:.1f}% > {threshold}', 
                transform=ax2.transAxes, fontsize=9, 
                verticalalignment='top', bbox=dict(boxstyle='round', 
                facecolor='white', alpha=0.8))
        plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
        
        # Column 3: Distribution histogram
        ax3 = fig.add_subplot(gs[t, 2])
        
        # Calculate histogram
        hist, _ = np.histogram(pred_map.flatten(), bins=bin_edges)
        
        # Create bar plot
        colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#6A994E']
        bars = ax3.bar(range(len(hist)), hist, color=colors, 
                      edgecolor='black', linewidth=0.5)
        
        ax3.set_xticks(range(len(bin_labels)))
        ax3.set_xticklabels(bin_labels, rotation=45, ha='right', fontsize=8)
        ax3.set_ylabel('Pixel Count', fontsize=9)
        ax3.set_title(f'Distribution T={timestamp}', fontsize=10, fontweight='bold')
        ax3.grid(axis='y', alpha=0.3, linestyle='--')
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax3.text(bar.get_x() + bar.get_width()/2., height,
                        f'{int(height)}',
                        ha='center', va='bottom', fontsize=7)
    
    # Add main title
    fig.suptitle(f'Prediction Analysis (Sample {sample_idx})', 
                fontsize=14, fontweight='bold', y=0.998)
    
    return fig, gs


def plot_prediction_distribution(predictions, sample_idx=0, cmap='viridis', figsize=None):
    """
    Plot prediction maps with automatic data-range distribution histograms.
    
    For each timestep, displays:
    1. Prediction map
    2. Distribution histogram with 5 equal bins from 0 to max value
    
    Parameters:
    -----------
    predictions : np.ndarray
        Prediction array with shape (batch, time, height, width, channels)
    sample_idx : int
        Index of the sample to plot from the batch (default: 0)
    cmap : str
        Colormap for the plots (default: 'viridis')
    figsize : tuple, optional
        Figure size (width, height). Auto-calculated if None
    
    Returns:
    --------
    fig, axes : matplotlib figure and axes objects
    """
    
    # Extract sample and handle channels
    pred_sample = predictions[sample_idx]  # Shape: (time, height, width, channels)
    
    if pred_sample.shape[-1] > 1:
        pred_sample = np.mean(pred_sample, axis=-1)  # Average channels
    else:
        pred_sample = pred_sample[..., 0]  # Remove channel dimension
    
    n_timesteps = pred_sample.shape[0]
    
    # Find the maximum value across all timesteps for consistent binning
    max_val = pred_sample.max()
    min_val = 0  # Always start from 0
    
    # Create 5 equal-width bins
    bin_edges = np.linspace(min_val, max_val, 6)  # 6 edges = 5 bins
    
    # Create bin labels
    bin_labels = []
    for i in range(5):
        bin_labels.append(f'{bin_edges[i]:.2f}-{bin_edges[i+1]:.2f}')
    
    # Auto-calculate figure size if not provided
    if figsize is None:
        figsize = (12, n_timesteps * 2.5)
    
    # Create figure with GridSpec (2 columns per row)
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(n_timesteps, 2, figure=fig, hspace=0.9, wspace=0.3,
                  width_ratios=[1.2, 1])
    
    for t in range(n_timesteps):
        pred_map = pred_sample[t]
        # timestamp = n_timesteps - t  # Reverse timestamp
        timestamp = t + 1

        # Column 1: Prediction map
        ax1 = fig.add_subplot(gs[t, 0])
        im1 = ax1.imshow(pred_map, cmap=cmap, vmin=min_val, vmax=max_val, aspect='equal')
        ax1.set_title(f'Prediction T={timestamp}', fontsize=10, fontweight='bold')
        ax1.axis('off')
        plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
        
        # Column 2: Distribution histogram
        ax2 = fig.add_subplot(gs[t, 1])
        
        # Calculate histogram
        hist, _ = np.histogram(pred_map.flatten(), bins=bin_edges)
        
        # Create bar plot
        colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#6A994E']
        bars = ax2.bar(range(len(hist)), hist, color=colors, 
                      edgecolor='black', linewidth=0.5)
        
        ax2.set_xticks(range(len(bin_labels)))
        ax2.set_xticklabels(bin_labels, rotation=45, ha='right', fontsize=8)
        ax2.set_ylabel('Pixel Count', fontsize=9)
        ax2.set_title(f'Distribution T={timestamp}', fontsize=10, fontweight='bold')
        ax2.grid(axis='y', alpha=0.3, linestyle='--')
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax2.text(bar.get_x() + bar.get_width()/2., height,
                        f'{int(height)}',
                        ha='center', va='bottom', fontsize=7)
    
    # Add main title with data range info
    fig.suptitle(f'Prediction Distribution Analysis (Sample {sample_idx}) - Range: [{min_val:.2f}, {max_val:.2f}]', 
                fontsize=14, fontweight='bold', y=0.998)
    
    return fig, gs


def plot_meteorological_regridded_data(variables, datetime_str, args, figsize=(15, 10)):
    """
    Plot meteorological data for specified variables at a given datetime.
    
    Parameters:
    -----------
    variables : list of str
        List of variable names to plot (e.g., ['RZC', 'HRV', 'occurrence'])
    datetime_str : str
        Date and time in format "yyyy-mm-dd hh:mm"
    figsize : tuple, optional
        Figure size for the plot
    """

    from c4dl.features.test_nc import (
        parse_nwp_date_and_time_filenames_to_iso, parse_filename_datetime,
        parse_nc4_filename_datetime, parse_lightning_filename, parse_nwcsaf_filename_simple
    )
    
    # Define variable categories
    radar = ["RZC", "CZC", "BZC", "EZC-20", "EZC-45", "HZC", "LZC", "CPCH"]
    lightning = ["occurrence", "density", "current"]
    satellite = [
        "HRV", "VIS006", "VIS008",
        "IR_016", "IR_039", "IR_087", "IR_097", "IR_108", "IR_120", "IR_134",
        "WV_062", "WV_073"
    ]
    nwcsaf = ["ctth_alti", "ctth_tempe", "cmic_phase", "cmic_cot"]
    nwp = [
        "CAPE_MU", "CIN_MU", "SLI", 
        "HZEROCL", "LCL_ML", "MCONV", "OMEGA",
        "T_2M", "T_SO", "SOILTYP"
    ]
    
    # Get repository root and base directory
    repo_root = Path(__file__).parent.parent
    base_dir = repo_root / 'our_data' / 'regridded_data'
    debug_dir = repo_root / 'debug_outputs'
    
    # Create debug output directory if it doesn't exist
    if not os.path.exists(debug_dir):
        os.makedirs(debug_dir)
    
    # Parse the datetime string
    target_dt = datetime.strptime(datetime_str, "%Y-%m-%d %H:%M")
    
    # Create figure for spatial plots
    n_vars = len(variables)
    n_cols = min(3, n_vars)
    n_rows = (n_vars + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    if n_vars == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    # Create figure for histograms
    fig_hist, axes_hist = plt.subplots(n_rows, n_cols, figsize=figsize)
    if n_vars == 1:
        axes_hist = np.array([axes_hist])
    axes_hist = axes_hist.flatten()
    
    # Process each variable
    for idx, var in enumerate(variables):
        # Determine variable category
        if var in radar:
            category = 'radar'
            var_data_dir = base_dir / 'radar_data'
            var_dir = var_data_dir / var
        elif var in lightning:
            category = 'lightning'
            var_data_dir = base_dir / 'lightning_data'
            var_dir = var_data_dir / var
        elif var in satellite:
            category = 'satellite'
            var_data_dir = base_dir / 'satellite_data'
            var_dir = var_data_dir / var
        elif var in nwcsaf:
            category = 'nwcsaf'
            var_data_dir = base_dir / 'nwcsaf_data'
            var_dir = var_data_dir
        elif var in nwp:
            category = 'nwp'
            var_data_dir = base_dir / 'nwp_data'
            var_dir = var_data_dir
        else:
            print(f"Unknown variable: {var}")
            continue
        
        # Find matching file
        data_file = None

        if category in ['radar', 'lightning', 'satellite']:
            # Structure: var_data_dir / var / timestamp / files
            print(var_dir)
            # exit(0)
            if os.path.exists(var_dir):
                for timestamp_dir in os.listdir(var_dir):
                    timestamp_path = var_dir / timestamp_dir
                    print(timestamp_path)
                    if os.path.isdir(timestamp_path):
                        for filename in os.listdir(timestamp_path):
                            if filename.endswith('.npy'):
                                # Parse filename based on category
                                if category == 'radar':
                                    result = parse_filename_datetime(filename.replace('.npy', '.nc'))
                                    iso_format = result['iso_format']
                                elif category == 'satellite':
                                    result = parse_nc4_filename_datetime(filename.replace('.npy', '.nc'))
                                    iso_format = (
                                        datetime.fromisoformat(result['iso_format']) + timedelta(minutes=1)
                                    ).isoformat()
                                elif category == 'lightning':
                                    iso_format = parse_lightning_filename(filename.replace('.npy', '.nc'))
                                    print(iso_format)
                                    # exit(0)
                                
                                # Remove nanosecond precision if present
                                if iso_format.endswith('.000000000'):
                                    iso_format = iso_format[:-10]
                                
                                file_dt = datetime.fromisoformat(iso_format)
                                
                                if file_dt == target_dt:
                                    data_file = timestamp_path / filename
                                    break
                    if data_file:
                        break

        elif category in ['nwp', 'nwcsaf']:
            # Structure: var_data_dir / timestamp / files
            if os.path.exists(var_dir):
                for timestamp_dir in os.listdir(var_dir):
                    timestamp_path = var_dir / timestamp_dir
                    if os.path.isdir(timestamp_path):
                        for filename in os.listdir(timestamp_path):
                            if filename.endswith('.nc'):
                                # Parse filename based on category
                                if category == 'nwp':
                                    iso_format = parse_nwp_date_and_time_filenames_to_iso(
                                        timestamp_dir, filename
                                    )
                                elif category == 'nwcsaf':
                                    iso_format = parse_nwcsaf_filename_simple(filename)
                                
                                # Remove nanosecond precision if present
                                if iso_format.endswith('.000000000'):
                                    iso_format = iso_format[:-10]
                                
                                file_dt = datetime.fromisoformat(iso_format)
                                
                                if file_dt == target_dt:
                                    # For NWCSAF, find the file containing the variable
                                    if category == 'nwcsaf':
                                        # Determine which file type we need based on variable
                                        if var.startswith('ctth'):
                                            file_type = 'CTTH'
                                        elif var.startswith('cmic'):
                                            file_type = 'CMIC'
                                        else:
                                            continue
                                        
                                        # Check if this file is the correct type
                                        if file_type in filename:
                                            data_file = timestamp_path / filename
                                            break
                                    else:
                                        data_file = timestamp_path / filename
                                        break
                    if data_file:
                        break
        print(f"Loading data from: {data_file}")
        # Load and plot data
        if data_file:
            try:
                if data_file.suffix == '.npy':
                    
                    data = np.load(data_file)
                elif data_file.suffix == '.nc':
                    ds = Dataset(data_file, 'r')
                    # Find the variable in the netcdf file
                    if category == 'nwp':
                        data = ds.variables[var.lower()][:]
                    elif category == 'nwcsaf':
                        # NWCSAF variable names match exactly
                        if var in ds.variables:
                            data = ds.variables[var][:]
                        else:
                            ds.close()
                            raise KeyError(f"Variable {var} not found in {data_file}")
                    ds.close()
                
                # Plot spatial data
                # Transform radar data
                if args.transformed and category == 'radar':
                    data = np.flipud(data)  # Flip vertically

                im = axes[idx].imshow(data, cmap='viridis', aspect='equal')
                axes[idx].set_title(f'{var}\n{datetime_str}')
                axes[idx].set_xlabel('X')
                axes[idx].set_ylabel('Y')
                plt.colorbar(im, ax=axes[idx])

                # Add red message if data was transformed
                if args.transformed:
                    axes[idx].text(
                        0.02, 0.98, 'TRANSFORMED\n(Flipped Horizontally)',
                        transform=axes[idx].transAxes,
                        verticalalignment='top',
                        horizontalalignment='left',
                        color='red',
                        fontweight='bold',
                        fontsize=9,
                        bbox=dict(boxstyle='round', facecolor='white', edgecolor='red', alpha=0.8)
                    )

                # Plot histogram
                data_flat = data.flatten()
                # Remove NaN and inf values for histogram
                data_flat = data_flat[np.isfinite(data_flat)]
                
                if len(data_flat) > 0:
                    axes_hist[idx].hist(data_flat, bins=50, color='steelblue', 
                                       edgecolor='black', alpha=0.7)
                    axes_hist[idx].set_title(f'{var} Distribution\n{datetime_str}')
                    axes_hist[idx].set_xlabel('Value')
                    axes_hist[idx].set_ylabel('Frequency')
                    axes_hist[idx].grid(True, alpha=0.3)
                    
                    # Add statistics text
                    stats_text = (f'Mean: {np.mean(data_flat):.2f}\n'
                                 f'Std: {np.std(data_flat):.2f}\n'
                                 f'Min: {np.min(data_flat):.2f}\n'
                                 f'Max: {np.max(data_flat):.2f}')
                    axes_hist[idx].text(0.95, 0.95, stats_text,
                                       transform=axes_hist[idx].transAxes,
                                       verticalalignment='top',
                                       horizontalalignment='right',
                                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                                       fontsize=9)
                else:
                    axes_hist[idx].text(0.5, 0.5, 'No valid data', 
                                       ha='center', va='center', 
                                       transform=axes_hist[idx].transAxes)
                
            except Exception as e:
                axes[idx].text(0.5, 0.5, f'Error loading {var}:\n{str(e)}', 
                             ha='center', va='center', transform=axes[idx].transAxes)
                axes[idx].set_title(f'{var} - Error')

                axes_hist[idx].text(0.5, 0.5, f'Error loading {var}:\n{str(e)}', 
                                   ha='center', va='center', 
                                   transform=axes_hist[idx].transAxes)
                axes_hist[idx].set_title(f'{var} - Error')
        else:
            axes[idx].text(0.5, 0.5, f'No data found for {var}\nat {datetime_str}', 
                         ha='center', va='center', transform=axes[idx].transAxes)
            axes[idx].set_title(f'{var} - Not Found')
            axes[idx].set_xlabel('X')
            axes[idx].set_ylabel('Y')

            axes_hist[idx].text(0.5, 0.5, f'No data found for {var}\nat {datetime_str}', 
                   ha='center', va='center', 
                   transform=axes_hist[idx].transAxes)
            axes_hist[idx].set_title(f'{var} - Not Found')
        
        axes[idx].set_xlabel('X')
        axes[idx].set_ylabel('Y')
    
    # Hide unused subplots
    for idx in range(n_vars, len(axes)):
        axes[idx].axis('off')
        axes_hist[idx].axis('off')
    
    plt.tight_layout()
    fig_hist.tight_layout()
    plt.show()
    
    return fig, axes


def _process_single_file_worker(task):
    """
    Worker function to process a single file and save the plot.
    Must be at module level for multiprocessing pickling.
    """
    from pathlib import Path
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from datetime import datetime, timedelta
    from netCDF4 import Dataset
    from c4dl.features.test_nc import (
        parse_nwp_date_and_time_filenames_to_iso, parse_filename_datetime,
        parse_nc4_filename_datetime, parse_lightning_filename, parse_nwcsaf_filename_simple
    )
    
    data_file, filename, var, category, var_output_dir, transform = task
    
    try:
        # Parse datetime from filename
        if category == 'radar':
            result = parse_filename_datetime(filename.replace('.npy', '.nc'))
            iso_format = result['iso_format']
        elif category == 'satellite':
            result = parse_nc4_filename_datetime(filename.replace('.npy', '.nc'))
            iso_format = (
                datetime.fromisoformat(result['iso_format']) + timedelta(minutes=1)
            ).isoformat()
        elif category == 'lightning':
            iso_format = parse_lightning_filename(filename.replace('.npy', '.nc'))
        elif category == 'nwp':
            # Extract timestamp_dir from file_path
            timestamp_dir = Path(data_file).parent.name
            iso_format = parse_nwp_date_and_time_filenames_to_iso(timestamp_dir, filename)
        elif category == 'nwcsaf':
            iso_format = parse_nwcsaf_filename_simple(filename)
        
        # Remove nanosecond precision if present
        if iso_format.endswith('.000000000'):
            iso_format = iso_format[:-10]
        
        file_dt = datetime.fromisoformat(iso_format)
        datetime_str = file_dt.strftime("%Y-%m-%d %H:%M")
        
        # Load data
        data_file_path = Path(data_file)
        if data_file_path.suffix == '.npy':
            data = np.load(data_file_path)
        elif data_file_path.suffix == '.nc':
            ds = Dataset(str(data_file_path), 'r')
            if category == 'nwp':
                data = ds.variables[var.lower()][:]
            elif category == 'nwcsaf':
                if var in ds.variables:
                    data = ds.variables[var][:]
                else:
                    ds.close()
                    raise KeyError(f"Variable {var} not found in {data_file}")
            ds.close()
        
        # Transform radar data if requested
        if transform and category == 'radar':
            data = np.flipud(data)
        
        # Create plot
        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(data, cmap='viridis', aspect='equal')
        ax.set_title(f'{var}\n{datetime_str}')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        plt.colorbar(im, ax=ax, label='Value')
        
        # Add transformation message if applicable
        if transform and category == 'radar':
            ax.text(
                0.02, 0.98, 'TRANSFORMED\n(Flipped Vertically)',
                transform=ax.transAxes,
                verticalalignment='top',
                horizontalalignment='left',
                color='red',
                fontweight='bold',
                fontsize=9,
                bbox=dict(boxstyle='round', facecolor='white', edgecolor='red', alpha=0.8)
            )
        
        # Save plot
        timestamp_safe = datetime_str.replace(':', '-').replace(' ', '_')
        output_filename = f"{var}_{timestamp_safe}.png"
        output_path = Path(var_output_dir) / output_filename
        
        plt.savefig(str(output_path), dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        return ('success', filename)
        
    except Exception as e:
        plt.close('all')
        return ('error', filename, str(e))


def save_all_regridded_data_plots(variables, args):
    """
    Save plots of all available timestamps for specified meteorological variables.
    Creates a folder for each variable in debug_outputs and saves all plots there.
    Uses multiprocessing to parallelize the work across all CPU cores.
    
    Parameters:
    -----------
    variables : list of str
        List of variable names to save plots for (e.g., ['RZC', 'HRV', 'occurrence'])
    args : argparse.Namespace
        Command line arguments containing transformation flags
    """
    from pathlib import Path
    import os
    from multiprocessing import Pool, cpu_count
    
    # Define variable categories
    radar = ["RZC", "CZC", "BZC", "EZC-20", "EZC-45", "HZC", "LZC", "CPCH"]
    lightning = ["occurrence", "density", "current"]
    satellite = [
        "HRV", "VIS006", "VIS008",
        "IR_016", "IR_039", "IR_087", "IR_097", "IR_108", "IR_120", "IR_134",
        "WV_062", "WV_073"
    ]
    nwcsaf = ["ctth_alti", "ctth_tempe", "cmic_phase", "cmic_cot"]
    nwp = [
        "CAPE_MU", "CIN_MU", "SLI", 
        "HZEROCL", "LCL_ML", "MCONV", "OMEGA",
        "T_2M", "T_SO", "SOILTYP"
    ]
    
    # Get repository root and base directory
    repo_root = Path(__file__).parent.parent
    base_dir = repo_root / 'our_data' / 'regridded_data'
    debug_dir = repo_root / 'debug_outputs'
    
    # Create debug output directory if it doesn't exist
    if not os.path.exists(debug_dir):
        os.makedirs(debug_dir)
    
    # Windows has a limit of 63 handles for multiprocessing
    max_workers = min(cpu_count(), 8)
    print(f"\n{'='*80}")
    print(f"SAVING REGRIDDED DATA PLOTS (Using {max_workers} of {cpu_count()} available cores)")
    print(f"{'='*80}\n")
    
    # Process each variable
    for var in variables:
        print(f"\n{'─'*80}")
        print(f"Processing variable: {var}")
        print(f"{'─'*80}")
        
        # Determine variable category
        if var in radar:
            category = 'radar'
            var_data_dir = base_dir / 'radar_data'
            var_dir = var_data_dir / var
        elif var in lightning:
            category = 'lightning'
            var_data_dir = base_dir / 'lightning_data'
            var_dir = var_data_dir / f'lightning_{var}'
        elif var in satellite:
            category = 'satellite'
            var_data_dir = base_dir / 'satellite_data'
            var_dir = var_data_dir / var
        elif var in nwcsaf:
            category = 'nwcsaf'
            var_data_dir = base_dir / 'nwcsaf_data'
            var_dir = var_data_dir
        elif var in nwp:
            category = 'nwp'
            var_data_dir = base_dir / 'nwp_data'
            var_dir = var_data_dir
        else:
            print(f"❌ Unknown variable: {var}, skipping...")
            continue
        
        # Create folder for this variable
        var_output_dir = debug_dir / var
        if not os.path.exists(var_output_dir):
            os.makedirs(var_output_dir)
        print(f"   Output directory: {var_output_dir}")
        
        # Collect all files for this variable
        files_to_process = []
        
        if category in ['radar', 'lightning', 'satellite']:
            # Structure: var_data_dir / var / timestamp / files
            if os.path.exists(var_dir):
                for timestamp_dir in os.listdir(var_dir):
                    timestamp_path = var_dir / timestamp_dir
                    if os.path.isdir(timestamp_path):
                        for filename in os.listdir(timestamp_path):
                            if filename.endswith('.npy'):
                                file_path = timestamp_path / filename
                                files_to_process.append((file_path, filename))
        
        elif category in ['nwp', 'nwcsaf']:
            # Structure: var_data_dir / timestamp / files
            if os.path.exists(var_dir):
                for timestamp_dir in os.listdir(var_dir):
                    timestamp_path = var_dir / timestamp_dir
                    if os.path.isdir(timestamp_path):
                        for filename in os.listdir(timestamp_path):
                            if filename.endswith('.nc'):
                                # For NWCSAF, filter by file type
                                if category == 'nwcsaf':
                                    if var.startswith('ctth') and 'CTTH' in filename:
                                        file_path = timestamp_path / filename
                                        files_to_process.append((file_path, filename))
                                    elif var.startswith('cmic') and 'CMIC' in filename:
                                        file_path = timestamp_path / filename
                                        files_to_process.append((file_path, filename))
                                else:
                                    file_path = timestamp_path / filename
                                    files_to_process.append((file_path, filename))
        
        print(f"   Found {len(files_to_process)} files to process")
        
        if len(files_to_process) == 0:
            print(f"   ⚠️  No files found for {var}")
            continue
        
        # Prepare tasks for multiprocessing
        # Convert Path objects to strings for pickling
        tasks = [
            (str(file_path), filename, var, category, str(var_output_dir), args.transformed)
            for file_path, filename in files_to_process
        ]
        
        # Process files in parallel
        print(f"   Processing with {max_workers} workers...")
        
        with Pool(processes=max_workers) as pool:
            results = pool.map(_process_single_file_worker, tasks)
        
        # Count successes and errors
        saved_count = sum(1 for r in results if r[0] == 'success')
        error_count = sum(1 for r in results if r[0] == 'error')
        
        # Print errors if any
        if error_count > 0:
            print(f"\n   ❌ Errors encountered:")
            for result in results:
                if result[0] == 'error':
                    print(f"      {result[1]}: {result[2]}")
        
        print(f"\n   ✅ Variable {var} complete:")
        print(f"      Saved: {saved_count} plots")
        print(f"      Errors: {error_count} files")
        print(f"      Location: {var_output_dir}")
    
    print(f"\n{'='*80}")
    print(f"✅ ALL VARIABLES PROCESSED")
    print(f"{'='*80}\n")


def plot_distillation_curves(epochs, total_loss, soft_loss, hard_loss, 
                             figsize=(12, 4), save_path=None):
    """
    Plot knowledge distillation training curves in 3 separate subplots.
    
    Parameters:
    -----------
    epochs : list or array
        Epoch numbers
    total_loss : list or array
        Average total loss values
    soft_loss : list or array
        Average soft loss (teacher guidance) values
    hard_loss : list or array
        Average hard loss (ground truth) values
    figsize : tuple, optional
        Figure size (width, height)
    save_path : str, optional
        Path to save the figure. If None, displays the plot.
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    # Plot Total Loss
    axes[0].plot(epochs, total_loss, marker='o', linewidth=2, color='#2E86AB')
    axes[0].set_xlabel('Epoch', fontsize=11)
    axes[0].set_ylabel('Loss', fontsize=11)
    axes[0].set_title('Total Loss', fontsize=12, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    
    # Plot Soft Loss (Teacher Guidance)
    axes[1].plot(epochs, soft_loss, marker='s', linewidth=2, color='#A23B72')
    axes[1].set_xlabel('Epoch', fontsize=11)
    axes[1].set_ylabel('Loss', fontsize=11)
    axes[1].set_title('Soft Loss (Teacher Guidance)', fontsize=12, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    
    # Plot Hard Loss (Ground Truth)
    axes[2].plot(epochs, hard_loss, marker='^', linewidth=2, color='#F18F01')
    axes[2].set_xlabel('Epoch', fontsize=11)
    axes[2].set_ylabel('Loss', fontsize=11)
    axes[2].set_title('Hard Loss (Ground Truth)', fontsize=12, fontweight='bold')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save or show
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    else:
        plt.show()


import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.colors import ListedColormap, BoundaryNorm
import matplotlib.animation as animation
import numpy as np
from pathlib import Path


def visualize_continuous_predictions(predictions, targets, sample_idx=0, save_dir='bzc_plots', 
                                    figsize=(20, 12), dpi=150, show_difference=True, 
                                    cmap='viridis', var_name='BZC'):
    """
    Visualize continuous meteorological predictions vs targets with temporal alignment.
    
    Parameters:
    -----------
    predictions : numpy.ndarray
        Shape (batch, 12, 256, 256, 1) - Predicted values
    targets : numpy.ndarray  
        Shape (batch, 12, 256, 256, 1) - Target values
    sample_idx : int
        Which sample from batch to visualize (0-7)
    save_dir : str
        Directory to save plots
    figsize : tuple
        Figure size
    dpi : int
        Plot resolution
    show_difference : bool
        Whether to show difference maps
    cmap : str
        Colormap to use (default: 'viridis')
    var_name : str
        Name of the variable being predicted
    """
    
    # Create save directory
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    # Extract single sample
    pred_sample = predictions[sample_idx, :, :, :, 0]  # (12, 256, 256)
    target_sample = targets[sample_idx, :, :, :, 0]    # (12, 256, 256)

    # Reverse for better visualization (most recent first)
    pred_sample = pred_sample[::-1]
    target_sample = target_sample[::-1]
    
    print(f"Pred shape {pred_sample.shape}, Target shape {target_sample.shape}")
    print(f"Pred range [{pred_sample.min():.3f}, {pred_sample.max():.3f}], Target range [{target_sample.min():.3f}, {target_sample.max():.3f}]")

    # Time labels (5-minute intervals)
    # time_labels = [f't+{(i+1)*5}min' for i in range(12)]
    time_labels = [f't+{(12-i)*5}min' for i in range(12)]
    
    # Determine common color scale for predictions and targets
    vmin = min(pred_sample.min(), target_sample.min())
    vmax = max(pred_sample.max(), target_sample.max())
    
    # Create comprehensive visualization
    fig = plt.figure(figsize=figsize)
    
    if show_difference:
        # 4 rows: Targets, Predictions, Difference, Metrics
        gs = GridSpec(4, 13, figure=fig, height_ratios=[1, 1, 1, 0.8], 
                     width_ratios=[1]*12 + [0.1])  # Extra column for colorbar
    else:
        # 3 rows: Targets, Predictions, Metrics
        gs = GridSpec(3, 13, figure=fig, height_ratios=[1, 1, 0.8], 
                     width_ratios=[1]*12 + [0.1])
    
    # Plot targets (row 1)
    for t in range(12):
        ax = fig.add_subplot(gs[0, t])
        im = ax.imshow(target_sample[t], cmap=cmap, vmin=vmin, vmax=vmax, aspect='equal')
        ax.set_title(f'Target\n{time_labels[t]}', fontsize=10, fontweight='bold')
        ax.set_xticks([])
        ax.set_yticks([])
        if t == 0:
            ax.set_ylabel('Ground Truth', fontsize=12, fontweight='bold')
    
    # Plot predictions (row 2) 
    for t in range(12):
        ax = fig.add_subplot(gs[1, t])
        im = ax.imshow(pred_sample[t], cmap=cmap, vmin=vmin, vmax=vmax, aspect='equal')
        ax.set_title(f'Predicted\n{time_labels[t]}', fontsize=10, fontweight='bold')
        ax.set_xticks([])
        ax.set_yticks([])
        if t == 0:
            ax.set_ylabel('Predictions', fontsize=12, fontweight='bold')
    
    # Add colorbar
    cbar_ax = fig.add_subplot(gs[0:2, 12])
    cbar = plt.colorbar(im, cax=cbar_ax)
    cbar.set_label(var_name, fontsize=12)
    
    if show_difference:
        # Plot differences (row 3) - NO TIMESTAMPS HERE
        diff_cmap = plt.cm.RdBu_r
        max_diff = max(abs(pred_sample.min() - target_sample.min()), 
                      abs(pred_sample.max() - target_sample.max()))
        
        for t in range(12):
            ax = fig.add_subplot(gs[2, t])
            diff = pred_sample[t] - target_sample[t]
            im_diff = ax.imshow(diff, cmap=diff_cmap, vmin=-max_diff, vmax=max_diff, aspect='equal')
            ax.set_title(f'Diff', fontsize=10, fontweight='bold')  # NO TIMESTAMP
            ax.set_xticks([])
            ax.set_yticks([])
            if t == 0:
                ax.set_ylabel('Prediction - Target', fontsize=12, fontweight='bold')
        
        # Add difference colorbar
        diff_cbar_ax = fig.add_subplot(gs[2, 12])
        diff_cbar = plt.colorbar(im_diff, cax=diff_cbar_ax)
        diff_cbar.set_label('Difference', fontsize=12)
        
        metrics_row = 3
    else:
        metrics_row = 2
    
    # Plot temporal metrics (bottom row)
    metrics_ax = fig.add_subplot(gs[metrics_row, :12])
    
    # Calculate metrics over time
    mae_scores = []
    rmse_scores = []
    mean_pred_values = []
    mean_target_values = []
    
    for t in range(12):
        # MAE (Mean Absolute Error)
        mae = np.mean(np.abs(pred_sample[t] - target_sample[t]))
        mae_scores.append(mae)
        
        # RMSE (Root Mean Square Error)
        rmse = np.sqrt(np.mean((pred_sample[t] - target_sample[t])**2))
        rmse_scores.append(rmse)
        
        # Mean values
        mean_pred_values.append(np.mean(pred_sample[t]))
        mean_target_values.append(np.mean(target_sample[t]))
    
    # Plot metrics
    time_minutes = np.arange(5, 65, 5)
    metrics_ax.plot(time_minutes, mae_scores, 'b-o', linewidth=2, markersize=6, label='MAE')
    metrics_ax.plot(time_minutes, rmse_scores, 'g-s', linewidth=2, markersize=6, label='RMSE')
    
    # Secondary y-axis for mean values
    ax2 = metrics_ax.twinx()
    ax2.plot(time_minutes, mean_pred_values, 'r--^', linewidth=2, markersize=6, label='Mean Predicted')
    ax2.plot(time_minutes, mean_target_values, 'orange', linestyle='--', marker='v', 
             linewidth=2, markersize=6, label='Mean Target')
    
    # Format metrics plot
    metrics_ax.set_xlabel('Forecast Time (minutes)', fontsize=12)
    metrics_ax.set_ylabel('Error Metrics', fontsize=12, color='blue')
    metrics_ax.tick_params(axis='y', labelcolor='blue')
    metrics_ax.set_xlim(0, 65)
    metrics_ax.grid(True, alpha=0.3)
    
    ax2.set_ylabel(f'Mean {var_name} Value', fontsize=12, color='red')
    ax2.tick_params(axis='y', labelcolor='red')
    
    # Combined legend
    lines1, labels1 = metrics_ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    metrics_ax.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=10)
    
    plt.suptitle(f'{var_name} Nowcasting: Sample {sample_idx} - Temporal Evolution (1 Hour Forecast)', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    # Save plot
    plot_path = save_path / f'{var_name.lower()}_nowcasting_sample_{sample_idx}.png'
    plt.savefig(plot_path, dpi=dpi, bbox_inches='tight')
    print(f"Saved comprehensive plot: {plot_path}")
    plt.close()


def create_continuous_summary_plot(predictions, targets, save_dir='bzc_plots', 
                                   figsize=(15, 10), var_name='BZC'):
    """
    Create summary plots showing performance across all samples and timesteps.
    """
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    batch_size, timesteps = predictions.shape[:2]
    
    # Calculate metrics for all samples and timesteps
    all_mae = np.zeros((batch_size, timesteps))
    all_rmse = np.zeros((batch_size, timesteps))
    all_mean_pred = np.zeros((batch_size, timesteps))
    all_mean_target = np.zeros((batch_size, timesteps))
    
    for b in range(batch_size):
        for t in range(timesteps):
            pred = predictions[b, t, :, :, 0]
            target = targets[b, t, :, :, 0]
            
            # MAE
            all_mae[b, t] = np.mean(np.abs(pred - target))
            
            # RMSE
            all_rmse[b, t] = np.sqrt(np.mean((pred - target)**2))
            
            # Mean values
            all_mean_pred[b, t] = np.mean(pred)
            all_mean_target[b, t] = np.mean(target)
    
    # Create summary plots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=figsize)
    
    time_minutes = np.arange(5, 65, 5)
    
    # MAE over time
    mean_mae = np.mean(all_mae, axis=0)
    std_mae = np.std(all_mae, axis=0)
    ax1.plot(time_minutes, mean_mae, 'b-o', linewidth=2, markersize=6)
    ax1.fill_between(time_minutes, mean_mae - std_mae, mean_mae + std_mae, alpha=0.3)
    ax1.set_title('MAE Over Time', fontweight='bold')
    ax1.set_xlabel('Forecast Time (minutes)')
    ax1.set_ylabel('Mean Absolute Error')
    ax1.grid(True, alpha=0.3)
    
    # RMSE over time
    mean_rmse = np.mean(all_rmse, axis=0)
    std_rmse = np.std(all_rmse, axis=0)
    ax2.plot(time_minutes, mean_rmse, 'g-s', linewidth=2, markersize=6)
    ax2.fill_between(time_minutes, mean_rmse - std_rmse, mean_rmse + std_rmse, alpha=0.3)
    ax2.set_title('RMSE Over Time', fontweight='bold')
    ax2.set_xlabel('Forecast Time (minutes)')
    ax2.set_ylabel('Root Mean Square Error')
    ax2.grid(True, alpha=0.3)
    
    # Mean values comparison
    mean_pred = np.mean(all_mean_pred, axis=0)
    mean_targ = np.mean(all_mean_target, axis=0)
    ax3.plot(time_minutes, mean_pred, 'r--^', linewidth=2, markersize=6, label='Predicted')
    ax3.plot(time_minutes, mean_targ, 'orange', linestyle='--', marker='v', 
             linewidth=2, markersize=6, label='Target')
    ax3.set_title(f'Mean {var_name} Value Comparison', fontweight='bold')
    ax3.set_xlabel('Forecast Time (minutes)')
    ax3.set_ylabel(f'Mean {var_name} Value')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
    # Performance heatmap across samples
    im = ax4.imshow(all_mae, aspect='equal', cmap='viridis')
    ax4.set_title('MAE Heatmap (Samples vs Time)', fontweight='bold')
    ax4.set_xlabel('Timestep')
    ax4.set_ylabel('Sample')
    ax4.set_xticks(range(0, 12, 2))
    ax4.set_xticklabels([f't+{(i+1)*5}min' for i in range(0, 12, 2)])
    plt.colorbar(im, ax=ax4, label='MAE')
    
    plt.tight_layout()
    
    summary_path = save_path / f'{var_name.lower()}_summary_analysis.png'
    plt.savefig(summary_path, dpi=150, bbox_inches='tight')
    print(f"Saved summary analysis: {summary_path}")
    plt.close()


def create_continuous_animation(predictions, targets, sample_idx=0, save_dir='bzc_plots', 
                               interval=500, figsize=(12, 6), cmap='viridis', var_name='BZC'):
    """
    Create animated visualization showing temporal evolution for continuous data.
    """
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    pred_sample = predictions[sample_idx, :, :, :, 0]
    target_sample = targets[sample_idx, :, :, :, 0]

    # Reverse for better visualization
    pred_sample = pred_sample[::-1]
    target_sample = target_sample[::-1]
    
    # Determine common color scale
    vmin = min(pred_sample.min(), target_sample.min())
    vmax = max(pred_sample.max(), target_sample.max())
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Initial plots
    im1 = ax1.imshow(target_sample[0], cmap=cmap, vmin=vmin, vmax=vmax)
    ax1.set_xticks([])
    ax1.set_yticks([])
    
    im2 = ax2.imshow(pred_sample[0], cmap=cmap, vmin=vmin, vmax=vmax)
    ax2.set_xticks([])
    ax2.set_yticks([])
    
    # Add colorbar
    cbar = plt.colorbar(im1, ax=[ax1, ax2], fraction=0.046, pad=0.04)
    cbar.set_label(var_name)
    
    # Time labels (5-minute intervals)
    # time_labels = [f't+{(i+1)*5}min' for i in range(12)]
    time_labels = [f't+{(12-i)*5}min' for i in range(12)]
    
    def animate(frame):
        im1.set_array(target_sample[frame])
        ax1.set_title(f'Ground Truth - {time_labels[frame]}')
        im2.set_array(pred_sample[frame])
        ax2.set_title(f'Predictions - {time_labels[frame]}')
        return [im1, im2]
    
    anim = animation.FuncAnimation(fig, animate, frames=12, interval=interval, 
                                 blit=True, repeat=True)
    
    # Save animation
    anim_path = save_path / f'{var_name.lower()}_animation_sample_{sample_idx}.gif'
    anim.save(anim_path, writer='pillow', fps=2)
    print(f"Saved animation: {anim_path}")
    plt.close()


def visualize_continuous_nowcasting_complete(predictions, targets, save_dir='bzc_plots', 
                                            cmap='viridis', var_name='BZC'):
    """
    Complete visualization pipeline for continuous meteorological nowcasting results.
    
    Parameters:
    -----------
    predictions : numpy.ndarray
        Predicted values with shape (batch, time, height, width, channels)
    targets : numpy.ndarray
        Target values with shape (batch, time, height, width, channels)
    save_dir : str
        Directory to save all visualizations
    cmap : str
        Colormap to use (default: 'viridis')
    var_name : str
        Name of the meteorological variable (e.g., 'BZC', 'Temperature', etc.)
    """
    print(f"Creating comprehensive {var_name} nowcasting visualizations...")
    
    # Visualize all samples
    for sample_idx in range(predictions.shape[0]):
        visualize_continuous_predictions(predictions, targets, sample_idx=sample_idx, 
                                        save_dir=save_dir, show_difference=True,
                                        cmap=cmap, var_name=var_name)
        
        # Create animation for each sample
        create_continuous_animation(predictions, targets, sample_idx=sample_idx, 
                                   save_dir=save_dir, cmap=cmap, var_name=var_name)
    
    # Create summary analysis
    create_continuous_summary_plot(predictions, targets, save_dir=save_dir, var_name=var_name)
    
    print(f"\nAll visualizations saved to: {Path(save_dir).absolute()}")


def correct_distribution_and_plot(predicted, target, methods=['quantile_mapping', 'ecdf_mapping', 'optimal_transport'], 
                                 sample_idx=0, timestamps_to_plot=3):
    """
    Apply selected distribution correction methods to all batches and visualize results for one batch.
    
    Parameters:
    -----------
    predicted : np.ndarray
        Predicted values with shape (8, 12, 256, 256, 1)
    target : np.ndarray
        Target values with shape (8, 12, 256, 256, 1)
    methods : list
        List of correction methods to apply. Options:
        ['quantile_mapping', 'ecdf_mapping', 'optimal_transport']
    batch_idx : int
        Index of the batch to plot (0-7). Corrections are applied to all batches.
    timestamps_to_plot : int
        Number of timestamps to plot (default: 3)
    
    Returns:
    --------
    dict : Dictionary containing corrected values for selected methods
           Keys: method names from the methods list
           Values: np.ndarray with shape (8, 12, 256, 256, 1)
    """
    
    # Validate methods
    valid_methods = ['quantile_mapping', 'ecdf_mapping', 'optimal_transport']
    for method in methods:
        if method not in valid_methods:
            raise ValueError(f"Invalid method: {method}. Valid options: {valid_methods}")
    
    # Get data shape
    n_batches, n_timestamps, height, width, channels = predicted.shape
    
    print(f"Processing all {n_batches} batches...")
    print(f"Applying methods: {methods}")
    print(f"Will plot results for batch {sample_idx + 1}")
    
    # Initialize output dictionary
    results = {}
    for method in methods:
        results[method] = np.zeros_like(predicted)
    
    # Define correction functions
    def quantile_mapping(pred_ts, target_ts, n_quantiles=100):
        """Standard quantile mapping for climate/weather bias correction"""
        quantiles = np.linspace(0, 1, n_quantiles)
        target_quantiles = np.quantile(target_ts.flatten(), quantiles)
        pred_quantiles = np.quantile(pred_ts.flatten(), quantiles)
        
        if len(np.unique(pred_quantiles)) < 2:
            return pred_ts
        
        corrected = np.interp(pred_ts.flatten(), 
                             pred_quantiles, 
                             target_quantiles)
        corrected = np.clip(corrected, target_ts.min(), target_ts.max())
        return corrected.reshape(pred_ts.shape)
    
    def ecdf_mapping(pred_ts, target_ts):
        """eCDF mapping for more precision with non-normal distributions"""
        target_flat = target_ts.flatten()
        pred_flat = pred_ts.flatten()
        
        target_sorted = np.sort(target_flat)
        target_ecdf = np.arange(1, len(target_sorted) + 1) / len(target_sorted)
        
        corrected = np.zeros_like(pred_flat)
        
        for i, val in enumerate(pred_flat):
            percentile = stats.percentileofscore(pred_flat, val, kind='mean') / 100
            idx = np.searchsorted(target_ecdf, percentile)
            idx = min(idx, len(target_sorted) - 1)
            corrected[i] = target_sorted[idx]
        
        return corrected.reshape(pred_ts.shape)
    
    def optimal_transport_mapping(pred_ts, target_ts):
        """Optimal transport to preserve spatial correlations"""
        pred_flat = pred_ts.flatten()
        target_flat = target_ts.flatten()
        
        # Sample if too large
        max_samples = 5000
        if len(pred_flat) > max_samples:
            indices_pred = np.random.choice(len(pred_flat), max_samples, replace=False)
            indices_target = np.random.choice(len(target_flat), max_samples, replace=False)
            pred_sample = pred_flat[indices_pred]
            target_sample = target_flat[indices_target]
        else:
            pred_sample = pred_flat
            target_sample = target_flat
            
        pred_sorted = np.sort(pred_sample)
        target_sorted = np.sort(target_sample)
        
        n_pred = len(pred_sorted)
        n_target = len(target_sorted)
        a = np.ones(n_pred) / n_pred
        b = np.ones(n_target) / n_target
        
        pred_sorted_2d = pred_sorted.reshape(-1, 1)
        target_sorted_2d = target_sorted.reshape(-1, 1)
        M = ot.dist(pred_sorted_2d, target_sorted_2d, metric='sqeuclidean')
        
        transport_plan = ot.emd(a, b, M)
        
        transported = np.zeros_like(pred_flat)
        
        for i, val in enumerate(pred_flat):
            closest_idx = np.argmin(np.abs(pred_sorted - val))
            target_weights = transport_plan[closest_idx, :]
            transported[i] = np.sum(target_sorted * target_weights) * n_pred
        
        transported = np.clip(transported, target_ts.min(), target_ts.max())
        return transported.reshape(pred_ts.shape)
    
    # Map method names to functions
    method_functions = {
        'quantile_mapping': quantile_mapping,
        'ecdf_mapping': ecdf_mapping,
        'optimal_transport': optimal_transport_mapping
    }
    
    # Apply corrections to ALL batches
    for b in range(n_batches):
        print(f"\nProcessing batch {b+1}/{n_batches}")
        
        # Extract batch and squeeze for processing
        pred_batch = predicted[b, :, :, :, 0]  # Shape: (12, 256, 256)
        target_batch = target[b, :, :, :, 0]    # Shape: (12, 256, 256)
        
        # Apply each selected method to each timestamp
        for method in methods:
            corrected_batch = np.zeros_like(pred_batch)
            
            for t in range(n_timestamps):
                print(f"  Applying {method} to timestamp {t+1}/{n_timestamps}", end='\r')
                corrected_batch[t] = method_functions[method](pred_batch[t], target_batch[t])
            
            # Store with original shape (add channel dimension back)
            results[method][b, :, :, :, 0] = corrected_batch
    
    print("\n\nAll corrections complete!")
    
    # Extract data for plotting (only for the selected batch)
    plot_predicted = predicted[sample_idx, :, :, :, 0]  # Shape: (12, 256, 256)
    plot_target = target[sample_idx, :, :, :, 0]        # Shape: (12, 256, 256)
    plot_corrected = {}
    for method in methods:
        plot_corrected[method] = results[method][sample_idx, :, :, :, 0]
    
    # Create visualization for the selected batch
    create_comparison_plot(plot_target, plot_predicted, plot_corrected, 
                          methods, timestamps_to_plot, sample_idx)
    
    return results

def create_comparison_plot(target, predicted, corrected_dict, methods, 
                           timestamps_to_plot, batch_idx):
    """
    Create visualization comparing selected methods for one batch
    """
    # Setup figure
    n_cols = 2 + len(methods) + 1  # target, original, methods, distributions
    fig = plt.figure(figsize=(3.5 * n_cols, 4 * timestamps_to_plot + 2))
    
    # Create grid spec
    import matplotlib.gridspec as gridspec
    gs = gridspec.GridSpec(timestamps_to_plot + 1, n_cols, 
                          height_ratios=[1]*timestamps_to_plot + [0.3],
                          hspace=0.3, wspace=0.3)
    
    # Determine common color scale
    all_data = [target, predicted] + list(corrected_dict.values())
    vmin = min(d.min() for d in all_data)
    vmax = max(d.max() for d in all_data)
    
    # Method display names
    method_names = {
        'quantile_mapping': 'Quantile Mapping',
        'ecdf_mapping': 'eCDF Mapping',
        'optimal_transport': 'Optimal Transport'
    }
    
    # Plot spatial maps for each timestamp
    for t in range(timestamps_to_plot):
        col_idx = 0
        
        # Target
        ax = plt.subplot(gs[t, col_idx])
        im = ax.imshow(target[t], cmap='viridis', vmin=vmin, vmax=vmax)
        ax.set_title(f'Target (t={t+1})', fontsize=10, fontweight='bold')
        ax.axis('off')
        col_idx += 1
        
        # Original Predicted
        ax = plt.subplot(gs[t, col_idx])
        ax.imshow(predicted[t], cmap='viridis', vmin=vmin, vmax=vmax)
        ax.set_title(f'Original (t={t+1})', fontsize=10)
        ax.axis('off')
        col_idx += 1
        
        # Corrected methods
        for method in methods:
            ax = plt.subplot(gs[t, col_idx])
            ax.imshow(corrected_dict[method][t], cmap='viridis', vmin=vmin, vmax=vmax)
            ax.set_title(f'{method_names[method]} (t={t+1})', fontsize=10)
            ax.axis('off')
            col_idx += 1
        
        # Value distributions
        ax = plt.subplot(gs[t, col_idx])
        
        # Plot distributions
        ax.hist(target[t].flatten(), bins=50, alpha=0.3, label='Target', 
                density=True, color='black')
        ax.hist(predicted[t].flatten(), bins=50, alpha=0.3, label='Original', 
                density=True, color='red')
        
        # Color map for methods
        colors = {'quantile_mapping': 'blue', 'ecdf_mapping': 'green', 
                 'optimal_transport': 'orange'}
        
        for method in methods:
            ax.hist(corrected_dict[method][t].flatten(), bins=50, alpha=0.3, 
                   label=method_names[method], density=True, color=colors[method])
        
        ax.set_xlabel('BZC Values', fontsize=9)
        ax.set_ylabel('Density', fontsize=9)
        ax.set_title(f'Distributions (t={t+1})', fontsize=10)
        ax.legend(fontsize=7, loc='upper right')
        ax.grid(True, alpha=0.3)
    
    # Add colorbar at the bottom
    cbar_ax = plt.subplot(gs[-1, :-1])
    plt.colorbar(im, cax=cbar_ax, orientation='horizontal', 
                 label='BZC Values', pad=0.1)
    
    # Add metrics in the bottom right
    ax_metrics = plt.subplot(gs[-1, -1])
    ax_metrics.axis('off')
    
    # Calculate RMSE for each method
    metrics_text = f"Batch {batch_idx + 1} RMSE:\n"
    rmse_original = np.sqrt(np.mean((predicted - target)**2))
    metrics_text += f"Original: {rmse_original:.4f}\n"
    
    for method in methods:
        rmse = np.sqrt(np.mean((corrected_dict[method] - target)**2))
        metrics_text += f"{method_names[method]}: {rmse:.4f}\n"
    
    ax_metrics.text(0.1, 0.5, metrics_text, fontsize=10, 
                   transform=ax_metrics.transAxes, verticalalignment='center')
    
    methods_str = ', '.join([method_names[m] for m in methods])
    plt.suptitle(f'Distribution Corrections - Batch {batch_idx + 1} | Methods: {methods_str}', 
                fontsize=14, fontweight='bold', y=0.98)
    
    plt.tight_layout()
    plt.show()
    
