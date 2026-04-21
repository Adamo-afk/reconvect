import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path

def plot_training_history(history, save_dir='plots', figsize=(12, 8), dpi=300):
    """
    Plot training history metrics and save them to a specified directory.
    
    Parameters:
    -----------
    history : tf.keras.callbacks.History or dict
        Training history object from model.fit() or dictionary with metric values
    save_dir : str, default='plots'
        Directory to save the plots
    figsize : tuple, default=(12, 8)
        Figure size for each plot
    dpi : int, default=300
        Resolution for saved plots
    """
    
    # Create plots directory if it doesn't exist
    plots_path = Path(save_dir)
    plots_path.mkdir(parents=True, exist_ok=True)
    
    # Extract history data
    if hasattr(history, 'history'):
        hist_dict = history.history
    else:
        hist_dict = history
    
    # Define metrics to plot
    metrics_to_plot = {
        'loss': 'Loss',
        'binary_accuracy': 'Binary Accuracy', 
        'true_neg': 'True Negatives',
        'false_neg': 'False Negatives',
        'true_pos': 'True Positives',
        'false_pos': 'False Positives',
        'iou_metric': 'IoU Metric',
        'dice_metric': 'Dice Metric'
    }
    
    # Set up the plotting style
    plt.style.use('default')
    
    epochs = range(1, len(list(hist_dict.values())[0]) + 1)
    
    # Plot each metric
    for metric_key, metric_name in metrics_to_plot.items():
        if metric_key in hist_dict:
            
            fig, ax = plt.subplots(figsize=figsize)
            
            # Plot training metric
            train_values = hist_dict[metric_key]
            ax.plot(epochs, train_values, 'b-', linewidth=2, label=f'Training {metric_name}')
            
            # Plot validation metric if available
            val_key = f'val_{metric_key}'
            if val_key in hist_dict:
                val_values = hist_dict[val_key]
                ax.plot(epochs, val_values, 'r-', linewidth=2, label=f'Validation {metric_name}')
            
            # Customize plot
            ax.set_title(f'{metric_name} Over Epochs', fontsize=16, fontweight='bold')
            ax.set_xlabel('Epoch', fontsize=12)
            ax.set_ylabel(metric_name, fontsize=12)
            ax.legend(fontsize=12)
            ax.grid(True, alpha=0.3)
            
            # Set appropriate y-axis limits based on metric type
            if metric_key == 'loss':
                ax.set_ylim(bottom=0)
            elif 'accuracy' in metric_key:
                ax.set_ylim(0, 1.05)
            elif metric_key in ['iou_metric', 'dice_metric']:
                ax.set_ylim(0, 1.05)
            
            # Add value annotations for key points
            if len(train_values) > 0:
                # Annotate final training value
                final_train = train_values[-1]
                ax.annotate(f'{final_train:.4f}', 
                           xy=(len(train_values), final_train),
                           xytext=(5, 5), textcoords='offset points',
                           fontsize=10, ha='left')
                
                # Annotate final validation value if available
                if val_key in hist_dict and len(hist_dict[val_key]) > 0:
                    final_val = hist_dict[val_key][-1]
                    ax.annotate(f'{final_val:.4f}', 
                               xy=(len(hist_dict[val_key]), final_val),
                               xytext=(5, -15), textcoords='offset points',
                               fontsize=10, ha='left', color='red')
            
            # Save plot
            plt.tight_layout()
            plot_filename = plots_path / f'{metric_key}_history.png'
            plt.savefig(plot_filename, dpi=dpi, bbox_inches='tight')
            print(f"Saved: {plot_filename}")
            
            plt.close(fig)
    
    # Create a comprehensive overview plot
    create_overview_plot(hist_dict, plots_path, figsize, dpi)
    
    print(f"\nAll plots saved to: {plots_path.absolute()}")

def create_overview_plot(hist_dict, plots_path, figsize, dpi):
    """Create a comprehensive overview plot with multiple metrics."""
    
    # Select key metrics for overview
    overview_metrics = []
    if 'loss' in hist_dict:
        overview_metrics.append(('loss', 'Loss'))
    if 'binary_accuracy' in hist_dict:
        overview_metrics.append(('binary_accuracy', 'Binary Accuracy'))
    if 'dice_metric' in hist_dict:
        overview_metrics.append(('dice_metric', 'Dice Metric'))
    if 'iou_metric' in hist_dict:
        overview_metrics.append(('iou_metric', 'IoU Metric'))
    
    if len(overview_metrics) == 0:
        return
    
    # Calculate subplot layout
    n_metrics = len(overview_metrics)
    n_cols = 2
    n_rows = (n_metrics + 1) // 2
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(figsize[0]*1.5, figsize[1]*n_rows/2))
    if n_metrics == 1:
        axes = [axes]
    elif n_rows == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    epochs = range(1, len(list(hist_dict.values())[0]) + 1)
    
    for idx, (metric_key, metric_name) in enumerate(overview_metrics):
        ax = axes[idx]
        
        # Plot training metric
        train_values = hist_dict[metric_key]
        ax.plot(epochs, train_values, 'b-', linewidth=2, label=f'Training')
        
        # Plot validation metric if available
        val_key = f'val_{metric_key}'
        if val_key in hist_dict:
            val_values = hist_dict[val_key]
            ax.plot(epochs, val_values, 'r-', linewidth=2, label=f'Validation')
        
        ax.set_title(metric_name, fontsize=12, fontweight='bold')
        ax.set_xlabel('Epoch', fontsize=10)
        ax.set_ylabel(metric_name, fontsize=10)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        
        # Set appropriate y-axis limits
        if metric_key == 'loss':
            ax.set_ylim(bottom=0)
        elif 'accuracy' in metric_key or 'metric' in metric_key:
            ax.set_ylim(0, 1.05)
    
    # Hide unused subplots
    for idx in range(len(overview_metrics), len(axes)):
        axes[idx].set_visible(False)
    
    plt.suptitle('Training History Overview', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    overview_filename = plots_path / 'training_overview.png'
    plt.savefig(overview_filename, dpi=dpi, bbox_inches='tight')
    print(f"Saved: {overview_filename}")
    plt.close(fig)

def plot_confusion_matrix_metrics(history, save_dir='plots', figsize=(10, 8), dpi=300):
    """
    Create a specialized plot for confusion matrix metrics.
    """
    plots_path = Path(save_dir)
    plots_path.mkdir(parents=True, exist_ok=True)
    
    if hasattr(history, 'history'):
        hist_dict = history.history
    else:
        hist_dict = history
    
    # Confusion matrix metrics
    cm_metrics = ['true_pos', 'true_neg', 'false_pos', 'false_neg']
    available_metrics = [m for m in cm_metrics if m in hist_dict]
    
    if not available_metrics:
        return
    
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    axes = axes.flatten()
    
    epochs = range(1, len(list(hist_dict.values())[0]) + 1)
    colors = ['green', 'blue', 'orange', 'red']
    
    for idx, metric in enumerate(available_metrics):
        ax = axes[idx]
        
        # Training values
        train_values = hist_dict[metric]
        ax.plot(epochs, train_values, color=colors[idx], linewidth=2, label='Training')
        
        # Validation values if available
        val_key = f'val_{metric}'
        if val_key in hist_dict:
            val_values = hist_dict[val_key]
            ax.plot(epochs, val_values, color=colors[idx], linestyle='--', 
                   linewidth=2, alpha=0.7, label='Validation')
        
        ax.set_title(metric.replace('_', ' ').title(), fontsize=12, fontweight='bold')
        ax.set_xlabel('Epoch', fontsize=10)
        ax.set_ylabel('Rate', fontsize=10)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(bottom=0)
    
    # Hide unused subplots
    for idx in range(len(available_metrics), 4):
        axes[idx].set_visible(False)
    
    plt.suptitle('Confusion Matrix Metrics', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    cm_filename = plots_path / 'confusion_matrix_metrics.png'
    plt.savefig(cm_filename, dpi=dpi, bbox_inches='tight')
    print(f"Saved: {cm_filename}")
    plt.close(fig)
