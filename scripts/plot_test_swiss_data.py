
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def plot_confusion_matrix_median(confusion_matrix, class_names=None, title="Confusion Matrix (Median across Thresholds)"):
    """
    Plot confusion matrix using median values across all thresholds.
    
    Parameters:
    -----------
    confusion_matrix : np.array
        Shape (2, 2, n_thresholds) - confusion matrix for each threshold
    class_names : list, optional
        Names for the classes. Default: ['Negative', 'Positive']
    title : str
        Plot title
    """
    # Calculate median across thresholds (axis=2)
    cm_median = np.median(confusion_matrix, axis=2)
    
    # Default class names if not provided
    if class_names is None:
        class_names = ['Negative', 'Positive']
    
    # Create the plot
    plt.figure(figsize=(8, 6))
    
    # Create heatmap
    sns.heatmap(cm_median, 
                annot=True, 
                fmt='.4f', 
                cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names,
                cbar_kws={'label': 'Median Value'})
    
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    
    # Add text annotations for clarity
    plt.text(0.5, -0.1, 'TN: True Negative, FP: False Positive\nFN: False Negative, TP: True Positive', 
             ha='center', va='top', transform=plt.gca().transAxes, fontsize=10, style='italic')
    
    plt.tight_layout()
    plt.show()
    
    # Print median values for reference
    print(f"Median Confusion Matrix:")
    print(f"TN (True Negative): {cm_median[0,0]:.6f}")
    print(f"FP (False Positive): {cm_median[0,1]:.6f}")
    print(f"FN (False Negative): {cm_median[1,0]:.6f}")
    print(f"TP (True Positive): {cm_median[1,1]:.6f}")


def plot_calibration_bins(calibration_values, title="Model Calibration Plot"):
    """
    Plot calibration values as a bin plot, replacing NaN values with 0.
    
    Parameters:
    -----------
    calibration_values : np.array
        Shape (n_bins,) - calibration values for each probability bin
    title : str
        Plot title
    """
    # Replace NaN values with 0
    calibration_clean = np.nan_to_num(calibration_values, nan=0.0)
    
    # Create bin centers (assuming equally spaced bins from 0 to 1)
    n_bins = len(calibration_clean)
    bin_centers = np.linspace(0, 1, n_bins)
    bin_width = 1.0 / n_bins
    
    # Create the plot
    plt.figure(figsize=(12, 6))
    
    # Bar plot
    bars = plt.bar(bin_centers, calibration_clean, 
                   width=bin_width * 0.8,  # Slightly smaller than bin width
                   alpha=0.7, 
                   color='skyblue', 
                   edgecolor='navy', 
                   linewidth=0.5)
    
    # Add perfect calibration line (diagonal)
    plt.plot([0, 1], [0, 1], 'r--', linewidth=2, label='Perfect Calibration', alpha=0.8)
    
    # Formatting
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel('Predicted Probability Bins', fontsize=12)
    plt.ylabel('Observed Frequency', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Set axis limits
    plt.xlim(0, 1)
    plt.ylim(0, max(1, np.max(calibration_clean) * 1.1))
    
    # Add statistics text
    n_valid_bins = np.sum(calibration_clean > 0)
    mean_calibration = np.mean(calibration_clean[calibration_clean > 0]) if n_valid_bins > 0 else 0
    
    stats_text = f'Valid bins: {n_valid_bins}/{n_bins}\nMean calibration: {mean_calibration:.4f}'
    plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, 
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plt.show()
    
    print(f"Calibration Statistics:")
    print(f"Total bins: {n_bins}")
    print(f"Valid bins (non-zero): {n_valid_bins}")
    print(f"NaN bins replaced: {np.sum(np.isnan(calibration_values))}")
    print(f"Mean calibration (valid bins): {mean_calibration:.6f}")
