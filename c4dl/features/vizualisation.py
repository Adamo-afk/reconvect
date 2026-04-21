import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os
from time import sleep


def plot_variable_patches_comparison(
        variable_name_1, variable_name_2, timestamp, 
        base_dir=".", save_dir=None, figsize=None, dpi=150, max_patches_per_row=10
):
    """
    Plot patches from two variables side by side for comparison, reading from single files containing all patches.
    
    Args:
        variable_name_1 (str): Name of the first variable (e.g., 'ctth_alti')
        variable_name_2 (str): Name of the second variable (e.g., 'ctth_tempe')  
        timestamp (str): Timestamp in HHMM format (e.g., '0000', '0015')
        base_dir (str, optional): Base directory containing the patch folders. Defaults to "."
        save_dir (str, optional): Directory to save the plot. If None, saves in base_dir
        figsize (tuple, optional): Figure size (width, height). Auto-calculated if None
        dpi (int, optional): DPI for saved figure. Defaults to 150
        max_patches_per_row (int, optional): Maximum patches per row before wrapping. Defaults to 10
    
    Returns:
        dict: Information about the plotting process:
            - 'output_file': Path to saved plot
            - 'num_patches_var1': Number of patches for variable 1
            - 'num_patches_var2': Number of patches for variable 2
            - 'total_patches': Total number of patches plotted
            - 'var1_range': Data range for variable 1
            - 'var2_range': Data range for variable 2
            - 'figure_size': Actual figure size used
            - 'patches_per_row': Number of patches per row
            - 'total_rows': Total number of rows in the plot
            - 'rows_var1': Number of rows for variable 1
            - 'rows_var2': Number of rows for variable 2
    
    Raises:
        FileNotFoundError: If patch files for either variable are not found
        ValueError: If patches have different counts or invalid data
    
    Examples:
        >>> # Plot different numbers of patches (e.g., 392 altitude vs 156 temperature patches)
        >>> result = plot_variable_patches_comparison('ctth_alti', 'ctth_tempe', '0000')
        >>> print(f"Plot saved to: {result['output_file']}")
        >>> print(f"Variable 1: {result['num_patches_var1']} patches, Variable 2: {result['num_patches_var2']} patches")
        
        >>> # Limit patches per row for better visualization of many patches
        >>> result = plot_variable_patches_comparison(
        ...     'ctth_alti', 'cmic_cot', '0015', 
        ...     max_patches_per_row=8
        ... )
    """
    
    print(f"Creating comparison plot for {variable_name_1} vs {variable_name_2} at timestamp {timestamp}")
    
    # Validate timestamp format
    if not isinstance(timestamp, str) or len(timestamp) != 4 or not timestamp.isdigit():
        raise ValueError("Timestamp must be a 4-digit string (HHMM format)")
    
    base_path = Path(base_dir)

    # Get current working directory
    crt_dir = Path.cwd()
    print(f"Current working directory: {crt_dir}")

    # Get all directories from the current working directory
    all_dirs = [d for d in crt_dir.iterdir() if d.is_dir()]
    print(f"Found directories: {all_dirs}")

    # Split variable names (use first part before any hyphen)
    variable_name_1 = variable_name_1.split('-')[0]  
    variable_name_2 = variable_name_2.split('-')[0] 
    print(f"Using variable names: {variable_name_1}, {variable_name_2}")
    sleep(5)
    
    # Construct paths to the patch files
    for dir in all_dirs:
        if variable_name_1 in dir.name:
            var1_folder = base_path / dir
        elif variable_name_2 in dir.name:
            var2_folder = base_path / dir
    
    for filename in os.listdir(var1_folder):
        if timestamp in filename:
            var1_file = var1_folder / filename
            break

    for filename in os.listdir(var2_folder):
        if timestamp in filename:
            var2_file = var2_folder / filename
            break
    
    # Check if files exist
    if not var1_file.exists():
        raise FileNotFoundError(f"Patch file not found: {var1_file}")
    
    if not var2_file.exists():
        raise FileNotFoundError(f"Patch file not found: {var2_file}")
    
    print(f"Loading patches from:")
    print(f"  Variable 1: {var1_file}")
    print(f"  Variable 2: {var2_file}")
    
    # Load patches
    try:
        patches_var1 = np.load(var1_file)
        patches_var2 = np.load(var2_file)
    except Exception as e:
        raise RuntimeError(f"Error loading patch files: {e}")
    
    print(f"Loaded patches:")
    print(f"  {variable_name_1}: {patches_var1.shape}")
    print(f"  {variable_name_2}: {patches_var2.shape}")
    
    # Get patch counts for each variable
    num_patches_var1 = patches_var1.shape[0]
    num_patches_var2 = patches_var2.shape[0]
    max_patches = max(num_patches_var1, num_patches_var2)
    
    if num_patches_var1 == 0 and num_patches_var2 == 0:
        raise ValueError("No patches found in either file")
    
    print(f"Patch counts: {variable_name_1}={num_patches_var1}, {variable_name_2}={num_patches_var2}")
    
    # Calculate data ranges for consistent scaling
    var1_valid = patches_var1[~np.isnan(patches_var1)]
    var2_valid = patches_var2[~np.isnan(patches_var2)]
    
    var1_range = (np.min(var1_valid), np.max(var1_valid)) if len(var1_valid) > 0 else (0, 1)
    var2_range = (np.min(var2_valid), np.max(var2_valid)) if len(var2_valid) > 0 else (0, 1)
    
    print(f"Data ranges:")
    print(f"  {variable_name_1}: {var1_range[0]:.2f} to {var1_range[1]:.2f}")
    print(f"  {variable_name_2}: {var2_range[0]:.2f} to {var2_range[1]:.2f}")
    
    # Calculate optimal layout for potentially different patch counts
    patches_per_row = min(max_patches, max_patches_per_row)
    num_rows_var1 = max(1, (num_patches_var1 + patches_per_row - 1) // patches_per_row) if num_patches_var1 > 0 else 0
    num_rows_var2 = max(1, (num_patches_var2 + patches_per_row - 1) // patches_per_row) if num_patches_var2 > 0 else 0
    total_rows = num_rows_var1 + num_rows_var2
    
    print(f"Plot layout:")
    print(f"  Patches per row: {patches_per_row}")
    print(f"  {variable_name_1}: {num_rows_var1} rows ({num_patches_var1} patches)")
    print(f"  {variable_name_2}: {num_rows_var2} rows ({num_patches_var2} patches)")
    print(f"  Total rows: {total_rows}")
    
    # Calculate figure size if not provided
    if figsize is None:
        patch_width = 2.5  # inches per patch
        patch_height = 2.0  # inches per patch row
        width = min(patch_width * patches_per_row, 25)  # Cap at 25 inches
        height = patch_height * total_rows + 3  # Add space for titles and margins
        figsize = (width, height)
    
    print(f"Creating figure with size: {figsize}")
    
    # Create figure and subplots
    fig, axes = plt.subplots(total_rows, patches_per_row, figsize=figsize, facecolor='white')
    
    # Handle different subplot configurations
    if total_rows == 1 and patches_per_row == 1:
        axes = np.array([[axes]])
    elif total_rows == 1:
        axes = axes.reshape(1, -1)
    elif patches_per_row == 1:
        axes = axes.reshape(-1, 1)
    
    # Get appropriate colormaps for each variable
    cmap1 = _get_colormap_for_variable(variable_name_1)
    cmap2 = _get_colormap_for_variable(variable_name_2)
    
    current_row = 0
    
    # Plot patches for variable 1 (first section)
    if num_patches_var1 > 0:
        for i in range(num_patches_var1):
            row = current_row + (i // patches_per_row)
            col = i % patches_per_row
            
            ax = axes[row, col]
            
            patch = patches_var1[i]
            
            # Handle different patch dimensions
            if len(patch.shape) == 3 and patch.shape[2] == 1:
                patch = patch[:, :, 0]  # Remove singleton dimension
            elif len(patch.shape) == 3:
                patch = patch[:, :, 0]  # Use first channel
            
            im1 = ax.imshow(patch, cmap=cmap1, vmin=var1_range[0], vmax=var1_range[1], aspect='equal')
            ax.set_title(f'{variable_name_1}\nPatch {i+1}', fontsize=9, pad=3)
            ax.set_xticks([])
            ax.set_yticks([])
            
            # Add colorbar for each patch
            cbar1 = plt.colorbar(im1, ax=ax, fraction=0.046, pad=0.04)
            cbar1.ax.tick_params(labelsize=7)
        
        # Hide unused subplots in first variable section
        for i in range(num_patches_var1, num_rows_var1 * patches_per_row):
            row = current_row + (i // patches_per_row)
            col = i % patches_per_row
            if row < current_row + num_rows_var1:
                axes[row, col].set_visible(False)
        
        current_row += num_rows_var1
    
    # Plot patches for variable 2 (second section)
    if num_patches_var2 > 0:
        for i in range(num_patches_var2):
            row = current_row + (i // patches_per_row)
            col = i % patches_per_row
            
            ax = axes[row, col]
            
            patch = patches_var2[i]
            
            # Handle different patch dimensions
            if len(patch.shape) == 3 and patch.shape[2] == 1:
                patch = patch[:, :, 0]  # Remove singleton dimension
            elif len(patch.shape) == 3:
                patch = patch[:, :, 0]
            
            im2 = ax.imshow(patch, cmap=cmap2, vmin=var2_range[0], vmax=var2_range[1], aspect='equal')
            ax.set_title(f'{variable_name_2}\nPatch {i+1}', fontsize=9, pad=3)
            ax.set_xticks([])
            ax.set_yticks([])
            
            # Add colorbar for each patch
            cbar2 = plt.colorbar(im2, ax=ax, fraction=0.046, pad=0.04)
            cbar2.ax.tick_params(labelsize=7)
        
        # Hide unused subplots in second variable section
        for i in range(num_patches_var2, num_rows_var2 * patches_per_row):
            row = current_row + (i // patches_per_row)
            col = i % patches_per_row
            if row < current_row + num_rows_var2:
                axes[row, col].set_visible(False)
    
    # Add main title
    time_formatted = f"{timestamp[:2]}:{timestamp[2:]}"
    main_title = f'{variable_name_1} ({num_patches_var1}) vs {variable_name_2} ({num_patches_var2}) - Patches at {time_formatted}'
    
    fig.suptitle(main_title, fontsize=12, fontweight='bold', y=0.98)
    
    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(top=0.95, hspace=0.4, wspace=0.3)
    
    # Determine save directory
    if save_dir is None:
        save_dir = base_dir
    
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    # Create output filename
    output_filename = f"{variable_name_1}_{variable_name_2}_{timestamp}_patches.png"
    output_path = save_path / output_filename
    
    # Save figure
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()  # Close to free memory
    
    # Calculate file size
    file_size_mb = output_path.stat().st_size / 1024 / 1024
    
    print(f"✓ Plot saved to: {output_path}")
    print(f"  File size: {file_size_mb:.2f} MB")
    print(f"  Figure size: {figsize}")
    print(f"  DPI: {dpi}")
    
    return {
        'output_file': str(output_path),
        'num_patches_var1': num_patches_var1,
        'num_patches_var2': num_patches_var2,
        'total_patches': num_patches_var1 + num_patches_var2,
        'var1_range': var1_range,
        'var2_range': var2_range,
        'figure_size': figsize,
        'file_size_mb': file_size_mb,
        'patches_per_row': patches_per_row,
        'total_rows': total_rows,
        'rows_var1': num_rows_var1,
        'rows_var2': num_rows_var2
    }


def _get_colormap_for_variable(variable_name):
    """
    Get appropriate colormap for different meteorological variables.
    
    Args:
        variable_name (str): Name of the variable
    
    Returns:
        str: Matplotlib colormap name
    """
    
    colormap_mapping = {
        # Cloud Top Height - terrain-like colors
        'ctth_alti': 'terrain',
        'altitude': 'terrain',
        'height': 'terrain',
        
        # Temperature - thermal colors
        'ctth_tempe': 'coolwarm',
        'temperature': 'coolwarm',
        'temp': 'coolwarm',
        
        # Cloud Optical Thickness - density colors
        'cmic_cot': 'viridis',
        'optical_thickness': 'viridis',
        'cot': 'viridis',
        
        # Cloud Phase - categorical colors
        'cmic_phase': 'Set1',
        'phase': 'Set1',
        
        # Cloud Mask - binary colors
        'cma_mask': 'gray',
        'mask': 'gray',
        
        # Default
        'default': 'viridis'
    }
    
    # Check for exact match first
    if variable_name in colormap_mapping:
        return colormap_mapping[variable_name]
    
    # Check for partial matches
    variable_lower = variable_name.lower()
    for key, cmap in colormap_mapping.items():
        if key in variable_lower:
            return cmap
    
    return colormap_mapping['default']