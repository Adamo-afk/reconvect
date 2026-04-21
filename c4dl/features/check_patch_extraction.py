from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


def plot_patches_on_blank(datamap, patches_dict, datetime_str, figsize=(20, 6)):
    """
    Extract patches from datamap and place them on a blank image.
    Analyzes patch patterns and draws black borders around expected 256×256 patch regions.
    
    Parameters:
    -----------
    datamap : np.ndarray
        The full data array to extract patches from
    patches_dict : dict
        Dictionary with timestamps as keys and list of lists of patch coordinates as values
        Format: {timestamp: [[(r1,c1,r2,c2), ...], [(r1,c1,r2,c2), ...], ...]}
    datetime_str : str
        Date and time in format "yyyy-mm-dd hh:mm"
    figsize : tuple
        Figure size for the plot
    
    Returns:
    --------
    blank_image : np.ndarray
        Image with only the extracted patches
    fig, axes : matplotlib objects
        Figure and axes for further customization if needed
    """
    
    
    # Convert datetime string to ISO format with nanoseconds
    target_dt = datetime.strptime(datetime_str, "%Y-%m-%d %H:%M")
    iso_format = target_dt.isoformat() + '.000000000'
    
    # Find patches in dictionary
    if iso_format not in patches_dict:
        print(f"No patches found for timestamp {iso_format}")
        print(f"Available timestamps: {list(patches_dict.keys())[:5]}...")
        return None, None, None
    
    patches_list = patches_dict[iso_format]
    
    print(f"Found timestamp: {iso_format}")
    print(f"Number of patch lists (256×256 patches): {len(patches_list)}")
    print("="*60)
    
    # Create blank image (same shape as datamap, filled with NaN)
    blank_image = np.full_like(datamap, np.nan, dtype=float)
    
    # Store information about 256×256 patch regions
    large_patch_regions = []
    
    # Extract and place patches
    total_patches = 0
    for list_idx, patch_list in enumerate(patches_list):
        print(f"\nPatch list {list_idx}: Number of 32×32 patches: {len(patch_list)}")
        
        if len(patch_list) == 0:
            continue
        
        # Analyze the pattern of patches in this list
        rows = set()
        cols = set()
        for patch_coords in patch_list:
            r1, c1, r2, c2 = patch_coords
            rows.add(r1)
            cols.add(c1)
        
        rows_sorted = sorted(rows)
        cols_sorted = sorted(cols)
        
        # Determine the expected 256×256 patch bounds (FULL SIZE)
        r_min = min(rows)
        r_max = r_min + 256  # ALWAYS 256 pixels, regardless of missing patches
        c_min = min(cols)
        c_max = c_min + 256  # ALWAYS 256 pixels, regardless of missing patches
        
        # Expected dimensions
        expected_rows = 8  # 256 / 32
        expected_cols = 8  # 256 / 32
        actual_rows = len(rows_sorted)
        actual_cols = len(cols_sorted)
        
        print(f"  Expected grid: {expected_rows}×{expected_cols} = 64 patches")
        print(f"  Actual grid: {actual_rows}×{actual_cols} = {len(patch_list)} patches")
        print(f"  Full 256×256 bounds: rows [{r_min}, {r_max}), cols [{c_min}, {c_max})")
        
        # Analyze missing patches
        # Expected row starts: r_min, r_min+32, ..., r_min+224 (8 positions)
        expected_row_starts = [r_min + i*32 for i in range(expected_rows)]
        expected_col_starts = [c_min + i*32 for i in range(expected_cols)]
        
        missing_rows = set(expected_row_starts) - rows
        missing_cols = set(expected_col_starts) - cols
        
        if missing_rows:
            print(f"  Missing row starts: {sorted(missing_rows)}")
        if missing_cols:
            print(f"  Missing column starts: {sorted(missing_cols)}")
        
        # Identify specifically which patches are missing
        missing_patches = []
        for r_start in expected_row_starts:
            for c_start in expected_col_starts:
                if r_start not in rows or c_start not in cols:
                    missing_patches.append((r_start, c_start))
        
        if missing_patches:
            print(f"  Total missing patches: {len(missing_patches)}")
            print(f"  First few missing: {missing_patches[:5]}")
        
        # Store the 256×256 region info
        large_patch_regions.append({
            'r_min': r_min,
            'r_max': r_max,
            'c_min': c_min,
            'c_max': c_max,
            'expected': expected_rows * expected_cols,
            'actual': len(patch_list),
            'missing': len(missing_patches)
        })
        
        # Extract and place patches on blank image
        for patch_coords in patch_list:
            r1, c1, r2, c2 = patch_coords
            
            # Extract 32x32 patch from datamap
            patch_data = datamap[r1:r2+1, c1:c2+1]
            
            # Place patch on blank image at the same location
            blank_image[r1:r2+1, c1:c2+1] = patch_data
            total_patches += 1
    
    print(f"\n{'='*60}")
    print(f"Total patches extracted: {total_patches}")
    
    # Create comparison plot with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    # 1. Original datamap
    im1 = axes[0].imshow(datamap, cmap='viridis', aspect='equal')
    axes[0].set_title(f'Original Data\n{datetime_str}')
    axes[0].set_xlabel('X')
    axes[0].set_ylabel('Y')
    plt.colorbar(im1, ax=axes[0], label='Value')
    
    # 2. Blank image with extracted patches AND black borders
    im2 = axes[1].imshow(blank_image, cmap='viridis', aspect='equal')
    axes[1].set_title(f'Extracted Patches Only\n{total_patches} patches (32×32)')
    axes[1].set_xlabel('X')
    axes[1].set_ylabel('Y')
    plt.colorbar(im2, ax=axes[1], label='Value')
    
    # Draw black borders around each FULL 256×256 patch region
    for region in large_patch_regions:
        # Rectangle: (x, y, width, height)
        # With default origin='upper', (x, y) is top-left corner
        rect = Rectangle(
            (region['c_min'], region['r_min']),  # (x, y) - top-left
            256,  # width - always 256
            256,  # height - always 256
            linewidth=2,
            edgecolor='black',
            facecolor='none',
            linestyle='-'
        )
        axes[1].add_patch(rect)
    
    # 3. Coverage mask with black borders
    coverage_mask = ~np.isnan(blank_image)  # True where patches exist
    im3 = axes[2].imshow(coverage_mask, cmap='RdYlGn', aspect='equal')
    axes[2].set_title(f'Patch Coverage\n(Green = extracted, Red = missing)')
    axes[2].set_xlabel('X')
    axes[2].set_ylabel('Y')
    plt.colorbar(im3, ax=axes[2], label='Has Patch')
    
    # Draw black borders on coverage mask too
    for region in large_patch_regions:
        rect = Rectangle(
            (region['c_min'], region['r_min']),
            256,  # width - always 256
            256,  # height - always 256
            linewidth=2,
            edgecolor='black',
            facecolor='none',
            linestyle='-'
        )
        axes[2].add_patch(rect)
    
    # Calculate coverage statistics
    total_pixels = datamap.size
    covered_pixels = np.sum(coverage_mask)
    coverage_percent = (covered_pixels / total_pixels) * 100
    
    print(f"\nCoverage statistics:")
    print(f"  Total pixels: {total_pixels}")
    print(f"  Covered pixels: {covered_pixels}")
    print(f"  Coverage: {coverage_percent:.2f}%")
    
    # Print summary of 256×256 patches
    print(f"\n256×256 Patch Summary:")
    for idx, region in enumerate(large_patch_regions):
        print(f"  Patch {idx}: {region['actual']}/{region['expected']} patches "
              f"({region['missing']} missing)")
    
    plt.tight_layout()
    plt.show()
    
    return blank_image, fig, axes
