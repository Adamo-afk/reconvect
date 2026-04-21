# patch_stitching_diagnostics.py - Comprehensive diagnostic tool for patch stitching

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import os
from datetime import datetime, timedelta


class StitchingDiagnostics:
    """
    Diagnostic tool to save and visualize patch stitching details
    """
    
    def __init__(self, output_dir="stitching_diagnostics"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def save_stitching_metadata(self, var_name, time_idx, pixel_i0, pixel_j0, 
                                box_size, patch_size, found_patches, 
                                missing_patches, batch_data, timestamp):
        """
        Save detailed metadata about a single stitching operation
        
        Args:
            var_name: Variable name (e.g., 'IR-087')
            time_idx: Time index
            pixel_i0, pixel_j0: Starting pixel coordinates
            box_size: (time_dim, box_i, box_j)
            patch_size: Size of individual patches
            found_patches: List of (i, j, patch_idx, value) tuples for found patches
            missing_patches: List of (i, j) tuples for missing patches
            batch_data: The actual stitched data (numpy array)
            timestamp: Unix timestamp
        """
        metadata = {
            'var_name': var_name,
            'time_idx': int(time_idx),
            'timestamp': int(timestamp),
            'datetime': datetime.fromtimestamp(timestamp).isoformat(),
            'pixel_origin': (int(pixel_i0), int(pixel_j0)),
            'box_size': tuple(int(x) for x in box_size),
            'patch_size': int(patch_size),
            'found_patches': [(int(i), int(j), int(idx), float(val)) 
                             for i, j, idx, val in found_patches],
            'missing_patches': [(int(i), int(j)) for i, j in missing_patches],
            'stats': {
                'mean': float(np.mean(batch_data)),
                'min': float(np.min(batch_data)),
                'max': float(np.max(batch_data)),
                'std': float(np.std(batch_data)),
                'num_found': len(found_patches),
                'num_missing': len(missing_patches),
                'coverage': len(found_patches) / (len(found_patches) + len(missing_patches))
            }
        }
        
        # Save metadata
        filename = f"{self.output_dir}/{var_name}_t{time_idx}_metadata.json"
        with open(filename, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Save actual data
        data_filename = f"{self.output_dir}/{var_name}_t{time_idx}_data.npy"
        np.save(data_filename, batch_data)
    
    def visualize_stitching(self, var_name, target_hour, timeframe='past', 
                           time_indices=None):
        """
        Create interactive plotly visualization of patch stitching
        
        Args:
            var_name: Variable name to visualize
            target_hour: Target hour (0-23)
            timeframe: 'past' (6 timesteps) or 'future' (12 timesteps)
            time_indices: Optional list of specific time indices to plot
        """
        # Find matching metadata files
        all_files = [f for f in os.listdir(self.output_dir) 
                     if f.startswith(f"{var_name}_") and f.endswith("_metadata.json")]
        
        if not all_files:
            print(f"No metadata files found for {var_name}")
            return None
        
        # Load all metadata
        metadata_list = []
        for f in all_files:
            with open(os.path.join(self.output_dir, f), 'r') as file:
                metadata = json.load(file)
                dt = datetime.fromisoformat(metadata['datetime'])
                if dt.hour == target_hour:
                    metadata_list.append(metadata)
        
        if not metadata_list:
            print(f"No data found for hour {target_hour}")
            return None
        
        # Sort by time
        metadata_list.sort(key=lambda x: x['time_idx'])
        
        # Select timeframes
        n_timesteps = 6 if timeframe == 'past' else 12
        metadata_list = metadata_list[:n_timesteps]
        
        if time_indices is not None:
            metadata_list = [m for m in metadata_list if m['time_idx'] in time_indices]
        
        # Create subplots
        n_plots = len(metadata_list)
        n_cols = min(3, n_plots)
        n_rows = (n_plots + n_cols - 1) // n_cols
        
        fig = make_subplots(
            rows=n_rows, cols=n_cols,
            subplot_titles=[m['datetime'] for m in metadata_list],
            vertical_spacing=0.15,
            horizontal_spacing=0.1
        )
        
        for idx, metadata in enumerate(metadata_list):
            row = idx // n_cols + 1
            col = idx % n_cols + 1
            
            # Load actual data
            data_file = f"{self.output_dir}/{var_name}_t{metadata['time_idx']}_data.npy"
            data = np.load(data_file)
            
            # If data has time dimension, take first timestep
            if len(data.shape) > 2:
                data = data[0]
            
            # Create heatmap
            fig.add_trace(
                go.Heatmap(
                    z=data,
                    colorscale='RdBu_r',
                    showscale=(col == n_cols),
                    hovertemplate='x: %{x}<br>y: %{y}<br>value: %{z:.3f}<extra></extra>',
                ),
                row=row, col=col
            )
            
            # Add patch boundaries
            patch_size = metadata['patch_size']
            box_size = metadata['box_size']
            
            # Vertical lines
            for i in range(box_size[1] + 1):
                x_pos = i * patch_size
                fig.add_shape(
                    type="line",
                    x0=x_pos, x1=x_pos,
                    y0=0, y1=data.shape[0],
                    line=dict(color="yellow", width=1, dash="dot"),
                    row=row, col=col
                )
            
            # Horizontal lines
            for j in range(box_size[2] + 1):
                y_pos = j * patch_size
                fig.add_shape(
                    type="line",
                    x0=0, x1=data.shape[1],
                    y0=y_pos, y1=y_pos,
                    line=dict(color="yellow", width=1, dash="dot"),
                    row=row, col=col
                )
            
            # Mark found patches with green circles
            for i, j, patch_idx, val in metadata['found_patches']:
                center_x = (i + 0.5) * patch_size
                center_y = (j + 0.5) * patch_size
                fig.add_annotation(
                    x=center_x, y=center_y,
                    text=f"✓{patch_idx}",
                    showarrow=False,
                    font=dict(color="lime", size=8),
                    row=row, col=col
                )
            
            # Mark missing patches with red X
            for i, j in metadata['missing_patches']:
                center_x = (i + 0.5) * patch_size
                center_y = (j + 0.5) * patch_size
                fig.add_annotation(
                    x=center_x, y=center_y,
                    text="✗",
                    showarrow=False,
                    font=dict(color="red", size=12),
                    row=row, col=col
                )
            
            # Add statistics annotation
            stats = metadata['stats']
            stats_text = (
                f"Mean: {stats['mean']:.3f}<br>"
                f"Min: {stats['min']:.3f}<br>"
                f"Max: {stats['max']:.3f}<br>"
                f"Std: {stats['std']:.3f}<br>"
                f"Found: {stats['num_found']}/{stats['num_found'] + stats['num_missing']}<br>"
                f"Coverage: {stats['coverage']*100:.1f}%<br>"
                f"Origin: ({metadata['pixel_origin'][0]}, {metadata['pixel_origin'][1]})"
            )
            
            fig.add_annotation(
                x=0.5, y=-0.15,
                xref=f'x{idx+1} domain', yref=f'y{idx+1} domain',
                text=stats_text,
                showarrow=False,
                font=dict(size=9),
                align='left',
                row=row, col=col
            )
        
        # Update layout
        fig.update_layout(
            title=f"{var_name} Patch Stitching Diagnostics - Hour {target_hour}:00 ({timeframe})",
            height=400 * n_rows,
            showlegend=False
        )
        
        # Save
        output_file = f"{self.output_dir}/{var_name}_hour{target_hour}_{timeframe}_diagnostic.html"
        fig.write_html(output_file)
        print(f"Saved diagnostic visualization to: {output_file}")
        
        return fig


def diagnose_constant_values(batch_gen, var_name, dataset='train', sample_idx=0):
    """
    Deep dive into why a variable has constant values
    
    Args:
        batch_gen: BatchGenerator instance
        var_name: Variable name showing constant values
        dataset: 'train', 'valid', or 'test'
        sample_idx: Which sample to diagnose
    """
    print("\n" + "="*80)
    print(f"DIAGNOSING CONSTANT VALUES FOR: {var_name}")
    print("="*80)
    
    # Get the patch index for this variable
    if var_name not in batch_gen.raw_batch_index:
        print(f"ERROR: {var_name} not found in batch generator")
        return
    
    patch_idx = batch_gen.raw_batch_index[var_name]
    
    # Basic info
    print(f"\nBasic Info:")
    print(f"  Original patch size: {patch_idx.original_patch_size}")
    print(f"  Working patch size: {patch_idx.patch_size}")
    print(f"  Final output size: {patch_idx.final_output_size}")
    print(f"  Number of patches: {len(patch_idx.patch_data)}")
    
    # Check patch data
    print(f"\nPatch Data Analysis:")
    if len(patch_idx.patch_data) > 0:
        sample_patches = patch_idx.patch_data[:10]
        patch_means = [p.mean() for p in sample_patches]
        patch_stds = [p.std() for p in sample_patches]
        
        print(f"  Sample patch means: {patch_means}")
        print(f"  Sample patch stds: {patch_stds}")
        
        # Check if all patches are identical
        all_same = all(np.allclose(p, patch_idx.patch_data[0]) for p in sample_patches)
        print(f"  All sample patches identical: {all_same}")
        
        if all_same:
            print(f"  ⚠️  ALL PATCHES ARE IDENTICAL!")
            print(f"  Sample patch:\n{patch_idx.patch_data[0]}")
    
    # Check coordinate mapping
    print(f"\nCoordinate Mapping:")
    print(f"  Total mapped coordinates: {len(patch_idx.patch_pixels_i)}")
    
    if len(patch_idx.patch_pixels_i) > 0:
        unique_coords = set(zip(patch_idx.patch_pixels_i, patch_idx.patch_pixels_j))
        print(f"  Unique coordinate positions: {len(unique_coords)}")
        print(f"  Sample coordinates:")
        for i in range(min(10, len(patch_idx.patch_pixels_i))):
            print(f"    ({patch_idx.patch_pixels_i[i]}, {patch_idx.patch_pixels_j[i]}) -> patch {patch_idx.patch_indices[i]}")
    
    # Test actual stitching
    print(f"\nStitching Test:")
    t_pred = batch_gen.time_coords[dataset][sample_idx:sample_idx+1]
    i, j = batch_gen.frame_spatial_coordinates(t_pred, dataset=dataset)
    
    print(f"  Time index: {t_pred[0]}")
    print(f"  Selected origin: ({i[0]}, {j[0]})")
    
    # Do actual stitch
    result = patch_idx(t_pred, i, j, num_timesteps=1)
    print(f"  Result shape: {result.shape}")
    print(f"  Result mean: {result.mean():.3f}")
    print(f"  Result std: {result.std():.3f}")
    print(f"  Result min: {result.min():.3f}")
    print(f"  Result max: {result.max():.3f}")
    
    if result.std() < 0.01:
        print(f"  ⚠️  STITCHED OUTPUT IS CONSTANT!")
        print(f"  Constant value: {result.mean():.6f}")
    
    print("="*80)
