import os
import re
from pathlib import Path
from datetime import datetime
from collections import defaultdict
import xarray as xr
import dask
from netCDF4 import Dataset
import shutil

def levenshtein_distance(s1, s2):
    """
    Calculate the Levenshtein distance between two strings
    """
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)

    if len(s2) == 0:
        return len(s1)

    previous_row = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    
    return previous_row[-1]

def find_best_variable_match(directory_name, variable_names_list, max_distance_threshold=3, fallback_strategy='use_list'):
    """
    Find the best matching variable name using Levenshtein distance
    
    Parameters:
    -----------
    directory_name : str
        Name of the directory to match
    variable_names_list : list
        List of possible variable names
    max_distance_threshold : int
        Maximum allowed Levenshtein distance for a match
    fallback_strategy : str
        'use_list' - use variable from list when no match found
        'use_directory' - use cleaned directory name when no match found
        
    Returns:
    --------
    tuple : (matched_variable_name, use_list_name_for_output)
    """
    if not variable_names_list:
        cleaned_name = clean_directory_name(directory_name)
        return cleaned_name, False
    
    # Clean the directory name first
    cleaned_dir_name = clean_directory_name(directory_name)
    
    print(f"    Matching '{directory_name}' (cleaned: '{cleaned_dir_name}') against variable list...")
    
    best_match = None
    best_distance = float('inf')
    
    # Convert to lowercase for matching
    dir_lower = cleaned_dir_name.lower()
    
    # Calculate Levenshtein distance for each variable
    for var_name in variable_names_list:
        var_lower = var_name.lower()
        distance = levenshtein_distance(dir_lower, var_lower)
        
        print(f"      '{dir_lower}' vs '{var_lower}': distance = {distance}")
        
        if distance < best_distance:
            best_distance = distance
            best_match = var_name
    
    # Check if the best match is within the threshold
    if best_distance <= max_distance_threshold:
        print(f"    ✓ Best match: '{directory_name}' -> '{best_match}' (distance: {best_distance})")
        return best_match, False
    else:
        print(f"    ✗ No good match found (best distance: {best_distance} > threshold: {max_distance_threshold})")
        
        if fallback_strategy == 'use_list':
            # Use the directory name for grouping but signal to use list variable for output naming
            print(f"    Using directory name for grouping, will use variable from list for output naming")
            return cleaned_dir_name, True
        else:
            print(f"    Using cleaned directory name: '{cleaned_dir_name}'")
            return cleaned_dir_name, False

def clean_directory_name(directory_name):
    """
    Clean directory name to extract a reasonable variable name
    """
    # Remove date patterns
    name = re.sub(r'\d{4}-\d{2}-\d{2}', '', directory_name)
    # Remove common prefixes
    name = re.sub(r'^nc4_', '', name)
    # Remove country names and common suffixes
    name = re.sub(r'-(Romania|Hungary|Bulgaria|Serbia)', '', name, flags=re.IGNORECASE)
    # Remove time patterns
    name = re.sub(r'_\d{4}_', '_', name)
    # Remove leading/trailing underscores and hyphens
    name = re.sub(r'^[-_]+|[-_]+$', '', name)
    # Replace multiple underscores/hyphens with single underscore
    name = re.sub(r'[-_]+', '_', name)
    
    return name.lower() if name else 'unknown'

def consolidate_weather_patches(weather_product_dir, variable_names_list=None, max_distance_threshold=3, fallback_strategy='use_list'):
    """
    Consolidate weather product patches using Levenshtein distance for variable matching
    
    Parameters:
    -----------
    weather_product_dir : str or Path
        Path to the directory containing weather product patch directories
    variable_names_list : list
        List of expected variable names for matching (e.g., ['solar', 'radar', 'satellite'])
        If None, uses directory names as-is
    max_distance_threshold : int
        Maximum Levenshtein distance allowed for variable matching (default: 3)
    fallback_strategy : str
        'use_list' - use variable from list when no match found
        'use_directory' - use cleaned directory name when no match found
    """
    weather_dir = Path(weather_product_dir)
    
    if not weather_dir.exists():
        raise ValueError(f"Directory does not exist: {weather_product_dir}")
    
    # Parse and group directories
    dir_info, unmatched_groups = parse_patch_directories(weather_dir, variable_names_list, max_distance_threshold, fallback_strategy)
    
    # Create all_patches output directory OUTSIDE the weather_product_dir
    all_patches_dir = weather_dir.parent / "all_patches"
    all_patches_dir.mkdir(exist_ok=True)
    
    print(f"Output directory: {all_patches_dir}")
    print(f"Processing patches from: {weather_dir}")
    
    # Process matched variables
    for variable, days_info in dir_info.items():
        print(f"Processing variable: {variable}")
        
        if len(days_info) > 1:
            print(f"  Multiple days found ({len(days_info)} days) - using xarray/dask")
            consolidate_multiple_days(days_info, variable, all_patches_dir)
        else:
            print(f"  Single day found - simple processing")
            consolidate_single_day(days_info[0], variable, all_patches_dir)
    
    # Process unmatched groups using variable names from list
    if unmatched_groups and variable_names_list:
        process_unmatched_groups(unmatched_groups, variable_names_list, all_patches_dir)

def process_unmatched_groups(unmatched_groups, variable_names_list, output_dir):
    """
    Process directories that couldn't be matched, using variable names from the list
    """
    print("\nProcessing unmatched directories using variable names from list:")
    print("=" * 60)
    
    # Sort unmatched groups by number of directories (largest first)
    sorted_groups = sorted(unmatched_groups.items(), key=lambda x: len(x[1]), reverse=True)
    
    # Assign variables from the list to unmatched groups
    used_variables = set()
    
    for i, (group_name, days_info) in enumerate(sorted_groups):
        # Try to find an unused variable from the list
        available_vars = [var for var in variable_names_list if var not in used_variables]
        
        if available_vars:
            # Use the first available variable
            assigned_variable = available_vars[0]
            used_variables.add(assigned_variable)
        else:
            # If all variables are used, append index to avoid conflicts
            assigned_variable = f"{variable_names_list[i % len(variable_names_list)]}_{i}"
        
        print(f"Assigning '{group_name}' -> '{assigned_variable}'")
        print(f"  Directories: {[info['original_name'] for info in days_info]}")
        
        # Process this group
        if len(days_info) > 1:
            print(f"  Multiple days found ({len(days_info)} days) - using xarray/dask")
            consolidate_multiple_days(days_info, assigned_variable, output_dir)
        else:
            print(f"  Single day found - simple processing")
            consolidate_single_day(days_info[0], assigned_variable, output_dir)

def parse_patch_directories(weather_dir, variable_names_list=None, max_distance_threshold=3, fallback_strategy='use_list'):
    """
    Parse patch directories and group by variable using Levenshtein distance matching
    
    Returns:
    --------
    tuple : (matched_dict, unmatched_dict)
        matched_dict: {variable: [directory_info]}
        unmatched_dict: {group_name: [directory_info]} for directories that need list variables
    """
    dir_pattern = re.compile(r'nc4_(\d{4}-\d{2}-\d{2})-[^_]+_(.+)')
    dir_info = defaultdict(list)
    unmatched_groups = defaultdict(list)
    
    print("Directory parsing and variable matching using Levenshtein distance:")
    print("=" * 70)
    if variable_names_list:
        print(f"Target variables: {variable_names_list}")
        print(f"Distance threshold: {max_distance_threshold}")
        print(f"Fallback strategy: {fallback_strategy}")
    print("-" * 70)
    
    for item in weather_dir.iterdir():
        if item.is_dir():
            match = dir_pattern.match(item.name)
            
            if match:
                date_str = match.group(1)  # 2024-06-12
                extracted_var = match.group(2)  # HRV, IR_108, etc.
                
                # Find best matching variable name using Levenshtein distance
                variable, use_list_for_output = find_best_variable_match(
                    extracted_var, variable_names_list, max_distance_threshold, fallback_strategy
                )
                
                try:
                    date_obj = datetime.strptime(date_str, '%Y-%m-%d')
                    year = date_obj.year
                    
                    dir_entry = {
                        'date': date_obj,
                        'date_str': date_str,
                        'path': item,
                        'year': year,
                        'original_name': item.name,
                        'extracted_var': extracted_var
                    }
                    
                    if use_list_for_output:
                        # Group unmatched directories separately
                        unmatched_groups[variable].append(dir_entry)
                    else:
                        # Group normally matched directories
                        dir_info[variable].append(dir_entry)
                        
                except ValueError:
                    print(f"Warning: Could not parse date from {item.name}")
            else:
                # Handle directories without clear variable/date structure
                print(f"  Non-standard directory: {item.name}")
                variable, use_list_for_output = find_best_variable_match(
                    item.name, variable_names_list, max_distance_threshold, fallback_strategy
                )
                
                dir_entry = {
                    'date': None,
                    'date_str': 'unknown',
                    'path': item,
                    'year': datetime.now().year,
                    'original_name': item.name,
                    'extracted_var': item.name
                }
                
                if use_list_for_output:
                    unmatched_groups[variable].append(dir_entry)
                else:
                    dir_info[variable].append(dir_entry)
    
    # Sort by date for each variable
    for variable in dir_info:
        dir_info[variable].sort(key=lambda x: x['date'] if x['date'] else datetime.min)
    
    for group in unmatched_groups:
        unmatched_groups[group].sort(key=lambda x: x['date'] if x['date'] else datetime.min)
    
    print("-" * 70)
    print("Matched variable groupings:")
    for variable, days_info in dir_info.items():
        original_names = [info['original_name'] for info in days_info]
        print(f"  {variable}: {len(days_info)} directories -> {original_names}")
    
    if unmatched_groups:
        print("\nUnmatched groups (will use variable names from list):")
        for group_name, days_info in unmatched_groups.items():
            original_names = [info['original_name'] for info in days_info]
            print(f"  {group_name}: {len(days_info)} directories -> {original_names}")
    
    print("=" * 70)
    
    return dict(dir_info), dict(unmatched_groups)

# [Include the same consolidate_multiple_days and consolidate_single_day functions from previous version]

def consolidate_multiple_days(days_info, variable, output_dir):
    """
    Consolidate multiple days using xarray and dask
    """
    output_filename = f"patches_{variable}.nc"
    output_path = output_dir / output_filename
    
    # Collect all .nc files from all days
    nc_files = []
    for day_info in days_info:
        day_dir = day_info['path']
        day_nc_files = list(day_dir.glob("*.nc"))
        nc_files.extend(day_nc_files)
        print(f"  Found {len(day_nc_files)} .nc files in {day_info['original_name']}")
    
    if not nc_files:
        print(f"  No .nc files found for variable {variable}")
        return
    
    print(f"  Total {len(nc_files)} .nc files to concatenate")
    print(f"  Output file: {output_path}")
    
    try:
        dask.config.set({'array.slicing.split_large_chunks': True})
        
        print("  Opening datasets with dask...")
        datasets = []
        
        for i, nc_file in enumerate(nc_files):
            try:
                ds = xr.open_dataset(nc_file, chunks={'dim_patch': 10})
                ds.attrs[f'source_file_{i}'] = nc_file.name
                day_info = next((d for d in days_info if str(nc_file).startswith(str(d['path']))), None)
                if day_info:
                    ds.attrs[f'source_day_{i}'] = day_info['date_str']
                    ds.attrs[f'source_dir_{i}'] = day_info['original_name']
                
                datasets.append(ds)
                print(f"    Opened {nc_file.name} - Shape: {dict(ds.dims)}")
                
            except Exception as e:
                print(f"    Warning: Could not open {nc_file}: {e}")
                continue
        
        if not datasets:
            print(f"  No valid datasets found for {variable}")
            return
        
        # Find patch dimension
        first_ds = datasets[0]
        available_dims = list(first_ds.dims.keys())
        patch_dim = None
        
        for dim_name in ['dim_patch', 'patch', 'sample', 'time', 'record']:
            if dim_name in available_dims:
                patch_dim = dim_name
                break
        
        if patch_dim is None:
            patch_dim = available_dims[0]
        
        print(f"  Using patch dimension: {patch_dim}")
        
        # Concatenate
        try:
            combined_ds = xr.concat(datasets, dim=patch_dim, combine_attrs='drop_conflicts')
            print(f"  Successfully concatenated - New shape: {dict(combined_ds.dims)}")
        except Exception as e:
            print(f"    Concatenation failed: {e}")
            combined_ds = xr.concat(datasets, dim='sample', combine_attrs='drop_conflicts')
        
        # Add metadata
        combined_ds.attrs.update({
            'consolidated_from_days': len(days_info),
            'consolidated_from_files': len(datasets),
            'consolidation_date': datetime.now().isoformat(),
            'variable': variable,
            'source_directories': [info['original_name'] for info in days_info],
            'total_patches': combined_ds.dims.get('dim_patch', combined_ds.dims.get('sample', 'unknown'))
        })
        
        # Save
        encoding = {var: {'zlib': True, 'complevel': 4, 'shuffle': True} for var in combined_ds.data_vars}
        combined_ds.to_netcdf(output_path, encoding=encoding, compute=True)
        
        # Clean up
        for ds in datasets:
            ds.close()
        combined_ds.close()
        
        print(f"  Successfully consolidated {variable} data to {output_path}")
        
    except Exception as e:
        print(f"  Error consolidating {variable}: {e}")

def consolidate_single_day(day_info, variable, output_dir):
    """
    Handle single day case
    """
    day_dir = day_info['path']
    nc_files = list(day_dir.glob("*.nc"))
    
    if not nc_files:
        print(f"  No .nc files found in {day_dir}")
        return
    
    output_filename = f"patches_{variable}.nc"
    output_path = output_dir / output_filename
    
    print(f"  Processing {len(nc_files)} files from {day_info['original_name']}")
    
    if len(nc_files) == 1:
        source_file = nc_files[0]
        print(f"  Copying single file: {source_file.name} -> {output_filename}")
        shutil.copy2(source_file, output_path)
    else:
        print(f"  Concatenating {len(nc_files)} files from single day")
        
        try:
            datasets = [xr.open_dataset(nc_file) for nc_file in nc_files]
            
            # Find patch dimension
            first_ds = datasets[0]
            available_dims = list(first_ds.dims.keys())
            patch_dim = available_dims[0]
            
            for dim_name in ['dim_patch', 'patch', 'sample']:
                if dim_name in available_dims:
                    patch_dim = dim_name
                    break
            
            combined_ds = xr.concat(datasets, dim=patch_dim, combine_attrs='drop_conflicts')
            combined_ds.attrs.update({
                'variable': variable,
                'source_directory': day_info['original_name']
            })
            
            combined_ds.to_netcdf(output_path)
            
            for ds in datasets:
                ds.close()
            combined_ds.close()
            
            print(f"  Successfully consolidated single day {variable} data")
            
        except Exception as e:
            print(f"  Error processing single day {variable}: {e}")