# batch.py - Fixed for mixed patch sizes with coordinate-based upscaling
# Strategy: Upscale 8×8 patches to 32×32 to match coordinate system,
#          then downscale final output to correct size

from datetime import timedelta
from itertools import chain
from numba import njit, prange
import numpy as np
from tensorflow.keras.utils import Sequence


def upscale_patch_4x(patch):
    """Upscale 8×8 patch to 32×32 using bilinear interpolation"""
    # Simple bilinear upscaling without scipy
    h, w = patch.shape
    new_h, new_w = h * 4, w * 4
    upscaled = np.zeros((new_h, new_w), dtype=patch.dtype)
    
    for i in range(new_h):
        for j in range(new_w):
            # Map to original coordinates
            src_i = i / 4.0
            src_j = j / 4.0
            
            # Get integer parts
            i0, j0 = int(src_i), int(src_j)
            i1, j1 = min(i0 + 1, h - 1), min(j0 + 1, w - 1)
            
            # Get fractional parts
            fi, fj = src_i - i0, src_j - j0
            
            # Bilinear interpolation
            val = (1 - fi) * (1 - fj) * patch[i0, j0] + \
                  (1 - fi) * fj * patch[i0, j1] + \
                  fi * (1 - fj) * patch[i1, j0] + \
                  fi * fj * patch[i1, j1]
            
            upscaled[i, j] = val
    
    return upscaled


def downscale_image_4x(img):
    """Downscale 256×256 to 64×64 using averaging"""
    # Average over 4×4 blocks
    h, w = img.shape
    new_h, new_w = h // 4, w // 4
    downscaled = np.zeros((new_h, new_w), dtype=img.dtype)
    
    for i in range(new_h):
        for j in range(new_w):
            # Average 4×4 block
            block = img[i*4:(i+1)*4, j*4:(j+1)*4]
            downscaled[i, j] = block.mean()
    
    return downscaled


class BatchGenerator:
    def __init__(self, 
        predictors,
        targets,
        raw,
        coords_by_time,
        primary_raw_var,
        box_size=(8,8),
        timesteps=(12,12),
        batch_size=8,
        interval=timedelta(minutes=5),
        random_seed=None,
        valid_frac=None,
        test_frac=None,
        dataset_block_size=12*24,
        upscale_mode=None  # "save", "load", or None
    ):
        self.batch_size = batch_size
        self.interval = interval
        self.timesteps = timesteps
        self.primary_raw_var = primary_raw_var
        self.rng = np.random.RandomState(seed=random_seed)
        self.upscale_mode = upscale_mode
        
        # mappings from source variables to predictors and targets
        self.pred_sources = {}
        self.pred_transform = {}
        pred_timeframe = {}
        for (pred_name, pred_data) in predictors.items():
            self.pred_sources[pred_name] = pred_data["source_vars"]
            self.pred_transform[pred_name] = pred_data["transform"]
            pred_timeframe[pred_name] = pred_data.get("timeframe", "past")
        self.pred_names_past = [v for v in predictors.keys() if pred_timeframe[v] == "past"]
        self.pred_names_future = [v for v in predictors.keys() if pred_timeframe[v] == "future"]
        self.pred_names_static = [v for v in predictors.keys() if pred_timeframe[v] == "static"]
        
        self.target_sources = {}
        self.target_transform = {}
        for (target_name, target_data) in targets.items():
            self.target_sources[target_name] = target_data["source_vars"]
            self.target_transform[target_name] = target_data["transform"]
        self.target_names = list(targets.keys())

        # indices for retrieving source data
        self.raw_batch_index = {}
        self.setup_batch_index(primary_raw_var, raw[primary_raw_var], box_size, 
                               upscale_mode=self.upscale_mode)
        index_limits = self.raw_batch_index[primary_raw_var].index_limits

        # indices for retrieving source data for the rest of the variables
        for (raw_name, raw_data) in raw.items():
            if raw_name == primary_raw_var:
                continue
            self.setup_batch_index(raw_name, raw_data, box_size,
                index_limits=index_limits, upscale_mode=self.upscale_mode)

        # Convert coords_by_time to numpy arrays
        self.coords_by_time = {}
        for t in coords_by_time:
            coords_list = coords_by_time[t]
            if isinstance(coords_list, (list, tuple)) and len(coords_list) > 0:
                self.coords_by_time[t] = np.array(coords_list)
            else:
                self.coords_by_time[t] = np.array([]).reshape(0, 2)

        self.time_coords = self.train_valid_test_split(
            valid_frac=valid_frac, test_frac=test_frac,
            block_size=dataset_block_size
        )

        self.fixed_coord = {}
        fixed_coord_times = chain(
            self.time_coords["valid"],
            self.time_coords["test"]
        )

        for t in fixed_coord_times:
            coords = self.coords_by_time.get(t, np.array([]).reshape(0, 2))
            if len(coords) > 0:
                self.fixed_coord[t] = self.rng.randint(len(coords))
            else:
                self.fixed_coord[t] = 0

    def setup_batch_index(self, raw_name, raw_data, box_size, index_limits=None, upscale_mode=None):
        time_dim = 0

        sources_past = set(chain(*[self.pred_sources[v] for v in self.pred_names_past]))
        sources_future = set(chain(*[self.pred_sources[v] for v in self.pred_names_future])) | \
            set(chain(*[self.target_sources[v] for v in self.target_names]))
        sources_static = set(chain(*[self.pred_sources[v] for v in self.pred_names_static]))
        
        if (raw_name in sources_past) or (raw_name == self.primary_raw_var):
            time_dim = self.timesteps[0]
        if raw_name in sources_future:
            time_dim = max(time_dim, self.timesteps[1])
        if raw_name in sources_static:
            time_dim = max(time_dim, 1)
        if time_dim == 0:
            return

        interp = raw_data.get("interpolation", None)
        zero_value = raw_data.get("zero_value", 0)
        missing_value = raw_data.get("missing_value", zero_value)
        
        if interp is None:
            self.raw_batch_index[raw_name] = PatchIndex(
                raw_data["patches"],
                raw_data["patch_coords"],
                raw_data["patch_times"],
                raw_data["zero_patch_coords"],
                raw_data["zero_patch_times"],
                zero_value=zero_value,
                missing_value=missing_value,
                interval=self.interval,
                box_size=(time_dim,)+box_size,
                index_limits=index_limits,
                static=raw_data.get("static", False),
                var_name=raw_name,
                upscale_mode=upscale_mode
            )
            print(f"Variable '{raw_name}': "
                  f"original_patch_size={self.raw_batch_index[raw_name].original_patch_size}, "
                  f"working_patch_size={self.raw_batch_index[raw_name].patch_size}, "
                  f"output_size={self.raw_batch_index[raw_name].final_output_size}, "
                  f"sample_shape={self.raw_batch_index[raw_name].sample_shape}")
        else:
            self.raw_batch_index[raw_name] = InterpolatingPatchIndex(
                raw_data["patches"],
                raw_data["patch_coords"],
                raw_data["patch_times"],                
                raw_data["zero_patch_coords"],
                raw_data["zero_patch_times"],
                zero_value=zero_value,
                missing_value=missing_value,
                interval=self.interval,
                box_size=(time_dim,)+box_size,
                index_limits=index_limits,
                method=interp,
                stride=raw_data["stride"],
                static=raw_data.get("static", False),
                var_name=raw_name,
                upscale_mode=upscale_mode
            )

    def train_valid_test_split(self, valid_frac=None, test_frac=None, block_size=None):
        times = np.array(sorted(self.coords_by_time))
        n = len(times)
        
        times_valid = []
        if valid_frac is not None:
            n_valid = valid_frac * n
            while len(times_valid) < n_valid:
                if len(times) == 0:
                    break
                t0 = times[self.rng.randint(len(times))]
                t1 = t0 + block_size
                selection = (t0 <= times) & (times < t1)
                times_valid.extend(list(times[selection]))
                times = times[~selection]
        times_valid = np.array(times_valid)
        
        times_test = []
        if test_frac is not None:
            n_test = test_frac * n
            while len(times_test) < n_test:
                if len(times) == 0:
                    break
                t0 = times[self.rng.randint(len(times))]
                t1 = t0 + block_size
                selection = (t0 <= times) & (times < t1)
                times_test.extend(list(times[selection]))
                times = times[~selection]
        times_test = np.array(times_test)

        self.rng.shuffle(times)
        self.rng.shuffle(times_valid)
        self.rng.shuffle(times_test)

        return {"train": times, "valid": times_valid, "test": times_test}

    def frame_spatial_coordinates(self, t_batch, dataset="train"):
        """Select pixel coordinates using 32px grid spacing"""
        i_batch = []
        j_batch = []
        box_size = self.raw_batch_index[self.primary_raw_var].box_size
        # Always use 32px for coordinate calculation (all coords are 32px spaced)
        grid_spacing = 32
        
        for t in t_batch:
            coords = self.coords_by_time.get(t, np.array([]).reshape(0, 2))
            if len(coords) == 0:
                i_batch.append(0)
                j_batch.append(0)
                continue
            
            i_min, j_min = coords.min(axis=0)
            i_max, j_max = coords.max(axis=0)
            
            storm_patches_i = (i_max - i_min) // grid_spacing + 1
            storm_patches_j = (j_max - j_min) // grid_spacing + 1
            
            if storm_patches_i >= box_size[1] and storm_patches_j >= box_size[2]:
                max_offset_i = i_max - (box_size[1] - 1) * grid_spacing
                max_offset_j = j_max - (box_size[2] - 1) * grid_spacing
                
                if dataset == "train":
                    pixel_i0 = self.rng.randint(i_min, max_offset_i + grid_spacing, dtype=np.int32)
                    pixel_j0 = self.rng.randint(j_min, max_offset_j + grid_spacing, dtype=np.int32)
                else:
                    pixel_i0 = i_min
                    pixel_j0 = j_min
                
                pixel_i0 = (pixel_i0 // grid_spacing) * grid_spacing
                pixel_j0 = (pixel_j0 // grid_spacing) * grid_spacing
            else:
                pixel_i0 = i_min
                pixel_j0 = j_min
            
            i_batch.append(int(pixel_i0))
            j_batch.append(int(pixel_j0))

        return (np.array(i_batch), np.array(j_batch))

    def random_augments(self):
        transpose = bool(self.rng.randint(2))
        flipud = bool(self.rng.randint(2))
        fliplr = bool(self.rng.randint(2))
        return (transpose, flipud, fliplr)

    def augment(self, batch, augments):
        (transpose, flipud, fliplr) = augments
        if transpose:
            batch = batch.transpose((0,1,3,2,4))
        if flipud:
            batch = batch[:,:,::-1,:,:]
        if fliplr:
            batch = batch[:,:,:,::-1,:]
        return batch

    def get_batch(self, t, i, j, var_names, var_sources, transform, num_timesteps, save_diagnostics=False):
        raw_data = {}
        for var_name in var_names:
            sources = var_sources[var_name]
            for raw_var in sources:
                if raw_var not in raw_data:
                    raw_data[raw_var] = self.raw_batch_index[raw_var](
                        t, i, j, num_timesteps=num_timesteps, 
                        save_diagnostics=save_diagnostics)

        batch_data = []
        for var_name in var_names:
            raw_vars = (raw_data[raw_var][:,:num_timesteps,...]
                for raw_var in var_sources[var_name])
            transformed_vars = transform[var_name](*raw_vars)
            has_channels = (len(transformed_vars.shape) == 5)
            if not has_channels:
                transformed_vars = transformed_vars.reshape(transformed_vars.shape + (1,))
            batch_data.append(transformed_vars)

        return batch_data

    def batch(self, idx, dataset="train", save_diagnostics=False):
        t_pred = self.time_coords[dataset][
            idx*self.batch_size:(idx+1)*self.batch_size
        ]
        t_target = t_pred + self.timesteps[0]
        (i,j) = self.frame_spatial_coordinates(t_pred, dataset=dataset)

        pred_batch_past = self.get_batch(t_pred, i, j,
            self.pred_names_past, self.pred_sources, self.pred_transform,
            self.timesteps[0], save_diagnostics=save_diagnostics
        )
        pred_batch_future = self.get_batch(t_target, i, j,
            self.pred_names_future, self.pred_sources, self.pred_transform,
            self.timesteps[1], save_diagnostics=save_diagnostics
        )
        pred_batch_static = self.get_batch(t_pred, i, j,
            self.pred_names_static, self.pred_sources, self.pred_transform, 1,
            save_diagnostics=save_diagnostics
        )
        pred_batch = pred_batch_past + pred_batch_future + pred_batch_static
        target_batch = self.get_batch(t_target, i, j,
            self.target_names, self.target_sources, self.target_transform,
            self.timesteps[1], save_diagnostics=save_diagnostics
        )    

        if dataset == "train":
            augments = self.random_augments()
            pred_batch = [self.augment(b, augments) for b in pred_batch]
            target_batch = [self.augment(b, augments) for b in target_batch]

        return (pred_batch, target_batch)


class BatchSequence(Sequence):
    def __init__(self, batch_gen, dataset="train"):
        super().__init__()
        self.batch_gen = batch_gen
        self.dataset = dataset

    def __len__(self):
        return len(self.batch_gen.time_coords[self.dataset]) // \
            self.batch_gen.batch_size

    def __getitem__(self, idx):
        return self.batch_gen.batch(idx, dataset=self.dataset)

    def on_epoch_end(self):
        self.batch_gen.rng.shuffle(self.batch_gen.time_coords["train"])


class PatchIndex:
    IDX_ZERO = -1
    IDX_MISSING = -2
    GRID_SPACING = 32  # All coordinates are 32px spaced

    def __init__(
        self, patch_data, patch_coords, patch_times,
        zero_patch_coords, zero_patch_times,
        interval=timedelta(minutes=5),
        box_size=(12,8,8), zero_value=0,
        missing_value=0,
        index_limits=None, static=False,
        var_name="unknown", upscale_mode=None
    ):
        if (index_limits is None) or static:
            if len(patch_times) > 0:
                t0 = patch_times.min()
                t1 = patch_times.max()
            else:
                t0 = t1 = 0
        else:
            (t0, t1, _, _) = index_limits
        
        self.t0 = t0
        self.t1 = t1
        self.dt = int(round(interval.total_seconds()))
        self.box_size = box_size
        self.zero_value = zero_value
        self.missing_value = missing_value
        self.static = static
        self.index_limits = (t0, t1, 0, 0)
        self.var_name = var_name

        # Detect original patch size
        if len(patch_data) > 0:
            first_patch = patch_data[0]
            self.original_patch_size = first_patch.shape[0]
        else:
            self.original_patch_size = 32
        
        # CRITICAL: Upscale 8×8 patches to 32×32 to match coordinate system
        if self.original_patch_size == 8:
            upscaled_file = f"upscaled_patches/{var_name}_upscaled_8x8_to_32x32.npy"
            
            if upscale_mode == "load":
                # Load pre-upscaled patches
                print(f"  Loading upscaled patches from {upscaled_file}...")
                import os
                if not os.path.exists(upscaled_file):
                    raise FileNotFoundError(
                        f"Upscaled patches file not found: {upscaled_file}\n"
                        f"Run with upscale_mode='save' first to generate the file."
                    )
                self.patch_data = np.load(upscaled_file)
                print(f"  Loaded {len(self.patch_data)} upscaled patches!")
                
            elif upscale_mode == "save":
                # Upscale and save
                print(f"  Upscaling 8×8 patches to 32×32 to match coordinate system...")
                upscaled_patches = []
                for idx, patch in enumerate(patch_data):
                    if idx % 1000 == 0:
                        print(f"    Upscaled {idx}/{len(patch_data)} patches...")
                    upscaled = upscale_patch_4x(patch)
                    upscaled_patches.append(upscaled.astype(patch.dtype))
                self.patch_data = np.array(upscaled_patches)
                
                # Save to disk
                import os
                os.makedirs("upscaled_patches", exist_ok=True)
                print(f"  Saving upscaled patches to {upscaled_file}...")
                np.save(upscaled_file, self.patch_data)
                print(f"  Upscaling complete and saved!")
                
            else:
                # No save/load - upscale in memory (default behavior)
                print(f"  Upscaling 8×8 patches to 32×32 to match coordinate system...")
                upscaled_patches = []
                for idx, patch in enumerate(patch_data):
                    if idx % 1000 == 0:
                        print(f"    Upscaled {idx}/{len(patch_data)} patches...")
                    upscaled = upscale_patch_4x(patch)
                    upscaled_patches.append(upscaled.astype(patch.dtype))
                self.patch_data = np.array(upscaled_patches)
                print(f"  Upscaling complete!")
            
            self.patch_size = 32  # Working size after upscaling
            self.final_output_size = 64  # Will downscale from 256 to 64
        else:
            self.patch_data = patch_data
            self.patch_size = 32  # Already 32×32
            self.final_output_size = 256  # Keep at 256×256
        
        # Sample shape is always 256×256 during stitching
        self.sample_shape = (box_size[0], 256, 256)

        # Build lookup
        num_times = (t1-t0)//self.dt + 1
        patch_locs = []
        
        # Regular patches - use 32px grid
        for k in range(len(patch_coords)):
            t = (patch_times[k]-t0) // self.dt
            if (t < 0) or (t >= num_times):
                continue
            pi, pj = patch_coords[k]
            pi = (pi // self.GRID_SPACING) * self.GRID_SPACING
            pj = (pj // self.GRID_SPACING) * self.GRID_SPACING
            patch_locs.append((t, pi, pj, k))
        
        # Zero patches
        existing = {(t, pi, pj) for t, pi, pj, _ in patch_locs}
        for k in range(len(zero_patch_coords)):
            t = (zero_patch_times[k]-t0) // self.dt
            if (t < 0) or (t >= num_times):
                continue
            pi, pj = zero_patch_coords[k]
            pi = (pi // self.GRID_SPACING) * self.GRID_SPACING
            pj = (pj // self.GRID_SPACING) * self.GRID_SPACING
            if (t, pi, pj) not in existing:
                patch_locs.append((t, pi, pj, self.IDX_ZERO))
        
        # Convert to arrays
        if len(patch_locs) > 0:
            self.patch_times_idx = np.array([x[0] for x in patch_locs], dtype=np.int32)
            self.patch_pixels_i = np.array([x[1] for x in patch_locs], dtype=np.int32)
            self.patch_pixels_j = np.array([x[2] for x in patch_locs], dtype=np.int32)
            self.patch_indices = np.array([x[3] for x in patch_locs], dtype=np.int32)
        else:
            self.patch_times_idx = np.array([], dtype=np.int32)
            self.patch_pixels_i = np.array([], dtype=np.int32)
            self.patch_pixels_j = np.array([], dtype=np.int32)
            self.patch_indices = np.array([], dtype=np.int32)

        self._batch = None

    def _alloc_batch(self, n):
        if (self._batch is None) or (self._batch.shape[0] < n):
            del self._batch
            self._batch = np.zeros((n,)+self.sample_shape, self.patch_data.dtype)
        return self._batch

    def __call__(self, t0_all, i0_all, j0_all, num_timesteps=None):
        n = len(t0_all)
        batch = self._alloc_batch(n)
        if num_timesteps is None:
            num_timesteps = self.box_size[0]

        if self.static:
            t0_all = np.zeros_like(t0_all)

        # Build at 256×256 resolution
        build_batch(
            batch, self.patch_data,
            self.patch_times_idx, self.patch_pixels_i,
            self.patch_pixels_j, self.patch_indices,
            t0_all, num_timesteps, i0_all, j0_all,
            self.box_size[1], self.box_size[2],
            self.zero_value, self.missing_value, self.static
        )

        # Downscale if this was originally 8×8 data
        if self.final_output_size == 64:
            # Downscale from 256×256 to 64×64
            downscaled = np.zeros((n, num_timesteps, 64, 64), dtype=batch.dtype)
            for i in range(n):
                for t in range(num_timesteps):
                    downscaled[i, t] = downscale_image_4x(batch[i, t])
            return downscaled
        else:
            return batch


IDX_ZERO = PatchIndex.IDX_ZERO
IDX_MISSING = PatchIndex.IDX_MISSING
GRID_SPACING = PatchIndex.GRID_SPACING

@njit(parallel=True)
def build_batch(
    batch, patch_data,
    patch_times_idx, patch_pixels_i, patch_pixels_j, patch_indices,
    t0_all, num_timesteps, pixel_i0_all, pixel_j0_all,
    box_size_i, box_size_j, zero_value, missing_value, static=False
):
    """Build batch at 32×32 resolution (all patches are now 32×32)"""
    for k in prange(len(t0_all)):
        t0 = t0_all[k]
        t1 = t0 + num_timesteps
        pixel_i0 = pixel_i0_all[k]
        pixel_j0 = pixel_j0_all[k]
        
        for t in range(t0, t1):
            bt = int(t-t0)
            tt = int(t0 if static else t)
            
            for patch_i_idx in range(box_size_i):
                for patch_j_idx in range(box_size_j):
                    # All coordinates and patches are 32px
                    target_pi = pixel_i0 + patch_i_idx * GRID_SPACING
                    target_pj = pixel_j0 + patch_j_idx * GRID_SPACING
                    target_pi = (target_pi // GRID_SPACING) * GRID_SPACING
                    target_pj = (target_pj // GRID_SPACING) * GRID_SPACING
                    
                    bi0 = patch_i_idx * 32
                    bi1 = bi0 + 32
                    bj0 = patch_j_idx * 32
                    bj1 = bj0 + 32
                    
                    # Find patch
                    ind = IDX_MISSING
                    for p in range(len(patch_times_idx)):
                        if (patch_times_idx[p] == tt and 
                            patch_pixels_i[p] == target_pi and 
                            patch_pixels_j[p] == target_pj):
                            ind = patch_indices[p]
                            break
                    
                    # Place 32×32 patch
                    if ind >= 0:
                        batch[k,bt,bi0:bi1,bj0:bj1] = patch_data[ind]
                    elif ind == IDX_ZERO:
                        batch[k,bt,bi0:bi1,bj0:bj1] = zero_value
                    else:
                        batch[k,bt,bi0:bi1,bj0:bj1] = missing_value


class InterpolatingPatchIndex(PatchIndex):
    def __init__(self, *args, stride=12, method='linear', **kwargs):
        super().__init__(*args, **kwargs)
        self.stride = stride
        self.method = method
        times_with_data = np.any(self.patch_times_idx >= 0) if len(self.patch_times_idx) > 0 else False
        self.first_valid_step = 0 if not times_with_data else int(self.patch_times_idx.min())
        self._batches = None

    def _alloc_batches(self, n_batches, n_samples):
        shape = (n_samples, n_batches)
        if (self._batches is None) or (self._batches.shape[:2] != shape):
            del self._batches
            # Use final output size for batches
            final_size = self.final_output_size
            self._batches = np.zeros((n_samples, n_batches, final_size, final_size), 
                                    self.patch_data.dtype)
        return self._batches

    def __call__(self, t0_all, i0_all, j0_all, num_timesteps=None):
        n_samples = len(t0_all)
        if num_timesteps is None:
            num_timesteps = self.box_size[0]
        
        dt0 = (t0_all - self.first_valid_step) % self.stride
        t0_mod = t0_all - dt0
        max_t = (self.t1-self.t0)//self.dt
        t0_mod = np.clip(t0_mod, 0, max_t)

        t_steps = range(0, num_timesteps+self.stride+1, self.stride)
        num_steps = len(t_steps)
        batches = self._alloc_batches(num_steps, n_samples)
        t_step = t0_mod.copy()
        for i in range(num_steps):
            b = super().__call__(t_step, i0_all, j0_all, num_timesteps=1)
            batches[:,i,...] = b[:,0,...]
            valid_ind = (t_step + self.stride) < max_t
            t_step[valid_ind] += self.stride

        # Allocate interpolated batch at correct size
        final_size = self.final_output_size
        batch_ip = np.zeros((n_samples, num_timesteps, final_size, final_size), 
                           self.patch_data.dtype)
        interp_batch(batch_ip, batches, t0_all, dt0, self.stride, num_timesteps,
            ip_linear=(self.method=='linear'))

        return batch_ip


@njit(parallel=True)
def interp_batch(batch_ip, batches, t0_all, dt0, stride, num_timesteps, ip_linear=True):
    n = len(t0_all)
    for k in prange(n):
        t0 = t0_all[k]
        t1 = t0 + num_timesteps
        dt = dt0[k]
        prev_batch_index = 0
        for t in range(t0,t1):
            bt = int(t-t0)
            prev_batch = batches[:,prev_batch_index,...]
            next_batch = batches[:,prev_batch_index+1,...]
            
            if ip_linear:
                w_next = dt/stride
                w_prev = 1-w_next
                batch_ip[k,bt,:,:] = w_prev*prev_batch[k,:,:] + w_next*next_batch[k,:,:]
            else:
                batch_ip[k,bt,:,:] = prev_batch[k,:,:] if dt < stride/2 else next_batch[k,:,:]

            dt += 1
            if dt >= stride:
                dt = 0
                prev_batch_index += 1
