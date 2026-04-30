import datetime
import os
import warnings
from glob import glob

import dask.array as da
import h5py
import matplotlib

# Suppress specific runtime warnings
warnings.filterwarnings('ignore', message='invalid value encountered in log')
warnings.filterwarnings('ignore', message='invalid value encountered in cast')
warnings.filterwarnings('ignore', message='Coordinate "longitude" referenced by dataarray accumulated_flash_area')
warnings.filterwarnings('ignore', message='Coordinate "latitude" referenced by dataarray accumulated_flash_area')
warnings.filterwarnings('ignore', message='Overlap checking not implemented')
warnings.filterwarnings('ignore', message='Consolidated metadata is currently not part in the Zarr format 3 specification')
warnings.filterwarnings('ignore', message='Import from the new location instead')
warnings.filterwarnings('ignore', message='has _Unsigned attribute but is not of integer type')

from satpy.modifiers.angles import _get_sensor_angles_ndarray  #, get_angles

matplotlib.use("agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.patches as patches
import pandas as pd
import s3fs
import xarray as xr

import cartopy.feature as cfeature
import numpy as np
from pyresample.area_config import create_area_def
from satpy import Scene
from satpy.modifiers.parallax import ParallaxCorrection

import lampo_radar.settings as settings
from lampo_radar.logger import setup_logger

logger = setup_logger(__name__)

from zarr.codecs import BloscCodec

compressor = BloscCodec(cname='zstd', clevel=5, shuffle='bitshuffle')

REFL_MAX_PLOT = 60

# TODO add support for config file

# TODO handle elevation file without local file?
# TODO handle resampling cache directory

# TODO implement FCILI quality control

def get_opera_adef():
    if settings.DATAGEN["area_name"] == "opera":
        return create_area_def(
            area_id="opera",
            proj_id="opera",
            projection="+proj=laea +lat_0=55.0 +lon_0=10.0 +x_0=1950000.0 +y_0=-2100000.0 +units=m +ellps=WGS84",
            shape=(4400, 3800),
            area_extent=[-10.4345768386404, 31.7462153182675, 57.8119647501499, 67.6210371071631],
            units='deg'
        )
    elif settings.DATAGEN["area_name"] == "ukraine":
        return create_area_def(
            area_id="ukraine",
            proj_id="ukraine",
            projection="+proj=laea +lat_0=49.0 +lon_0=32.0 +units=m +ellps=WGS84",
            shape=(1080, 2320),
            area_extent=[20, 42, 45, 54],
            units='deg'
        )
    else:
        raise ValueError(f"Unsupported area name: {settings.DATAGEN['area_name']}")


def get_modified_cmap_for_radar(first_color_alpha=1):
    """Create a custom colormap with gray for the first color in jet."""
    jet_cmap = cm.get_cmap('jet')
    jet_colors = jet_cmap(np.linspace(0, 1, 256))
    jet_colors[0] = mcolors.to_rgba('lightgray', first_color_alpha)
    modified_jet = mcolors.ListedColormap(jet_colors)
    return modified_jet


def get_modified_cmap_for_li():
    viridis_cmap = cm.get_cmap('viridis')
    viridis_colors = viridis_cmap(np.linspace(0, 1, 256))
    viridis_colors[0] = mcolors.to_rgba('white', alpha=0)  # Make first color transparent
    modified_viridis = mcolors.ListedColormap(viridis_colors)
    return modified_viridis


def get_cmap_for_opera_coverage():
    colors = ["white", "lightgrey"]
    cmap = mcolors.ListedColormap(colors)
    bounds = [-0.5, 0.5, 1.5]
    norm = mcolors.BoundaryNorm(bounds, cmap.N)
    return cmap, norm


def add_coastlines_and_features(ax):
    ax.coastlines(resolution='50m', edgecolor='black', linewidth=0.5)
    ax.add_feature(cfeature.BORDERS, linestyle=':', edgecolor='black', linewidth=0.5)


def find_valid_patches(radar_image, patch_size, n_patches, max_tries=1e5):
    min_distance = 1.2 * np.sqrt((0.5 * patch_size) ** 2 + (0.5 * patch_size) ** 2)  # about 1/8 of overlap

    h, w = radar_image.shape
    ph, pw = patch_size, patch_size
    valid_coords = []

    tries = 0
    while len(valid_coords) < n_patches and tries < max_tries:
        i = np.random.randint(0, h - ph + 1)
        j = np.random.randint(0, w - pw + 1)

        patch = radar_image[i:i + ph, j:j + pw]
        if np.all(~np.isnan(patch)):
            # Check overlap: compute distances to existing patches
            too_close = False
            for (ii, jj) in valid_coords:
                # Euclidean distance between patch origins (top-left corners)
                dist = np.sqrt((i - ii) ** 2 + (j - jj) ** 2)
                if dist < min_distance:
                    too_close = True
                    break

            if not too_close:
                valid_coords.append((i, j))

        tries += 1

    logger.info(f"Found {len(valid_coords)} valid patches after {tries} tries.")
    return np.array(valid_coords)

def get_patch_stats(patch_data):
    stats = {
        'min': float(np.nanmin(patch_data)),
        'max': float(np.nanmax(patch_data)),
        'mean': float(np.nanmean(patch_data)),
        'std': float(np.nanstd(patch_data)),
        'npix_above_zero': int(np.sum(patch_data > 0)),
    }

    return stats
class S3BucketAccessor:

    def __init__(self, s3_config):
        self.fs = s3fs.S3FileSystem(**s3_config)
        self.s3_config = s3_config
        return

    def get_h5_dataset(self, filename, dataset_names):
        path_without_prefix = filename.replace("s3://", "")

        datasets = {}
        with self.fs.open(path_without_prefix, mode='rb') as f:
            with h5py.File(f, 'r') as h5f:
                for dataset_name in dataset_names:
                    dataset = h5f[dataset_name][:]
                    datasets.update({dataset_name: dataset})
                logger.info(f"Successfully read {dataset_names} from {filename}")
                return datasets


class DataPreprocessor:
    def __init__(self, start_time, eumdac_data_folder=None, output_folder=None, save_to_s3=False):
        self.append_mode = False
        self.start_time = start_time
        self.eumdac_data_folder = eumdac_data_folder
        self.output_folder = os.path.join(output_folder, start_time.replace(':', '')) if output_folder else None
        self.save_to_s3 = save_to_s3

        self.fci_datasets = settings.DATAGEN["fci_channels_for_ds"]
        self.li_dataset = "accumulated_flash_area"
        self.oca_dataset = "retrieved_cloud_top_height"
        self.opera_dataset = "opera_maxrefl"

        self.patch_size = 256

        self.target_adef = get_opera_adef()

        self._ewc_lampo_s3_accessor = None
        self._destine_lampo_radar_accessor = None

        self.data_initialised = None
        self.data_processed = None
        self.s3_zarr_exists = None
        self.local_zarr_exists = None

        self.opera_maxrefl = None
        self.oca_filenames = None
        self.li_filenames = None
        self.fci_filenames = None
        self.li_scn = None
        self.oca_scn = None
        self.fci_scn = None
        self.li_scn_res = None
        self.oca_scn_res = None
        self.fci_scn_res = None
        self.fci_scn_corr = None
        self.li_scn_corr = None

        self.combined_proc_ds = None
        self.patch_statistics_df = None

        self.patches_coords = None
        self.patch_statistics = {}
        return

    @property
    def ewc_lampo_s3_accessor(self):
        if self._ewc_lampo_s3_accessor is None:
            s3_config = {
                'client_kwargs': {
                    'endpoint_url': 'https://s3.waw3-1.cloudferro.com',
                    'verify': False
                },
                'key': settings.get_secret("EWC_S3_ACCESS_KEY"),
                'secret': settings.get_secret("EWC_S3_SECRET_KEY"),
            }
            self._ewc_lampo_s3_accessor = S3BucketAccessor(s3_config=s3_config)
        return self._ewc_lampo_s3_accessor

    @property
    def destine_lampo_s3_accessor(self):
        if self._destine_lampo_radar_accessor is None:
            s3_config = {
                'client_kwargs': {
                    'endpoint_url': f'https://s3.eumetsat.data.destination-earth.eu',
                    # 'verify': False,
                    'region_name': 'DEDL-EUMETSAT',
                },
                'key': settings.get_secret("AWS_ACCESS_KEY_ID"),
                'secret': settings.get_secret("AWS_SECRET_ACCESS_KEY"),
            }
            self._destine_lampo_radar_accessor = S3BucketAccessor(s3_config=s3_config)
        return self._destine_lampo_radar_accessor

    def initialise_data(self):
        logger.info("Initialising data")
        self.fci_filenames = sorted(glob(os.path.join(self.eumdac_data_folder, "*FCI-1C*.nc")))
        self.li_filenames = sorted(glob(os.path.join(self.eumdac_data_folder, "*LI-2*BODY*.nc")))
        self.oca_filenames = sorted(glob(os.path.join(self.eumdac_data_folder, "*FCI-2-OCA*.nc")))

        if not (self.fci_filenames and self.li_filenames and self.oca_filenames and len(self.fci_filenames) == 20):
            logger.warning("EUMDAC Data missing for current time step.")
            self.data_initialised = False
            return

        if not self.check_s3_file_presence(self.opera_s3_filename):
            logger.warning("OPERA Data missing for current time step.")
            self.data_initialised = False
            return

        self.data_initialised = True
        return

    def get_scenes(self, scene_types=None, fci_variables=None):
        """
        Load scenes from FCI, OCA, and LI files.

        Args:
            fci_variables: List of FCI variables to load. If None, loads all self.fci_datasets
        """
        if scene_types is None:
            scene_types = ['fci', 'oca', 'li']
        logger.info("Getting scenes")

        if 'fci' in scene_types:
            if fci_variables is None:
                fci_variables = self.fci_datasets

            self.fci_scn = Scene(filenames=self.fci_filenames, reader='fci_l1c_nc')
            for fci_variable in fci_variables:
                if "_epl_corr" in fci_variable:
                    fci_variable_base = fci_variable.replace("_epl_corr", "")
                    self.fci_scn.load([fci_variable_base], upper_right_corner='NE', pad_data=False,
                                      modifiers=('effective_solar_pathlength_corrected',))
                    self.fci_scn[fci_variable] = self.fci_scn[fci_variable_base]
                    del self.fci_scn[fci_variable_base]
                else:
                    self.fci_scn.load([fci_variable], upper_right_corner='NE', pad_data=False,)

        if 'oca' in scene_types:
            self.oca_scn = Scene(filenames=self.oca_filenames, reader='fci_l2_nc')
            self.oca_scn.load([self.oca_dataset], upper_right_corner='NE', pad_data=False,)

        if 'li' in scene_types:
            self.li_scn = Scene(filenames=self.li_filenames, reader='li_l2_nc')
            self.li_scn.load([self.li_dataset], upper_right_corner='NE', pad_data=False,)

        return

    def fill_li_array(self):
        logger.info("Filling LI array")
        self.li_scn[self.li_dataset] = self.li_scn[self.li_dataset].fillna(0)

    def fill_fci_array(self, fci_variables=None):
        """
        Fill FCI arrays with minimum values.

        Args:
            fci_variables: List of FCI variables to fill. If None, fills all loaded variables in fci_scn
        """

        if fci_variables is None:
            fci_variables = [key for key in self.fci_scn.keys()]

        logger.info(f"Filling FCI array for variables: {fci_variables}")

        for fci_dataset in fci_variables:
            if "epl_corr" in fci_dataset:
                logger.info("Filling epl_corr data with 0")
                self.fci_scn[fci_dataset] = self.fci_scn[fci_dataset].fillna(0)
            else:
                self.fci_scn[fci_dataset] = self.fci_scn[fci_dataset].fillna(da.nanmin(self.fci_scn[fci_dataset]))
        return

    def fix_oca_height(self):
        if (datetime.datetime.strptime(self.start_time, "%Y-%m-%dT%H:%M:%S") <
                datetime.datetime(2025, 8, 11, 9, 0, 0, 0)):
            logger.info("Fixing OCA height")
            # see https://user.eumetsat.int/news-events/news/mtg-level-2-product-enhancements
            with xr.open_dataset(settings.DATAGEN["fci_altitude_filepath"]) as altitude_data:
                self.oca_scn[self.oca_dataset].data = self.oca_scn[self.oca_dataset].data - np.flipud(
                    altitude_data['fci_altitude'].fillna(0))

    def get_resampled_scenes(self, scene_types=None):
        if scene_types is None:
            scene_types = ['fci', 'oca', 'li']
        logger.info(f"Getting resampled scenes for {scene_types}")
        if 'fci' in scene_types:
            self.fci_scn_res = self.fci_scn.resample(self.target_adef, resampler='nearest', radius_of_influence=5e4,
                                                     cache_dir=settings.DATAGEN["resampling_cache_folder"])
        if 'oca' in scene_types:
            self.oca_scn_res = self.oca_scn.resample(self.target_adef, resampler='nearest', radius_of_influence=5e4,
                                                     cache_dir=settings.DATAGEN["resampling_cache_folder"])
        if 'li' in scene_types:
            self.li_scn_res = self.li_scn.resample(self.target_adef, resampler='nearest', radius_of_influence=5e4,
                                                   cache_dir=settings.DATAGEN["resampling_cache_folder"])
        return

    def apply_parallax_correction(self, scene_types=None, fci_variables=None):
        """
        Apply parallax correction to FCI and LI scenes.

        Args:
            fci_variables: List of FCI variables to correct. If None, corrects all loaded variables in fci_scn_res
        """
        if scene_types is None:
            scene_types = ['fci', 'oca', 'li']
        logger.info(f"Applying parallax correction to {scene_types}")

        if fci_variables is None:
            fci_variables = [key for key in self.fci_scn_res.keys()]

        # add orbital parameters for the plax to work (missing in L2 data)
        # Use first available FCI variable to get orbital parameters
        first_fci_var = fci_variables[0]
        self.oca_scn_res[self.oca_dataset].attrs['orbital_parameters'] = (
            self.fci_scn_res[first_fci_var].attrs['orbital_parameters'])
        # self.oca_scn_res[self.oca_dataset].attrs['orbital_parameters']['satellite_actual_altitude'] /= 1e3
        # self.oca_scn_res[self.oca_dataset].attrs['orbital_parameters']['satellite_nominal_altitude'] /= 1e3
        # self.oca_scn_res[self.oca_dataset].attrs['orbital_parameters']['projection_altitude'] /= 1e3

        parallax_correction = ParallaxCorrection(self.target_adef)
        plax_corr_area = parallax_correction(self.oca_scn_res[self.oca_dataset])

        if 'fci' in scene_types:
            fci_scn_c_plax_res_satpy = self.fci_scn_res.resample(plax_corr_area, resampler='nearest', mask_area=True,
                                                                 radius_of_influence=10e6)

            fci_scn_c_plaxcorr_satpy = self.fci_scn_res.copy()
            for fci_dataset in fci_variables:
                fci_scn_c_plaxcorr_satpy[fci_dataset].data = fci_scn_c_plax_res_satpy[fci_dataset].data

            self.fci_scn_corr = fci_scn_c_plaxcorr_satpy.resample(self.target_adef,
                                                                  resampler='nearest', mask_area=True,
                                                                  radius_of_influence=10e6)

        # Only correct LI if it hasn't been done already
        if not self.append_mode:
            li_scn_c_plax_res_satpy = self.li_scn_res.resample(plax_corr_area, resampler='nearest', mask_area=True,
                                                               radius_of_influence=10e6)
            li_scn_c_plaxcorr_satpy = self.li_scn_res.copy()
            li_scn_c_plaxcorr_satpy[self.li_dataset].data = li_scn_c_plax_res_satpy[self.li_dataset].data
            self.li_scn_corr = li_scn_c_plaxcorr_satpy.resample(
                self.target_adef,
                resampler='nearest', mask_area=True,
                radius_of_influence=10e6)

        return

    @property
    def opera_s3_filename(self):

        dt = datetime.datetime.strptime(self.start_time, "%Y-%m-%dT%H:%M:%S")
        dt = dt + datetime.timedelta(minutes=10)

        filename = (f"{dt.strftime('%Y')}-{dt.strftime('%m')}-{dt.strftime('%d')}T{dt.strftime('%H%M%S')}Z"
                    f"-reflectivity-composite-opera.h5")

        s3_path = (f"s3://reflectivity.composite.opera.hdf5/"
                   f"{dt.strftime('%Y')}/{dt.strftime('%m')}/{dt.strftime('%d')}/{filename}")

        return s3_path

    def check_s3_file_presence(self, s3_path):
        path_without_prefix = s3_path.replace("s3://", "")

        exists = self.ewc_lampo_s3_accessor.fs.exists(path_without_prefix)

        if exists:
            logger.info(f"File {s3_path} found in S3 bucket")
        else:
            logger.info(f"File {s3_path} not found in  S3 bucket")

        return exists

    def process_opera(self):
        logger.info("Processing OPERA S3 file")
        if settings.DATAGEN['area_name'] == 'opera':
            datasets = self.ewc_lampo_s3_accessor.get_h5_dataset(self.opera_s3_filename,
                                                                 dataset_names=['dataset1/data1/data'])
            opera_maxrefl = datasets['dataset1/data1/data']
            opera_maxrefl[opera_maxrefl == -9999000] = np.nan

            # see https://www.eumetnet.eu/wp-content/uploads/2024/06/OPERA_Max-Reflectivity_Product-Sheet_Ed-2.0.pdf
            opera_maxrefl[opera_maxrefl < 0.12619] = 0

            self.opera_maxrefl = opera_maxrefl
        else:
            self.opera_maxrefl = np.full(self.target_adef.shape, np.nan)
        return

    @property
    def proc_dataset_filename_nc(self):
        return os.path.join(self.output_folder, f"proc_dataset_{self.start_time.replace(':', '')}.nc")

    @property
    def proc_dataset_filename_zarr(self):
        return os.path.join(self.output_folder, f"proc_dataset_{self.start_time.replace(':', '')}.zarr")

    @property
    def proc_dataset_path_s3_zarr(self):
        return (f"s3://{settings.DATAGEN['destine_bucket']}/{settings.DATAGEN['destine_prefix']}/"
                f"{self.start_time.replace(':', '')}/proc_dataset_{self.start_time.replace(':', '')}.zarr")

    def save_proc_dataset(self):
        logger.info("Extracting datasets into xarray")

        fci_xarray = self.fci_scn_corr.to_xarray(flatten_attrs=True, include_lonlats=False)
        li_xarray = self.li_scn_corr.to_xarray(flatten_attrs=True, include_lonlats=False)
        # oca_xarray = self.oca_scn_res.to_xarray(flatten_attrs=True, include_lonlats=False)

        opera_array = xr.DataArray(
            self.opera_maxrefl,
            dims=fci_xarray.dims,
            coords=fci_xarray.coords,
            name=self.opera_dataset  # Name of the new variable
        )

        # Combine the datasets
        self.combined_proc_ds = xr.merge([fci_xarray, li_xarray, opera_array],
                                         compat='override')

        self.combined_proc_ds = self.combined_proc_ds.compute()

        # # Save to netCDF without rechunking
        # self.combined_proc_ds.to_netcdf(self.proc_dataset_filename_nc)

        # TODO revise rechunking
        # Save to zarr with compression
        encoding = {var: {'compressors': [compressor], 'chunks': (1024, 1024)} for
                    var in self.combined_proc_ds.data_vars if var != "opera"}

        # Save locally
        logger.info(f"Saving dataset to local Zarr file: {self.proc_dataset_filename_zarr}")
        self.combined_proc_ds.to_zarr(self.proc_dataset_filename_zarr, mode='w', encoding=encoding)

        if self.save_to_s3:
            logger.info(f"Saving dataset to S3 Zarr file: {self.proc_dataset_path_s3_zarr}")
            self.combined_proc_ds.to_zarr(self.proc_dataset_path_s3_zarr, mode='w', encoding=encoding,
                                          storage_options=self.destine_lampo_s3_accessor.s3_config)

        return

    def append_new_variables_to_zarr(self, new_variables):
        """
        Append new FCI variables to existing zarr datasets without reprocessing everything.
        Reuses existing preprocessing methods.

        Args:
            new_variables: List of new variable names to process and append.
        """
        self.append_mode = True
        logger.info(f"Appending new variables {new_variables} to existing zarr dataset")

        # Reuse existing methods with specific variables
        self.get_scenes(scene_types = ['fci', 'oca'], fci_variables=new_variables)
        self.fill_fci_array(fci_variables=new_variables)
        self.get_resampled_scenes(scene_types=['fci', 'oca'])
        self.apply_parallax_correction(fci_variables=new_variables)

        # sata, satz, suna, sunz = get_angles(self.fci_scn_corr[new_variables[0]])
        #
        #
        # sunz_da = xr.DataArray(
        #     sunz,
        #     dims=self.fci_scn_corr[new_variables[0]].dims,
        #     coords=self.fci_scn_corr[new_variables[0]].coords,
        #     name='sunz'
        # )
        # self.fci_scn_corr['sunz'] = sunz_da

        # Convert to xarray
        logger.info("Converting new variables to xarray")
        new_fci_xarray = self.fci_scn_corr.to_xarray(flatten_attrs=True, include_lonlats=False)
        new_fci_xarray = new_fci_xarray.compute()

        # Prepare encoding for new variables
        encoding = {var: {'compressors': [compressor], 'chunks': (1024, 1024)}
                   for var in new_variables}

        # Append to local zarr
        logger.info(f"Appending new variables to local Zarr file: {self.proc_dataset_filename_zarr}")
        new_fci_xarray.to_zarr(self.proc_dataset_filename_zarr, mode='a', encoding=encoding)

        if self.save_to_s3:
            logger.info(f"Appending new variables to S3 Zarr file: {self.proc_dataset_path_s3_zarr}")
            new_fci_xarray.to_zarr(self.proc_dataset_path_s3_zarr, mode='a', encoding=encoding,
                                  storage_options=self.destine_lampo_s3_accessor.s3_config)

        logger.info(f"Successfully appended {new_variables} to zarr datasets")
        return

    def preprocess_data(self):
        try:
            self.get_scenes()
            self.fill_li_array()
            self.fill_fci_array()
            self.fix_oca_height()
            self.get_resampled_scenes()
            self.apply_parallax_correction()
            self.process_opera()
            self.data_processed = True
        except Exception as e:
            logger.error(f"Error processing data: {e}")
            self.data_processed = False

        return

    def is_timestep_processed(self):
        local_exists = os.path.exists(self.proc_dataset_filename_zarr)

        if local_exists:
            logger.info(f"Local zarr file exists: {self.proc_dataset_filename_zarr}")
            self.local_zarr_exists = True

        # if self.save_to_s3:
        #     if self.check_s3_file_presence(self.proc_dataset_path_s3_zarr):
        #         logger.info(f"S3 zarr file exists: {self.proc_dataset_path_s3_zarr}")
        #         self.s3_zarr_exists = True

        return local_exists  # or self.s3_zarr_exists

    def open_preprocessed_dataset_zarr(self):
        logger.info("Opening preprocessed dataset")
        self.combined_proc_ds = xr.open_zarr(self.proc_dataset_filename_zarr)
        return

    def process_patches(self):
        self.extract_patches()
        if len(self.patches_coords) == 0:
            logger.warning("No valid patches found")
            return
        self.compute_patch_statistics()
        self.save_patch_statistics_to_parquet()
        self.plot_extracted_patches()
        return

    def extract_patches(self):
        logger.info("Extracting patches")
        self.patches_coords = find_valid_patches(self.combined_proc_ds[self.opera_dataset].values,
                                                 self.patch_size, 200, max_tries=1e4)
        if len(self.patches_coords) == 0:
            return "No valid patches found"
        self.filter_patches_for_vza()

    def filter_patches_for_vza(self):
        lons, lats = self.target_adef.get_lonlat_from_array_coordinates(self.patches_coords[:, 1],
                                                                        self.patches_coords[:, 0])
        angles = _get_sensor_angles_ndarray(lons, lats, datetime.datetime(2000, 1, 1),
                                            0, 0, 35786400)
        self.patches_coords = self.patches_coords[angles[1, :] < 75]
        return

    def compute_patch_statistics(self):
        logger.info("Computing patch statistics")
        self.patch_statistics = {}

        for i, (row_start, col_start) in enumerate(self.patches_coords):
            self.patch_statistics[i] = {}

            datasets = {}

            row_end = row_start + self.patch_size
            col_end = col_start + self.patch_size

            coords = {
                'col_start': col_start,
                'col_end': col_end,
                'row_start': row_start,
                'row_end': row_end,
            }

            opera_patch = self.combined_proc_ds[self.opera_dataset][row_start:row_end, col_start:col_end]
            opera_stats = get_patch_stats(opera_patch)
            datasets[self.opera_dataset] = opera_stats | coords

            for fci_dataset in self.fci_datasets:
                fci_patch = self.combined_proc_ds[fci_dataset][row_start:row_end, col_start:col_end]
                fci_stats = get_patch_stats(fci_patch)
                datasets[fci_dataset] = fci_stats | coords

            li_patch = self.combined_proc_ds[self.li_dataset][row_start:row_end, col_start:col_end]
            li_stats = get_patch_stats(li_patch)
            datasets[self.li_dataset] = li_stats | coords

            # Store all dataset statistics in a 'datasets' key
            self.patch_statistics[i]['datasets'] = datasets

        logger.info(f"Computed statistics for {len(self.patches_coords)} patches across all datasets")

    def save_patch_statistics_to_parquet(self):
        logger.info("Saving patch statistics to parquet file")

        rows = []

        for patch_idx, patch_stats in self.patch_statistics.items():
            datasets = patch_stats.get('datasets', {})

            for dataset_name, dataset_stats in datasets.items():
                # Create a row for this patch and dataset
                row = {
                    'patch_idx': patch_idx,
                    'dataset': dataset_name,
                    'start_time': self.start_time,
                }
                row.update(dataset_stats)
                rows.append(row)

        self.patch_statistics_df = pd.DataFrame(rows)

        self.patch_statistics_df.to_parquet(self.patches_parquet_filename)
        logger.info(f"Saved patch statistics to {self.patches_parquet_filename}")

        if self.save_to_s3:
            # TODO refactor and cleanup
            s3_path = (f"s3://{settings.DATAGEN['destine_bucket']}/{settings.DATAGEN['destine_prefix']}/"
                       f"{self.start_time.replace(':', '')}/patch_statistics_{self.start_time.replace(':', '')}.parquet")
            logger.info(f"Saving patch statistics to S3: {s3_path}")
            self.patch_statistics_df.to_parquet(s3_path, storage_options=self.destine_lampo_s3_accessor.s3_config)
            logger.info(f"Successfully saved patch statistics to S3")

        return

    @property
    def patches_parquet_filename(self):
        return os.path.join(self.output_folder, f"patch_statistics_{self.start_time.replace(':', '')}.parquet")

    def open_patch_statistics_from_parquet(self, filename=None):
        logger.info(f"Opening patch statistics from {filename}")
        try:
            self.patch_statistics_df = pd.read_parquet(self.patches_parquet_filename)
            logger.info(
                f"Successfully read patch statistics with {len(self.patch_statistics_df.groupby('patch_idx'))} patches")
        except Exception as e:
            logger.error(f"Error reading patch statistics from {self.patch_statistics_df}: {e}")

    def plot_extracted_patches(self):
        plt.figure(figsize=(10, 10))
        plt.imshow(self.combined_proc_ds[self.opera_dataset].values, cmap=get_modified_cmap_for_radar())

        for i, (row_start, col_start) in enumerate(self.patches_coords):
            rect = patches.Rectangle((col_start, row_start), self.patch_size, self.patch_size,
                                     linewidth=2, edgecolor='r', facecolor='none')
            plt.gca().add_patch(rect)

        plt.colorbar(label='Reflectivity (dBZ)')
        plt.title(f'OPERA Max Reflectivity with Selected Patches - Start Time: {self.start_time}')
        plt.savefig(os.path.join(self.output_folder, f"patches_{self.start_time.replace(':', '')}.png"),
                    dpi=300, bbox_inches='tight')
        plt.close()
        return

    def get_combined_proc_ds(self):
        if self.combined_proc_ds is None:
            try:
                self.combined_proc_ds = xr.open_zarr(self.proc_dataset_filename_zarr, chunks=None)
            except Exception as e:
                logger.error(f"Error loading dataset for plotting: {e}")
                logger.info(f"Trying to load dataset from S3")
                self.combined_proc_ds = xr.open_zarr(self.proc_dataset_path_s3_zarr, chunks=None,
                                                     storage_options=self.destine_lampo_s3_accessor.s3_config)
                return None

    def plot_datasets(self, save_path=None):

        logger.info("Plotting datasets")

        self.get_combined_proc_ds()

        projection = self.target_adef.to_cartopy_crs()

        if settings.DATAGEN['area_name'] == 'ukraine':
            figsize = (15, 5)
        else:
            figsize = (15, 7)
        fig = plt.figure(figsize=figsize)

        # ir_105
        ax1 = fig.add_subplot(1, 3, 1, projection=projection)
        self.plot_ir_105_to_ax(ax1, projection)

        # LI AFA
        ax = fig.add_subplot(1, 3, 2, projection=projection)
        self.plot_li_to_ax(ax, projection)

        # OPERA
        ax3 = fig.add_subplot(1, 3, 3, projection=projection)
        self.plot_opera_to_ax(ax3, projection)

        plt.tight_layout()

        fig.suptitle(f'Dataset Overview - Start Time: {self.start_time}', fontsize=14, y=0.98)

        if save_path is None:
            save_path = os.path.join(self.output_folder, f"dataset_plot_{self.start_time.replace(':', '')}.png")
        logger.debug(f"Saving plot to {save_path}")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Plot saved")

        plt.close(fig)

        return

    def plot_ir_105_to_ax(self, ax, projection):
        add_coastlines_and_features(ax)
        _ = self.combined_proc_ds['ir_105'].plot(
            ax=ax,
            cmap='cividis',
            add_colorbar=True,
            cbar_kwargs={
                'label': 'Brightness Temperature (K)',
                'orientation': 'horizontal',
                'location': 'bottom',
                'fraction': 0.046,
                'pad': 0.01,
                'extend': 'both'
            },
            transform=projection,
            vmin=190,
            vmax=320,
        )
        ax.set_title('FCI IR_105')

    def plot_li_to_ax(self, ax, projection):
        add_coastlines_and_features(ax)
        modified_viridis = get_modified_cmap_for_li()
        _ = self.combined_proc_ds[self.li_dataset].plot(
            ax=ax,
            cmap=modified_viridis,
            add_colorbar=True,
            cbar_kwargs={'label': 'Counts', 'orientation': 'horizontal',
                         'location': 'bottom',
                         'fraction': 0.046,
                         'pad': 0.01, "extend": "max", },
            transform=projection,
            vmin=0,
            vmax=20,
        )
        ax.set_title('LI Accumulated Flash Area')

    def plot_opera_to_ax(self, ax, projection):
        add_coastlines_and_features(ax)
        modified_jet = get_modified_cmap_for_radar()
        _ = self.combined_proc_ds[self.opera_dataset].plot(
            ax=ax,
            cmap=modified_jet,
            add_colorbar=True,
            cbar_kwargs={'label': 'Reflectivity (dBZ)', 'orientation': 'horizontal',
                         'location': 'bottom',
                         'fraction': 0.046,
                         'pad': 0.01,
                         'extend': 'neither'},
            transform=projection,
            vmin=5,
            vmax=REFL_MAX_PLOT,
        )
        ax.set_title('OPERA Max Reflectivity (Target)')

        # # Add text box with nanmax value
        # max_value = float(np.nanmax(self.combined_proc_ds[self.opera_dataset].values))
        # textstr = f'Max: {max_value:.2f} dBZ'
        # props = dict(boxstyle='round', facecolor='white', alpha=0.8)
        # ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
        #         verticalalignment='top', bbox=props)

    def plot_prediction_to_ax(self, ax, projection, prediction_file):
        prediction = self.load_prediction_xr(prediction_file)

        modified_jet = get_modified_cmap_for_radar(first_color_alpha=0)

        opera_coverage = self.combined_proc_ds[self.opera_dataset] >= 0

        cmap, norm = get_cmap_for_opera_coverage()

        _ = opera_coverage.plot(
            ax=ax,
            cmap=cmap, norm=norm,
            transform=projection,
            add_colorbar=False,
        )

        add_coastlines_and_features(ax)
        _ = prediction.plot(
            ax=ax,
            cmap=modified_jet,
            add_colorbar=True,
            cbar_kwargs={'label': 'Reflectivity (dBZ)', 'orientation': 'horizontal',
                         'location': 'bottom',
                         'fraction': 0.046,
                         'pad': 0.01,
                         'extend': 'neither'
                         },
            transform=projection,
            vmin=5,
            vmax=REFL_MAX_PLOT,
        )
        ax.set_title('Predicted Max Reflectivity (Model Output)')

    def load_prediction_xr(self, prediction_file):
        prediction = np.load(prediction_file)
        prediction = prediction.squeeze()
        prediction = xr.DataArray(prediction, dims=self.combined_proc_ds.dims, coords=self.combined_proc_ds.coords)
        return prediction

    def plot_datasets_with_prediction(self, prediction_file, save_path=None, overwrite=False):
        logger.info(f"Plotting datasets with prediction from file: {prediction_file}")
        if save_path is None:
            save_path = os.path.join(self.output_folder,
                                     f"dataset_prediction_plot_{self.start_time.replace(':', '')}.png")

        if os.path.exists(save_path) and not overwrite:
            logger.info(f"Plot already exists at {save_path}. Skipping.")
            return

        self.get_combined_proc_ds()

        projection = self.target_adef.to_cartopy_crs()

        fig = plt.figure(figsize=(20, 7))

        ax = fig.add_subplot(1, 4, 1, projection=projection)
        self.plot_ir_105_to_ax(ax, projection)

        ax = fig.add_subplot(1, 4, 2, projection=projection)
        self.plot_li_to_ax(ax, projection)

        ax = fig.add_subplot(1, 4, 3, projection=projection)
        self.plot_prediction_to_ax(ax, projection, prediction_file)

        ax = fig.add_subplot(1, 4, 4, projection=projection)
        self.plot_opera_to_ax(ax, projection)


        fig.suptitle(f'Dataset Overview with Prediction - Start Time: {self.start_time}', fontsize=14, y=0.98)
        plt.tight_layout()


        logger.debug(f"Saving plot to {save_path}")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Plot saved")

        plt.close(fig)
