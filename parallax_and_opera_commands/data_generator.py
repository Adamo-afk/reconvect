import argparse
import gc
import multiprocessing
import os
import time
from datetime import datetime, timedelta
from functools import partial

import lampo_radar.settings as settings
from lampo_radar.dataset_tools.dataset_gen.data_preprocessor import DataPreprocessor
from lampo_radar.logger import setup_logger
from mtgi_oi_tools.download_tools.download_from_archive import EumdacDownloader

logger = setup_logger(__name__)


# TODO handle satzen limit of OCA

class DataGenerator:
    time_window = 10  # minutes

    def __init__(self, start_time, output_folder, save_to_s3=False, overwrite=False, wait_for_files_timeout=0):
        self.start_time = start_time
        self.output_folder = output_folder
        self.output_folder_timestamp = os.path.join(self.output_folder, start_time.replace(':', ''))

        self.eumdac_data_folder = os.path.join(self.output_folder, start_time.replace(':', ''), "eumdac_data")
        self.save_to_s3 = save_to_s3
        self.overwrite = overwrite
        self.wait_for_files_timeout = wait_for_files_timeout  # in minutes

        self.end_time = (datetime.strptime(start_time, "%Y-%m-%dT%H:%M:%S") +
                         timedelta(minutes=self.time_window)).strftime("%Y-%m-%dT%H:%M:%S")

        self.data_preprocessor = DataPreprocessor(self.start_time,
                                                  self.eumdac_data_folder,
                                                  self.output_folder,
                                                  save_to_s3=self.save_to_s3)

        self.is_data_downloaded = None
        self.is_data_processed = None

        return

    def initialise_output_folder(self):
        os.makedirs(self.output_folder_timestamp, exist_ok=True)

    def download_data_for_timestep(self):
        eumdac_downloader = EumdacDownloader(
            settings.get_secret("EUMDAC_KEY"),
            settings.get_secret("EUMDAC_SECRET"),
        )

        collection_ids = [
            "EO:EUM:DAT:0687",  # LI AFA
            "EO:EUM:DAT:0684",  # OCA
            "EO:EUM:DAT:0662",  # FDHSI
            "EO:EUM:DAT:0665",  # HRFI
        ]

        for collection_id in collection_ids:
            n_products = eumdac_downloader.search_products_for_collection(collection_id, self.start_time, self.end_time)
            if n_products == 0:
                logger.info(
                    f"No products found for collection {collection_id} in time range {self.start_time} - {self.end_time}")
                self.is_data_downloaded = False
                return

        file_endings = ['.nc']
        # geographical bounds of search area (ll lon, ll lat, ur lon, ur lat)
        lonlat_bbox = [-10.5, 31.8, 57.9, 72]

        n_parallel_downloads = settings.DATAGEN.get("n_parallel_eumdac_downloads", 1)
        eumdac_downloader.download_products_for_collections(collection_ids, self.eumdac_data_folder, '', file_endings,
                                                            fci_l1c_chunks_lonlat_bbox=lonlat_bbox,
                                                            n_parallel_downloads=n_parallel_downloads)
        self.is_data_downloaded = True
        return

    def append_variables_to_timestep(self, new_variables=None):
        """
        Append new variables to existing zarr dataset for this timestep.

        Args:
            new_variables: List of variables to append. If None, uses ['vis_06', 'nir_16', 'nir_22']
        """

        if not self.overwrite and self.data_preprocessor.is_timestep_processed():
            logger.info(f"Timestep {self.start_time} already processed and overwrite=False. Skipping.")
            return

        self.download_data_for_timestep()
        self.data_preprocessor.initialise_data()
        if not self.data_preprocessor.data_initialised:
            logger.warning("Data could not be initialised for variable appending.")
            return

        try:
            self.data_preprocessor.append_new_variables_to_zarr(new_variables=new_variables)
            logger.info(f"Successfully appended variables to timestep {self.start_time}")
        except Exception as e:
            logger.error(f"Error appending variables to timestep {self.start_time}: {e}", exc_info=True)

        self.cleanup()
        if settings.DATAGEN["cleanup_ds_data_after_processing"]:
            self.remove_eumdac_data()
        return

    def process_data_for_timestep(self):
        if not self.overwrite and self.data_preprocessor.is_timestep_processed():
            logger.info(f"Timestep {self.start_time} already processed and overwrite=False. Skipping.")
            return

        # Wait for files with retry logic if timeout is configured
        timeout_seconds = self.wait_for_files_timeout * 60
        start_time = time.time()
        data_initialized = False

        while True:
            self.data_preprocessor.initialise_data()
            if not self.data_preprocessor.data_initialised:
                self.download_data_for_timestep()
                if not self.is_data_downloaded:
                    logger.warning("Data could not be successfully downloaded.")
                else:
                    self.data_preprocessor.initialise_data()
                    if not self.data_preprocessor.data_initialised:
                        logger.warning("Data could not be successfully initialised after attempted download.")

                # Check if we should retry
                if not self.data_preprocessor.data_initialised:
                    elapsed = time.time() - start_time
                    if timeout_seconds > 0 and elapsed < timeout_seconds:
                        remaining = timeout_seconds - elapsed
                        logger.info(f"Waiting for data files to appear. Time remaining: {remaining:.0f} seconds. Sleeping for 60 seconds...")
                        time.sleep(60)
                        continue
                    else:
                        logger.warning("Timeout reached or no retry configured. Exiting.")
                        return
                else:
                    logger.info("Data initialised")
                    data_initialized = True
                    break
            else:
                logger.info("Data initialised")
                data_initialized = True
                break

        if not data_initialized:
            logger.warning("Data could not be successfully initialised. Giving up. Exiting.")
            return

        self.data_preprocessor.preprocess_data()
        if not self.data_preprocessor.data_processed:
            logger.warning("Data not processed")
            return

        self.data_preprocessor.save_proc_dataset()

        # self.data_preprocessor.open_preprocessed_dataset_zarr()
        self.data_preprocessor.process_patches()

        return

    def plot_datasets(self):
        self.data_preprocessor.plot_datasets()

    def remove_eumdac_data(self):
        import shutil

        if os.path.exists(self.eumdac_data_folder):
            logger.info(f"Removing EUMDAC data folder: {self.eumdac_data_folder}")
            try:
                shutil.rmtree(self.eumdac_data_folder)
                logger.info(f"Successfully removed EUMDAC data folder")
            except Exception as e:
                logger.error(f"Error removing EUMDAC data folder: {e}")
        else:
            logger.info(f"EUMDAC data folder does not exist: {self.eumdac_data_folder}")

    def cleanup(self):
        logger.debug("Cleaning up")
        del self.data_preprocessor
        gc.collect()
        return

    def run_complete_timestep(self):
        self.process_data_for_timestep()
        if self.data_preprocessor.data_processed:
            self.plot_datasets()
        self.cleanup()
        if settings.DATAGEN["cleanup_ds_data_after_processing"]:
            self.remove_eumdac_data()


class DataGeneratorRunner:
    def __init__(self, start_datetime, end_datetime, time_step_delta, output_folder, save_to_s3=False,
                 num_processes=None, overwrite=False, append_variables_mode=False, append_variables=None,
                 skip_if_folder_exists=False, wait_for_files_timeout=0):
        self.start_datetime = datetime.strptime(start_datetime, "%Y-%m-%dT%H:%M:%S")
        self.end_datetime = datetime.strptime(end_datetime, "%Y-%m-%dT%H:%M:%S")
        self.time_step_delta = time_step_delta
        self.output_folder = output_folder
        self.save_to_s3 = save_to_s3
        self.overwrite = overwrite
        self.append_variables_mode = append_variables_mode
        self.append_variables = append_variables
        self.skip_if_folder_exists = skip_if_folder_exists
        self.wait_for_files_timeout = wait_for_files_timeout
        self.current_datetime = self.start_datetime
        self.num_processes = num_processes if num_processes is not None else settings.DATAGEN["default_num_processes"]

    @staticmethod
    def process_timestep(datetime_str, output_folder, save_to_s3, overwrite=False,
                        append_variables_mode=False, append_variables=None, skip_if_folder_exists=False,
                        wait_for_files_timeout=0):
        logger.info(f"Processing timestep: {datetime_str}")
        try:
            data_generator = DataGenerator(datetime_str, output_folder, save_to_s3=save_to_s3, overwrite=overwrite,
                                          wait_for_files_timeout=wait_for_files_timeout)
            if skip_if_folder_exists and os.path.exists(data_generator.output_folder_timestamp):
                logger.info(f"Output folder already exists for timestep: {datetime_str}. Skipping.")
                return
            data_generator.initialise_output_folder()

            if append_variables_mode:
                logger.info(f"Appending variables mode for timestep: {datetime_str}")
                data_generator.append_variables_to_timestep(new_variables=append_variables)
            else:
                data_generator.run_complete_timestep()

            del data_generator
            return f"Successfully processed timestep: {datetime_str}"
        except Exception as e:
            logger.error(f"Error processing data for time step {datetime_str}: {e}")
            return f"Error processing timestep: {datetime_str} - {str(e)}"

    def run(self):
        timesteps = []
        current = self.start_datetime
        while current <= self.end_datetime:
            timesteps.append(current.strftime("%Y-%m-%dT%H:%M:%S"))
            current += self.time_step_delta

        if self.num_processes > 1:
            with multiprocessing.get_context('spawn').Pool(self.num_processes) as pool:
                process_func = partial(self.process_timestep,
                                       output_folder=self.output_folder,
                                       save_to_s3=self.save_to_s3,
                                       overwrite=self.overwrite,
                                       append_variables_mode=self.append_variables_mode,
                                       append_variables=self.append_variables,
                                       skip_if_folder_exists=self.skip_if_folder_exists,
                                       wait_for_files_timeout=self.wait_for_files_timeout)
                pool.map(process_func, timesteps, 1)
        else:
            for ts in timesteps:
                self.process_timestep(ts, output_folder=self.output_folder,
                                       save_to_s3=self.save_to_s3,
                                       overwrite=self.overwrite,
                                       append_variables_mode=self.append_variables_mode,
                                       append_variables=self.append_variables,
                                       skip_if_folder_exists=self.skip_if_folder_exists,
                                       wait_for_files_timeout=self.wait_for_files_timeout)

        logger.info("Processing complete.")


def main(args):

    settings.load_config(args.config)

    # Convert time_step_minutes to timedelta
    time_step_delta = timedelta(minutes=args.time_step_minutes)

    # Parse append_variables if provided
    append_variables = None
    if hasattr(args, 'append_variables') and args.append_variables:
        append_variables = args.append_variables

    data_generator_runner = DataGeneratorRunner(
        args.start_datetime, args.end_datetime, time_step_delta, args.output_folder,
        save_to_s3=args.save_to_s3, num_processes=args.num_processes, overwrite=args.overwrite,
        append_variables_mode=args.append_variables_mode if hasattr(args, 'append_variables_mode') else False,
        append_variables=append_variables,
        skip_if_folder_exists=args.skip_if_folder_exists,
        wait_for_files_timeout=args.wait_for_files_timeout
    )

    data_generator_runner.run()


if __name__ == "__main__":
   # from satpy.utils import debug_on; debug_on()

    parser = argparse.ArgumentParser(description='Run the DataGeneratorRunner to process data for multiple timesteps.')
    parser.add_argument('--config', type=str, default=str(settings.DEFAULT_CONFIG_PATH),
                        help='Path to YAML config file')
    parser.add_argument('--start-datetime', type=str, required=True,
                        help='Start datetime in format YYYY-MM-DDTHH:MM:SS')
    parser.add_argument('--end-datetime', type=str, required=True,
                        help='End datetime in format YYYY-MM-DDTHH:MM:SS')
    parser.add_argument('--time-step-minutes', type=int, default=10,
                        help='Time step in minutes (default: 10)')
    parser.add_argument('--output-folder', type=str, required=True,
                        help='Output folder path')
    parser.add_argument('--save-to-s3', action='store_true',
                        help='Save data to S3 (default: False)')
    parser.add_argument('--num-processes', type=int, default=settings.DATAGEN["default_num_processes"],
                        help=f'Number of processes to use (default: {settings.DATAGEN["default_num_processes"]})')
    parser.add_argument('--overwrite', action='store_true',
                        help='Overwrite existing data (default: False)')
    parser.add_argument('--skip-if-folder-exists', action='store_true',
                        help='Skip processing if output folder already exists (default: False)')
    parser.add_argument('--append-variables-mode', action='store_true',
                        help='Append new variables to existing zarr datasets instead of full processing (default: False)')
    parser.add_argument('--append-variables', nargs='+', default=None,
                        help='List of variables to append (e.g., vis_06 nir_16 nir_22). If not specified, defaults to vis_06 nir_16 nir_22')
    parser.add_argument('--wait-for-files-timeout', type=int, default=0,
                        help='Timeout in minutes to wait for data files to appear. Will retry every 30 seconds until timeout. Default: 0 (no retry)')

    args = parser.parse_args()

    main(args)
