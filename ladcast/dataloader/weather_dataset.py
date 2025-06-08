import io
import os
import tarfile
from datetime import datetime, timedelta
from typing import List, Optional

import datasets
import numpy as np
import torch
from datasets import Array3D, DatasetInfo, Features, Value

from ladcast.models.embeddings import convert_timestamp_to_int

# Constants
_TAR_FILES_PATH = "path_to_tar_files"
TRAIN_START = 1979
TRAIN_END = 2017
VAL_START = 2018
VAL_END = 2018
TEST_START = 2022
TEST_END = 2022
FULL_START = 1979
FULL_END = 2022


def _get_monthly_tar_files(start_year, end_year):
    """Generate paths to monthly tar files for the given year range."""
    files = []
    for year in range(start_year, end_year + 1):
        for month in range(1, 13):
            tar_path = f"{_TAR_FILES_PATH}/{year}_{month:02d}.tar"
            if os.path.exists(tar_path):
                files.append(tar_path)
    return files


# Define data files for each split using monthly tar files
_DATA_FILES = {
    "train": _get_monthly_tar_files(TRAIN_START, TRAIN_END),
    "validation": _get_monthly_tar_files(VAL_START, VAL_END),
    "test": _get_monthly_tar_files(TEST_START, TEST_END),
    "full": _get_monthly_tar_files(FULL_START, FULL_END),
    "2018": _get_monthly_tar_files(2018, 2018),
    "2019": _get_monthly_tar_files(2019, 2019),
    "2020": _get_monthly_tar_files(2020, 2020),
    "2021": _get_monthly_tar_files(2021, 2021),
    "2022": _get_monthly_tar_files(2022, 2022),
}


class WeatherDataset(datasets.GeneratorBasedBuilder):
    """Weather dataset builder for ERA5 data stored in monthly tar files."""

    VERSION = datasets.Version("3.2.0")
    DEFAULT_WRITER_BATCH_SIZE = 1000

    def _info(self) -> DatasetInfo:
        """Returns the dataset metadata."""
        n_surface_vars = 7
        n_atmos_vars = 6
        n_levels = 13
        total_features = n_surface_vars + (n_atmos_vars * n_levels)

        return DatasetInfo(
            description="ERA5 Weather Dataset containing atmospheric and surface variables",
            features=Features(
                {
                    "data": Array3D(shape=(total_features, 121, 240), dtype="float32"),
                    "timestamp": Value("int32"),
                }
            ),
            supervised_keys=None,
        )

    def _split_generators(
        self, dl_manager: datasets.DownloadManager
    ) -> List[datasets.SplitGenerator]:
        """Returns SplitGenerators."""
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "archives": [
                        dl_manager.iter_archive(archive)
                        for archive in _DATA_FILES["train"]
                    ],
                    "split": "train",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "archives": [
                        dl_manager.iter_archive(archive)
                        for archive in _DATA_FILES["validation"]
                    ],
                    "split": "validation",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "archives": [
                        dl_manager.iter_archive(archive)
                        for archive in _DATA_FILES["test"]
                    ],
                    "split": "test",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split("full"),
                gen_kwargs={
                    "archives": [
                        dl_manager.iter_archive(archive)
                        for archive in _DATA_FILES["full"]
                    ],
                    "split": "full",
                },
            ),
            datasets.SplitGenerator(
                name="2018",
                gen_kwargs={
                    "archives": [
                        dl_manager.iter_archive(archive)
                        for archive in _DATA_FILES["2018"]
                    ],
                    "split": "2018",
                },
            ),
            datasets.SplitGenerator(
                name="2019",
                gen_kwargs={
                    "archives": [
                        dl_manager.iter_archive(archive)
                        for archive in _DATA_FILES["2019"]
                    ],
                    "split": "2019",
                },
            ),
            datasets.SplitGenerator(
                name="2020",
                gen_kwargs={
                    "archives": [
                        dl_manager.iter_archive(archive)
                        for archive in _DATA_FILES["2020"]
                    ],
                    "split": "2020",
                },
            ),
            datasets.SplitGenerator(
                name="2021",
                gen_kwargs={
                    "archives": [
                        dl_manager.iter_archive(archive)
                        for archive in _DATA_FILES["2021"]
                    ],
                    "split": "2021",
                },
            ),
            datasets.SplitGenerator(
                name="2022",
                gen_kwargs={
                    "archives": [
                        dl_manager.iter_archive(archive)
                        for archive in _DATA_FILES["2022"]
                    ],
                    "split": "2022",
                },
            ),
        ]

    def _generate_examples(self, archives, split):
        """Yields examples."""

        idx = 0
        for archive in archives:
            for path, file in archive:
                if path.endswith(".npy"):
                    # Extract timestamp from filename
                    timestamp_int = convert_timestamp_to_int(path.replace(".npy", ""))

                    # Create BytesIO object from file content
                    buffer = io.BytesIO(file.read())
                    tensor = torch.from_numpy(np.load(buffer))

                    # Since datasets expects numpy arrays or lists, we convert the tensor
                    # but keep it as float32 numpy array for efficiency
                    yield (
                        idx,
                        {
                            "data": tensor,  # Convert to numpy array instead of list
                            "timestamp": timestamp_int,
                        },
                    )
                    idx += 1

                    buffer.close()
                    del buffer, tensor
                else:
                    raise ValueError(f"Unexpected file in archive: {path}")


def weather_dataset_preprocess_batch(
    batch,
    mean,
    std,
    crop_south_pole: bool = True,
    sst_channel_idx: Optional[int] = None,
    incl_sur_pressure: bool = True,
):
    # batch (B, C, H, W)
    if crop_south_pole:
        batch = batch[..., 1:, :]  # H: lat starts from -90
    if not incl_sur_pressure:
        batch = batch[:, :-1, ...]  # Remove surface pressure
    batch = (batch - mean) / std
    if sst_channel_idx is not None:
        nan_mask = torch.isnan(batch[:, sst_channel_idx, ...])  # (B, H, W)
        batch[:, sst_channel_idx, ...][
            nan_mask
        ] = -2  # GenCast mask as min. value, we consider -2 is out of distribution for normalized sst
        return batch, nan_mask
    else:
        return batch


def weather_dataset_postprocess_batch(batch, mean, std):
    # batch (B, C, H, W)
    return batch * std + mean


def read_tar_files(start_ts, end_ts, dh=1, shape=None, tar_dir=_TAR_FILES_PATH):
    """
    Reads npy files stored in monthly tar archives.

    Parameters:
        start_ts (str): Start timestamp in 'YYYYMMDDHH' format.
        end_ts (str): End timestamp in 'YYYYMMDDHH' format.
        dh (int): Hour interval between selections.
        shape (tuple): Expected shape of each npy file.
        tar_dir (str): Directory where tar files (named 'YYYY_MM.tar') are stored.

    Returns:
        tuple: (result, time_list) where result is a numpy array of shape (N, *shape) and time_list is
               a list of timestamps in 'YYYYMMDDHH' format.

    Raises:
        FileNotFoundError: If the tar file or npy file is missing.
        ValueError: If the loaded npy file doesn't match the expected shape.
    """
    # Convert timestamps to datetime objects
    start_dt = datetime.strptime(start_ts, "%Y%m%d%H")
    end_dt = datetime.strptime(end_ts, "%Y%m%d%H")

    # Generate a list of datetimes at interval dh hours
    dt_list = []
    cur_dt = start_dt
    while cur_dt <= end_dt:
        dt_list.append(cur_dt)
        cur_dt += timedelta(hours=dh)

    n_times = len(dt_list)
    # Preallocate the output array (adjust dtype if needed)
    result = np.empty((n_times,) + shape, dtype=np.float32)

    # Cache opened tar files by year_month
    tar_files = {}

    try:
        for i, cur_dt in enumerate(dt_list):
            print(cur_dt)
            year = cur_dt.year
            month = cur_dt.month
            year_month_key = f"{year}_{month:02d}"

            # Open tar file for the year_month if not already opened
            if year_month_key not in tar_files:
                tar_path = os.path.join(tar_dir, f"{year_month_key}.tar")
                if not os.path.exists(tar_path):
                    raise FileNotFoundError(f"Tar file '{tar_path}' does not exist.")
                tar_files[year_month_key] = tarfile.open(tar_path, "r")
            tar = tar_files[year_month_key]

            # Construct the file name inside the tar archive: "YYYY-MM-DDTHH.npy"
            file_name = cur_dt.strftime("%Y-%m-%dT%H.npy")
            try:
                member = tar.getmember(file_name)
            except KeyError:
                raise FileNotFoundError(
                    f"File '{file_name}' not found in tar archive for {year_month_key}."
                )

            # Extract the file object from the tar archive
            f = tar.extractfile(member)
            if f is None:
                raise FileNotFoundError(
                    f"Could not extract file '{file_name}' from tar archive for {year_month_key}."
                )

            # Read the file content into a BytesIO object to support np.load
            buffer = io.BytesIO(f.read())
            data = np.load(buffer)
            buffer.close()

            # Optional: Validate the shape of the loaded data
            if shape and data.shape != shape:
                raise ValueError(
                    f"Shape mismatch for file '{file_name}': expected {shape}, got {data.shape}."
                )
            result[i] = data
    finally:
        # Close all tar files
        for tar in tar_files.values():
            tar.close()

    # Create list of timestamps in the same format as input
    time_list = [dt.strftime("%Y%m%d%H") for dt in dt_list]
    return result, time_list
