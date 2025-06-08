import argparse
import gc
import io
import tarfile
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import pandas as pd
import xarray as xr
from tqdm import tqdm


def save_xarray_to_tar(
    ds: xr.Dataset,
    surface_vars: list,
    atmospheric_vars: list,
    output_path: str,
    years: Union[List[int], range],
    months: Optional[List[int]] = None,
):
    """
    Save xarray Dataset slices to tar files for specified years and months, processing one time slice at a time.

    Parameters:
    -----------
    ds : xr.Dataset
        The input dataset
    surface_vars : list
        List of surface variable names
    atmospheric_vars : list
        List of atmospheric variable names
    output_path : str
        Path to save the output tar files
    years : Union[List[int], range]
        Years to process
    months : Optional[List[int]]
        Specific months to process (if None, process all 12 months)
    """
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    # Pre-create lazy concatenations for surface and atmospheric data
    surface_data = xr.concat([ds[var] for var in surface_vars], dim="channel")
    atmospheric_data = xr.concat([ds[var] for var in atmospheric_vars], dim="tmp")
    atmospheric_data = atmospheric_data.stack(channel=("tmp", "level"))
    atmospheric_data = atmospheric_data.reset_index("tmp").drop_vars("tmp")

    # Retrieve level and assign coordinate names lazily
    level_values = ds["level"].values
    surface_feature_names = surface_vars
    atmospheric_feature_names = [
        f"{var}_level_{level}" for var in atmospheric_vars for level in level_values
    ]
    surface_data = surface_data.assign_coords(channel=surface_feature_names)
    atmospheric_data = atmospheric_data.assign_coords(channel=atmospheric_feature_names)

    # Check if 1979 is included and ensure proper starting time
    if 1979 in years:
        earliest_1979_time = np.datetime64("1979-01-01T05:00:00")
        if ds.time.min().values < earliest_1979_time:
            print(
                f"Warning: For year 1979, filtering data to start from {earliest_1979_time}"
            )
            ds = ds.sel(time=slice(earliest_1979_time, None))

    for year in years:
        # Select times for the current year lazily
        year_data = ds.sel(time=str(year))

        if months is None:
            # Process all months (1-12)
            months_to_process = range(1, 13)
        else:
            # Process specific months
            months_to_process = months

        # Process each month separately
        for month in months_to_process:
            month_data = year_data.sel(time=f"{year}-{month:02d}")
            if len(month_data.time) > 0:  # Only process if there's data for this month
                tar_path = output_path / f"{year}_{month:02d}.tar"
                _process_data_to_tar(
                    month_data,
                    tar_path,
                    surface_data,
                    atmospheric_data,
                    f"Processing {year}-{month:02d}",
                )


def _process_data_to_tar(
    data_slice, tar_path, surface_data, atmospheric_data, progress_desc
):
    """Helper function to process data slices to tar files"""
    with tarfile.open(tar_path, "w") as tar:
        time_values = data_slice.time.values
        for cur_datetime in tqdm(time_values, desc=progress_desc):
            # Process one time slice at a time to minimize memory use
            timestamp = pd.Timestamp(cur_datetime)
            filename = timestamp.strftime("%Y-%m-%dT%H") + ".npy"

            # Select surface and atmospheric slices lazily for the current time
            time_slice_surface = surface_data.sel(time=cur_datetime)
            time_slice_atmo = atmospheric_data.sel(time=cur_datetime)

            # Concatenate along 'channel' for this time slice
            time_slice_combined = xr.concat(
                [time_slice_atmo, time_slice_surface], dim="channel"
            ).transpose("channel", "latitude", "longitude")

            # Convert to NumPy array (this triggers actual data loading for this slice)
            np_data = time_slice_combined.values.astype(np.float32)

            # Save NumPy array to buffer
            buffer = io.BytesIO()
            np.save(buffer, np_data)
            buffer.seek(0)

            # Create tar info and add to tar
            info = tarfile.TarInfo(name=filename)
            info.size = buffer.getbuffer().nbytes
            tar.addfile(info, buffer)

            # Cleanup for the current slice
            del (
                time_slice_surface,
                time_slice_atmo,
                time_slice_combined,
                np_data,
                buffer,
            )
            gc.collect()


def main():
    parser = argparse.ArgumentParser(
        description="Process ERA5 data for specified years and optionally specific months."
    )
    parser.add_argument(
        "--years",
        type=int,
        nargs="+",
        required=True,
        help="One or more years to process (e.g., --years 1979 1980 1981)",
    )
    parser.add_argument(
        "--months",
        type=int,
        nargs="+",
        help="Optional: Specific months to process (1-12). If omitted, processes all months.",
    )
    parser.add_argument(
        "--ds_path", type=str, required=True, help="Path to the Zarr dataset"
    )
    parser.add_argument(
        "--output_folder", type=str, required=True, help="Output folder for tar files"
    )
    args = parser.parse_args()

    # Validate months if provided
    if args.months:
        for month in args.months:
            if month < 1 or month > 12:
                parser.error(
                    f"Invalid month value: {month}. Months must be between 1 and 12."
                )

    # Open dataset with lazy Dask chunks to avoid loading entire data into memory
    ds = xr.open_dataset(
        args.ds_path,
        engine="zarr",
        chunks={"time": 1, "latitude": 121, "longitude": 240},
    ).sel(time=slice(str(min(args.years)), str(max(args.years) + 1)))

    surface_vars = [
        "10m_u_component_of_wind",
        "10m_v_component_of_wind",
        "2m_temperature",
        "mean_sea_level_pressure",
        "sea_surface_temperature",
        "total_precipitation_6hr",
        "surface_pressure",
    ]
    atmospheric_vars = [
        "geopotential",
        "specific_humidity",
        "temperature",
        "u_component_of_wind",
        "v_component_of_wind",
        "vertical_velocity",
    ]

    save_xarray_to_tar(
        ds, surface_vars, atmospheric_vars, args.output_folder, args.years, args.months
    )


if __name__ == "__main__":
    main()

"""
example usage:
python xr2webdataset.py \
    --years 1990 1991 1992 1993 1994 1995 1996 1997 1998 1999 2000 \
    --ds_path "ERA5_path.zarr" \
    --output_folder "tar_files"

"""
