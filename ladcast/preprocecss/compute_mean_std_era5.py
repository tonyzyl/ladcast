import json

import xarray as xr

# Define the variable groups
surface_vars = [
    "2m_temperature",  # 2-m temperature
    "10m_u_component_of_wind",  # 10-m u wind component
    "10m_v_component_of_wind",  # 10-m v wind component
    "mean_sea_level_pressure",  # Mean sea level pressure
    "total_precipitation",  # Total precipitation
    "total_precipitation_6hr",
    "surface_pressure",  # Surface pressure
    "sea_surface_temperature",  # Sea surface temperature
]

atmospheric_vars = [
    "u_component_of_wind",  # U component of wind
    "v_component_of_wind",  # V component of wind
    "geopotential",  # Geopotential
    "specific_humidity",  # Specific humidity
    "vertical_velocity",  # Vertical wind speed
    "temperature",  # Temperature
]

static_vars = [
    "angle_of_sub_gridscale_orography",
    "anisotropy_of_sub_gridscale_orography",
    "geopotential_at_surface",
    "high_vegetation_cover",
    "lake_cover",
    "land_sea_mask",
    "low_vegetation_cover",
    "slope_of_sub_gridscale_orography",
    "soil_type",
    "standard_deviation_of_filtered_subgrid_orography",
    "standard_deviation_of_orography",
    "type_of_high_vegetation",
    "type_of_low_vegetation",
]


# Function to process and compute the mean and std for selected variables, considering different dimensions
def process_variables(dataset, variables):
    results = {}
    try:
        for var in variables:
            selected_data = dataset[var]

            # Determine which dimensions are present
            dims = selected_data.dims

            # Compute mean and standard deviation over appropriate dimensions
            if "time" in dims and "level" in dims:
                # Atmospheric variables with time and level (e.g., (time, level, longitude, latitude))
                mean_per_level = selected_data.mean(
                    dim=["time", "longitude", "latitude"], skipna=True
                ).compute()
                std_per_level = selected_data.std(
                    dim=["time", "longitude", "latitude"], skipna=True
                ).compute()

                # Store the mean and std by level
                results[var] = {
                    "mean": {
                        int(level): float(mean_per_level.sel(level=level))
                        for level in mean_per_level["level"].values
                    },
                    "std": {
                        int(level): float(std_per_level.sel(level=level))
                        for level in std_per_level["level"].values
                    },
                }

            elif "time" in dims:
                # Surface variables with time (e.g., (time, longitude, latitude))
                mean = selected_data.mean(
                    dim=["time", "longitude", "latitude"], skipna=True
                ).compute()
                std = selected_data.std(
                    dim=["time", "longitude", "latitude"], skipna=True
                ).compute()

                # Convert mean and std to float for JSON serialization
                mean = float(mean)
                std = float(std)

                # Store results as scalar mean and std
                results[var] = {"mean": mean, "std": std}

            else:
                # Static variables without time (e.g., (longitude, latitude))
                mean = selected_data.mean(skipna=True).compute()
                std = selected_data.std(skipna=True).compute()

                # Convert mean and std to float for JSON serialization
                mean = float(mean)
                std = float(std)

                # Store results as scalar mean and std
                results[var] = {"mean": mean, "std": std}

        return results

    except Exception as e:
        print(f"Error processing variables {variables}: {e}")
        return None


# Function to append results to a JSON file
def append_results_to_file(results):
    try:
        with open("ERA5_normal_1979_2017.json", "a") as f:
            json.dump(results, f, indent=4)
            f.write("\n")  # Ensure each result is on a new line
    except Exception as e:
        print(f"Error writing to ERA5_normal.json: {e}")


# Process the ERA5 dataset
def process_era5_dataset():
    try:
        # Open the dataset with Dask chunks
        ds = xr.open_zarr("ERA5_path")
        # ds = ds.unify_chunks()

        # Select the time range from January 1, 1979, onward
        red_ds = ds.sel(time=slice("1979-01-01", "2017-12-31"))
        # red_ds = ds.sel(time=slice('1979-01-01', '1979-01-15'))

        # Process static variables
        print("Processing static variables...")
        static_results = process_variables(red_ds, static_vars)
        if static_results is not None:
            append_results_to_file(static_results)

        # Process surface variables
        print("Processing surface variables...")
        surface_results = process_variables(red_ds, surface_vars)
        if surface_results is not None:
            append_results_to_file(surface_results)

        # Process atmospheric variables
        print("Processing atmospheric variables...")
        atmospheric_results = process_variables(red_ds, atmospheric_vars)
        if atmospheric_results is not None:
            append_results_to_file(atmospheric_results)

    except Exception as e:
        print(f"Error processing ERA5 dataset: {e}")


# Run the function to process the dataset
process_era5_dataset()
