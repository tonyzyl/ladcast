import xarray as xr

ds = xr.open_zarr(
    "gs://weatherbench2/datasets/era5/1959-2023_01_10-1h-240x121_equiangular_with_poles_conservative.zarr/"
)

# Select the time range from January 1, 1979, onward
red_ds = ds.sel(time=slice("1979-01-01", None))

surface_vars = [
    "2m_temperature",  # 2-m temperature
    "10m_u_component_of_wind",  # 10-m u wind component
    "10m_v_component_of_wind",  # 10-m v wind component
    "mean_sea_level_pressure",  # Mean sea level pressure
    "sea_surface_temperature",  # Sea surface temperature
    "total_precipitation",  # Total precipitation
    "surface_pressure",  # Surface pressure
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

desired_vars = surface_vars + atmospheric_vars + static_vars
red_ds = red_ds[desired_vars]
red_ds = red_ds.chunk(chunks="auto").unify_chunks()

# depending on your zarr version, you might need to specify: zarr_format=2
# red_ds.to_zarr("save_path.zarr")
