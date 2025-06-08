import importlib

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np
import torch
import xarray as xr
from diffusers.utils import logging
from matplotlib.animation import FuncAnimation, PillowWriter

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


def get_static_tensor(
    dataset, static_name_list, path_to_static_zarr="ERA5_static.zarr"
):
    """
    Generate a static conditioning tensor from the ERA5 dataset.
    Return (1, C, H, W)
    """
    static_mean_tensor, static_std_tensor = dataset.precompute_mean_std(
        dataset.normalization_param_dict, static_name_list
    )
    static_conditioning_tensor = xr.open_zarr(path_to_static_zarr).load()
    if dataset.crop_south_pole:
        static_conditioning_tensor = static_conditioning_tensor.isel(
            latitude=slice(1, None)
        )
    static_conditioning_tensor = (
        static_conditioning_tensor.to_array(dim="channel", name=None)
        .transpose("channel", "latitude", "longitude")
        .values
    )
    static_conditioning_tensor = torch.as_tensor(
        static_conditioning_tensor
    )  # (C, H, W)
    static_conditioning_tensor = dataset._normalize_tensor(
        static_conditioning_tensor, static_mean_tensor, static_std_tensor
    )
    return static_conditioning_tensor.unsqueeze(0)  # (1, C, H, W)


def get_obj_from_str(string):
    module, cls = string.rsplit(".", 1)
    return getattr(importlib.import_module(module, package=None), cls)


def instantiate_from_config(config):
    if "target" not in config:
        raise Exception("target not in config! ", config)
    return get_obj_from_str(config["target"])(**config.get("params", dict()))


def flatten_dict(d, parent_key="", sep="."):
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def flatten_and_filter_config(config):
    flat_config = flatten_dict(config)
    filtered_config = {}
    for key, value in flat_config.items():
        if isinstance(value, (int, float, str, bool, torch.Tensor)):
            filtered_config[key] = value
        else:
            filtered_config[key] = str(value)  # Convert unsupported types to string
    return filtered_config


def convert_to_rgb(images, min_val=None, max_val=None):
    # Get the colormap
    cmap = plt.get_cmap("jet")

    # Ensure images are detached and converted to numpy for colormap application
    images_np = images.squeeze(1).detach().cpu().numpy()  # shape: (B, H, W)

    converted_images = []
    for img in images_np:
        # Normalize img to range [0, 1]
        tmp_min = img.min() if min_val is None else min_val
        tmp_max = img.max() if max_val is None else max_val
        img_normalized = (img - tmp_min) / (tmp_max - tmp_min + 1e-5)

        # Apply colormap and convert to RGB
        img_rgb = cmap(img_normalized)

        # Convert from RGBA (4 channels) to RGB (3 channels)
        img_rgb = img_rgb[..., :3]  # shape: (H, W, 3)

        # Convert to PyTorch tensor and scale to range [0, 255]
        img_rgb_tensor = torch.tensor(img_rgb * 255, dtype=torch.uint8).permute(
            2, 0, 1
        )  # shape: (3, H, W)
        img_rgb_tensor = torch.clamp(img_rgb_tensor, 0, 255)

        converted_images.append(img_rgb_tensor)

    return converted_images


def plot_recreated_vs_original(
    recreated_ds: xr.Dataset,
    original_ds: xr.Dataset,
    level: int = None,
    cmap: str = "jet",
):
    """
    Compare variables from the recreated xarray dataset with the original dataset using Cartopy,
    ensuring that both plots share the same color bar using the original dataset's range as reference.

    Parameters:
    - recreated_ds (xr.Dataset): The dataset that was recreated from tensor.
    - original_ds (xr.Dataset): The original dataset with possibly more variables.
    - level (int, optional): If provided, the function will select this level for comparison.

    This function plots variables that exist in both datasets. For variables with a 'level' dimension,
    the plot uses the middle level for comparison.
    """

    # Find common variables between recreated and original datasets
    common_vars = [
        var for var in recreated_ds.data_vars if var in original_ds.data_vars
    ]

    for var in common_vars:

        # Extract the variable from both datasets
        recreated_var = recreated_ds[var]
        original_var = original_ds[var]

        # If the variable has a 'level' dimension, select the middle level for plotting
        if "level" in recreated_var.dims:
            if level is None:
                selected_level = recreated_var.level[len(recreated_var.level) // 2]
            else:
                selected_level = level
            recreated_var = recreated_var.sel(level=selected_level)
            original_var = original_var.sel(level=selected_level)
            print(f"Plotting variable: {var} at level {selected_level}")
        else:
            print(f"Plotting variable: {var}")

        # Select the first time step for plotting
        recreated_var = recreated_var.isel(time=0)
        original_var = original_var.isel(time=0)

        # Check if lat and lon dimensions need alignment (transpose if necessary)
        # if not (recreated_var.latitude.equals(original_var.latitude) and recreated_var.longitude.equals(original_var.longitude)):
        original_var = original_var.transpose("latitude", "longitude")

        # Determine the min and max values from the original data for consistent color scale
        vmin = original_var.min().item()
        vmax = original_var.max().item()

        # Set up the plot with Cartopy for geographic visualization
        fig, axes = plt.subplots(
            1,
            2,
            figsize=(14, 6),
            subplot_kw={"projection": ccrs.PlateCarree()},
            dpi=300,
        )
        ax1, ax2 = axes

        # Add coastlines and features with transparency (alpha)
        ax1.coastlines(alpha=0.3)  # Adjust transparency with alpha
        ax2.coastlines(alpha=0.3)
        ax1.add_feature(
            cfeature.BORDERS, linestyle=":", alpha=0.3
        )  # Borders with transparency
        ax2.add_feature(cfeature.BORDERS, linestyle=":", alpha=0.3)

        im1 = original_var.plot(
            ax=ax1,
            transform=ccrs.PlateCarree(),
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,  # Use original data's color scale
            add_colorbar=False,
        )
        if level is not None:
            ax1.set_title(f"Original: {var} at level {level}")
        else:
            ax1.set_title(f"Original: {var}")

        im2 = recreated_var.plot(
            ax=ax2,
            transform=ccrs.PlateCarree(),
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,  # Use original data's color scale
            add_colorbar=False,
        )
        if level is not None:
            ax2.set_title(f"Recreated: {var} at level {level}")
        else:
            ax2.set_title(f"Recreated: {var}")

        # Create a custom axes for the colorbar outside the main plot area
        cbar_ax = fig.add_axes([0.92, 0.25, 0.02, 0.5])  # [left, bottom, width, height]
        cbar = fig.colorbar(im2, cax=cbar_ax)  # Use the custom colorbar axis
        cbar.set_label(f"{var} (shared scale)")

        # Adjust layout for the plots and colorbar
        plt.subplots_adjust(wspace=0.05)  # Increase the space between the subplots

        # Show the plot
        plt.show()


def plot_traj_animation(
    samples,
    y_true,
    lon,
    lat,
    title=None,
    var_name=None,
    cb=True,
    save=False,
    err_metric=None,
    alpha=0.5,
    plot_residual=True,
):
    """
    Create an animation comparing predicted samples to true values over time for single-channel geospatial data using Cartopy.

    Parameters:
    - samples (numpy.ndarray or torch.Tensor): Predicted data of shape (T, H, W).
    - y_true (numpy.ndarray or torch.Tensor): Ground truth data of shape (T, H, W).
    - lon (1D array-like): Longitude coordinates of length W.
    - lat (1D array-like): Latitude coordinates of length H.
    - title (str, optional): Title for the animation and saved file.
    - var_name (str, optional): Name of the variable being plotted (for titles and labels).
    - cb (bool, optional): Whether to display colorbars. Default is True.
    - save (bool, optional): Whether to save the animation as a GIF. Default is False.
    - err_metric (tuple of numpy.ndarray, optional): Tuple containing error metrics (RMSE, nRMSE, CSV) each of shape (T,).
    - alpha (float, optional): Transparency level for Cartopy overlays. Default is 0.5.
    - plot_residual (bool, optional): Whether to plot the L1 residual panel. Default is True.

    Example:
    pred = pred.transpose('prediction_timedelta', 'latitude', 'longitude')
    true = true.transpose('time', 'latitude', 'longitude')

    plot_traj_animation(pred.values, true.values, var_name="Specific Humidity-500hPa",
                        save=True, title='tmp/specific_humidity_500',
                        lat=true['latitude'].values, lon=true['longitude'].values)
    """

    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    import matplotlib.pyplot as plt
    import numpy as np
    import torch

    # Convert torch tensors to numpy arrays if necessary
    if isinstance(samples, torch.Tensor):
        samples = samples.detach().cpu().numpy()
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()

    # Convert longitude from 0-360 to -180-180 format if necessary
    lon = np.array(lon)
    lon_180 = np.where(lon > 180, lon - 360, lon)

    # Sort longitude and rearrange data accordingly
    sort_idx = np.argsort(lon_180)
    lon_180 = lon_180[sort_idx]
    samples = samples[..., sort_idx]
    y_true = y_true[..., sort_idx]

    # Validate input shapes
    if samples.shape != y_true.shape:
        raise ValueError(
            f"Shape mismatch: samples shape {samples.shape} != y_true shape {y_true.shape}"
        )

    T, H, W = samples.shape

    # Initialize error metrics if provided
    if err_metric is not None:
        if len(err_metric) != 3:
            raise ValueError(
                "err_metric must be a tuple of three numpy arrays: (RMSE, nRMSE, CSV)"
            )
        err_RMSE, err_nRMSE, err_CSV = err_metric
        if not (
            err_RMSE.shape[0] == T and err_nRMSE.shape[0] == T and err_CSV.shape[0] == T
        ):
            raise ValueError(
                "Each error metric array must have length equal to the number of time steps (T)"
            )

    # Compute L1 Residuals (always computed if plot_residual is True)
    residual = np.abs(y_true - samples)

    # Determine color scales based on the combined range of samples and y_true
    if cb:
        vmin_true_pred = min(samples.min(), y_true.min())
        vmax_true_pred = max(samples.max(), y_true.max())
        if plot_residual:
            vmin_residual = residual.min()
            vmax_residual = residual.max()
    else:
        vmin_true_pred = vmax_true_pred = None
        if plot_residual:
            vmin_residual = vmax_residual = None

    # Define the Cartopy projection
    projection = ccrs.PlateCarree()

    # Set up the figure and axes with Cartopy projections
    if plot_residual:
        fig, axes = plt.subplots(
            1, 3, figsize=(30, 6), subplot_kw={"projection": projection}
        )
        ax_true, ax_pred, ax_residual = axes
    else:
        fig, axes = plt.subplots(
            1, 2, figsize=(20, 6), subplot_kw={"projection": projection}
        )
        ax_true, ax_pred = axes

    # Configure the axes with Cartopy overlays
    all_axes = axes if not plot_residual else [ax_true, ax_pred, ax_residual]
    for ax in all_axes:
        # Add coastlines and features
        ax.coastlines(resolution="50m", linewidth=1, alpha=alpha)
        ax.add_feature(cfeature.BORDERS, linestyle=":", alpha=alpha * 0.6)
        ax.add_feature(cfeature.LAND, facecolor=(0.8, 0.8, 0.8, alpha))
        ax.add_feature(cfeature.OCEAN, facecolor=(1.0, 1.0, 1.0, alpha * 0.6))

        # Add gridlines
        gl = ax.gridlines(
            crs=projection,
            draw_labels=True,
            linewidth=0.5,
            color="gray",
            alpha=0.5,
            linestyle="--",
        )
        gl.top_labels = False
        gl.right_labels = False
        gl.xlines = True
        gl.ylines = True
        gl.xlabel_style = {"size": 8}
        gl.ylabel_style = {"size": 8}

        # Set extent based on input coordinates with a small buffer
        buffer = 1  # degrees
        ax.set_extent(
            [
                lon_180.min() - buffer,
                lon_180.max() + buffer,
                lat.min() - buffer,
                lat.max() + buffer,
            ],
            crs=projection,
        )

    # Create meshgrid for pcolormesh
    Lon, Lat = np.meshgrid(lon, lat)

    # Initialize the plots for True and Predicted
    im_true = ax_true.pcolormesh(
        Lon,
        Lat,
        y_true[0],
        cmap="jet",
        vmin=vmin_true_pred,
        vmax=vmax_true_pred,
        transform=projection,
        shading="auto",
    )
    im_pred = ax_pred.pcolormesh(
        Lon,
        Lat,
        samples[0],
        cmap="jet",
        vmin=vmin_true_pred,
        vmax=vmax_true_pred,
        transform=projection,
        shading="auto",
    )

    # Initialize residual plot if requested
    if plot_residual:
        im_residual = ax_residual.pcolormesh(
            Lon,
            Lat,
            residual[0],
            cmap="viridis",
            vmin=vmin_residual,
            vmax=vmax_residual,
            transform=projection,
            shading="auto",
        )

    # Set initial titles
    if var_name:
        ax_true.set_title(f"True {var_name}", fontsize=16)
        ax_pred.set_title(f"Predicted {var_name}", fontsize=16)
        if plot_residual:
            ax_residual.set_title("L1 Residuals", fontsize=16)
    else:
        ax_true.set_title("True", fontsize=16)
        ax_pred.set_title("Predicted", fontsize=16)
        if plot_residual:
            ax_residual.set_title("L1 Residuals", fontsize=16)

    # Add error metrics text if provided (only for True and Predicted)
    if err_metric is not None:
        text_true = ax_true.text(
            0.5,
            -0.1,
            "",
            transform=ax_true.transAxes,
            fontsize=12,
            va="center",
            ha="center",
            bbox=dict(facecolor="white", alpha=0.7, boxstyle="round"),
        )
        text_pred = ax_pred.text(
            0.5,
            -0.1,
            "",
            transform=ax_pred.transAxes,
            fontsize=12,
            va="center",
            ha="center",
            bbox=dict(facecolor="white", alpha=0.7, boxstyle="round"),
        )
        # Initialize with first frame's metrics
        text_true.set_text(
            f"RMSE: {err_RMSE[0]:.4f}\nnRMSE: {err_nRMSE[0]:.4f}\nCSV: {err_CSV[0]:.4f}"
        )
        text_pred.set_text(
            f"RMSE: {err_RMSE[0]:.4f}\nnRMSE: {err_nRMSE[0]:.4f}\nCSV: {err_CSV[0]:.4f}"
        )
        if plot_residual:
            text_residual = ax_residual.text(
                0.5,
                -0.1,
                "",
                transform=ax_residual.transAxes,
                fontsize=12,
                va="center",
                ha="center",
                bbox=dict(facecolor="white", alpha=0.7, boxstyle="round"),
            )
            text_residual.set_text("")  # Can be used for additional metrics if needed
    else:
        text_true = text_pred = text_residual = None

    # Create colorbars if required
    if cb:
        # Position the colorbar for True and Predicted
        cbar_ax_true_pred = fig.add_axes(
            [0.94, 0.15, 0.01, 0.7]
        )  # [left, bottom, width, height]
        cbar_true_pred = fig.colorbar(im_true, cax=cbar_ax_true_pred)
        if var_name:
            cbar_true_pred.set_label(
                f"{var_name}", rotation=270, labelpad=15, fontsize=12
            )
        else:
            cbar_true_pred.set_label("Value", rotation=270, labelpad=15, fontsize=12)

        # Position the colorbar for Residuals if plotted
        if plot_residual:
            cbar_ax_residual = fig.add_axes(
                [0.97, 0.15, 0.01, 0.7]
            )  # [left, bottom, width, height]
            cbar_residual = fig.colorbar(
                im_residual, cax=cbar_ax_residual, cmap="viridis"
            )
            cbar_residual.set_label(
                "L1 Residuals", rotation=270, labelpad=15, fontsize=12
            )

    # Adjust layout to prevent overlapping and ensure colorbars fit nicely
    if plot_residual:
        plt.tight_layout(rect=[0, 0, 0.935, 1])
    else:
        plt.tight_layout(rect=[0, 0, 0.945, 1])

    # Update function for animation
    def update(frame):
        im_true.set_array(y_true[frame].flatten())
        im_pred.set_array(samples[frame].flatten())
        if plot_residual:
            im_residual.set_array(residual[frame].flatten())

        time_label = f"+{(frame) * 6}hr"
        if var_name:
            ax_true.set_title(f"True {var_name} {time_label}", fontsize=16)
            ax_pred.set_title(f"Predicted {var_name} {time_label}", fontsize=16)
            if plot_residual:
                ax_residual.set_title(f"L1 Residuals {time_label}", fontsize=16)
        else:
            ax_true.set_title(f"True {time_label}", fontsize=16)
            ax_pred.set_title(f"Predicted {time_label}", fontsize=16)
            if plot_residual:
                ax_residual.set_title(f"L1 Residuals {time_label}", fontsize=16)

        if err_metric is not None:
            text_true.set_text(
                f"RMSE: {err_RMSE[frame]:.4f}\nnRMSE: {err_nRMSE[frame]:.4f}\nCSV: {err_CSV[frame]:.4f}"
            )
            text_pred.set_text(
                f"RMSE: {err_RMSE[frame]:.4f}\nnRMSE: {err_nRMSE[frame]:.4f}\nCSV: {err_CSV[frame]:.4f}"
            )
            if plot_residual:
                # Update additional metrics for residual if needed
                text_residual.set_text("")

        # Return updated artists; include residual if plotted
        artists = [im_true, im_pred]
        if plot_residual:
            artists.append(im_residual)
        if err_metric is not None:
            artists.extend([text_true, text_pred])
            if plot_residual:
                artists.append(text_residual)
        return artists

    # Create the animation
    anim = FuncAnimation(fig, update, frames=T, blit=False, repeat=True, interval=500)

    # Save the animation if required
    if save and title:
        writergif = PillowWriter(fps=2)
        anim.save(f"{title}.gif", writer=writergif)
        print(f"Animation saved as {title}.gif")

    plt.show()


def plot_single_traj_animation(
    data, lon, lat, title=None, var_name=None, cb=True, save=False, alpha=0.5
):
    """
    Create an animation of single-channel geospatial data over time using Cartopy.

    Parameters:
    - data (numpy.ndarray or torch.Tensor): Data of shape (T, H, W).
    - lon (1D array-like): Longitude coordinates of length W.
    - lat (1D array-like): Latitude coordinates of length H.
    - title (str, optional): Title for the animation and saved file.
    - var_name (str, optional): Name of the variable being plotted (for titles and labels).
    - cb (bool, optional): Whether to display colorbar. Default is True.
    - save (bool, optional): Whether to save the animation as a GIF. Default is False.
    - alpha (float, optional): Transparency level for Cartopy overlays. Default is 0.5.

    example:
    data = data.transpose('time', 'latitude', 'longitude')

    plot_single_traj_animation(data.values, var_name="Specific Humidity-500hPa",
                         save=True, title='tmp/specific_humidity_500',
                         lat=data['latitude'].values, lon=data['longitude'].values)
    """
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    import matplotlib.pyplot as plt
    import numpy as np
    import torch
    from matplotlib.animation import PillowWriter

    # Convert torch tensors to numpy arrays if necessary
    if isinstance(data, torch.Tensor):
        data = data.detach().cpu().numpy()

    # Convert longitude from 0-360 to -180-180 format if necessary
    lon = np.array(lon)
    lon_180 = np.where(lon > 180, lon - 360, lon)

    # Sort longitude and rearrange data accordingly
    sort_idx = np.argsort(lon_180)
    lon_180 = lon_180[sort_idx]
    data = data[..., sort_idx]

    # Get the shape of the data
    T, H, W = data.shape

    # Determine color scale based on the entire range of data across all time steps
    if cb:
        vmin = data.min()
        vmax = data.max()
    else:
        vmin = vmax = None

    # Define the Cartopy projection
    projection = ccrs.PlateCarree()

    # Set up the figure and axis with Cartopy projection
    fig, ax = plt.subplots(figsize=(12, 8), subplot_kw={"projection": projection})

    # Configure the axis with features
    # Add coastlines and features
    ax.coastlines(resolution="50m", linewidth=1, alpha=alpha)
    ax.add_feature(cfeature.BORDERS, linestyle=":", alpha=alpha * 0.6)
    ax.add_feature(cfeature.LAND, facecolor=(0.8, 0.8, 0.8, alpha))
    ax.add_feature(cfeature.OCEAN, facecolor=(1.0, 1.0, 1.0, alpha * 0.6))

    # Add gridlines
    gl = ax.gridlines(
        crs=projection,
        draw_labels=True,
        linewidth=0.5,
        color="gray",
        alpha=0.5,
        linestyle="--",
    )
    gl.top_labels = False
    gl.right_labels = False
    gl.xlines = True
    gl.ylines = True
    gl.xlabel_style = {"size": 8}
    gl.ylabel_style = {"size": 8}

    # Set extent based on input coordinates with a small buffer
    buffer = 1  # degrees
    ax.set_extent(
        [
            lon_180.min() - buffer,
            lon_180.max() + buffer,
            lat.min() - buffer,
            lat.max() + buffer,
        ],
        crs=projection,
    )

    # Create meshgrid for pcolormesh
    Lon, Lat = np.meshgrid(lon_180, lat)

    # Initialize the plot
    im = ax.pcolormesh(
        Lon,
        Lat,
        data[0],
        cmap="jet",
        vmin=vmin,
        vmax=vmax,
        transform=projection,
        shading="auto",
    )

    # Set initial title
    time_label = "+0hr"
    display_title = f"{var_name} {time_label}" if var_name else f"Data {time_label}"
    ax.set_title(display_title, fontsize=16)

    # Create colorbar if required
    if cb:
        cbar = fig.colorbar(
            im, ax=ax, orientation="horizontal", pad=0.05, fraction=0.05
        )
        if var_name:
            cbar.set_label(f"{var_name}", fontsize=12)
        else:
            cbar.set_label("Value", fontsize=12)

    plt.tight_layout()

    # Update function for animation
    def update(frame):
        # Update data in pcolormesh
        im.set_array(data[frame].flatten())

        # Update title with current time step (assuming 6-hour intervals as in the original)
        time_label = f"+{(frame) * 6}hr"
        display_title = f"{var_name} {time_label}" if var_name else f"Data {time_label}"
        ax.set_title(display_title, fontsize=16)

        return [im]

    # Create the animation
    anim = FuncAnimation(fig, update, frames=T, blit=False, repeat=True, interval=500)

    # Save the animation if required
    if save and title:
        writergif = PillowWriter(fps=2)
        anim.save(f"{title}.gif", writer=writergif)
        print(f"Animation saved as {title}.gif")

    plt.show()

    return anim


def plot_traj_static(
    samples,
    y_true,
    lon,
    lat,
    var_name=None,
    cb=True,
    save=False,
    err_metric=None,
    alpha=0.5,
    time_step=0,
    unit=None,
):
    """
    Create a static plot comparing predicted samples to true values for single-channel geospatial data using Cartopy.
    Handles both -180 to 180 and 0 to 360 longitude formats.

    Parameters:
    - samples (numpy.ndarray or torch.Tensor): Predicted data of shape (T, H, W).
    - y_true (numpy.ndarray or torch.Tensor): Ground truth data of shape (T, H, W).
    - lon (1D array-like): Longitude coordinates of length W (can be in 0-360 or -180-180 format).
    - lat (1D array-like): Latitude coordinates of length H.
    - var_name (str, optional): Name of the variable being plotted (for titles and labels).
    - cb (bool, optional): Whether to display colorbars. Default is True.
    - save (bool, str, optional): Whether to save the plot as an image. If a string is provided, it will be used as the filename.
    - err_metric (tuple of numpy.ndarray, optional): Tuple containing error metrics (RMSE, nRMSE, CSV) each of shape (T,).
    - alpha (float, optional): Transparency level for Cartopy overlays. Default is 0.5.
    - time_step (int, optional): The specific time step to plot. Default is 0.
    """

    # Convert torch tensors to numpy arrays if necessary
    if isinstance(samples, torch.Tensor):
        samples = samples.detach().cpu().numpy()
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()

    # Convert longitude from 0-360 to -180-180 format if necessary
    lon = np.array(lon)
    lon_180 = np.where(lon > 180, lon - 360, lon)

    # Sort longitude and rearrange data accordingly
    sort_idx = np.argsort(lon_180)
    lon_180 = lon_180[sort_idx]
    samples = samples[..., sort_idx]
    y_true = y_true[..., sort_idx]

    # Validate input shapes
    if samples.shape != y_true.shape:
        raise ValueError(
            f"Shape mismatch: samples shape {samples.shape} != y_true shape {y_true.shape}"
        )

    T, H, W = samples.shape

    if not (0 <= time_step < T):
        raise ValueError(
            f"time_step {time_step} is out of bounds for data with {T} time steps."
        )

    # Initialize error metrics if provided
    if err_metric is not None:
        if len(err_metric) != 3:
            raise ValueError(
                "err_metric must be a tuple of three numpy arrays: (RMSE, nRMSE, CSV)"
            )
        err_RMSE, err_nRMSE, err_CSV = err_metric
        if not (
            err_RMSE.shape[0] == T and err_nRMSE.shape[0] == T and err_CSV.shape[0] == T
        ):
            raise ValueError(
                "Each error metric array must have length equal to the number of time steps (T)"
            )

    # Compute L1 Residuals for the specified time step
    residual = np.abs(y_true[time_step] - samples[time_step])

    # Determine color scales based on the combined range of samples and y_true for the specified time step
    if cb:
        vmin_true_pred = min(samples[time_step].min(), y_true[time_step].min())
        vmax_true_pred = max(samples[time_step].max(), y_true[time_step].max())
        vmin_residual = residual.min()
        vmax_residual = residual.max()
    else:
        vmin_true_pred = vmax_true_pred = vmin_residual = vmax_residual = None

    # Define the Cartopy projection
    projection = ccrs.PlateCarree(central_longitude=0)

    # Set up the figure and axes with Cartopy projections
    fig, axes = plt.subplots(
        1, 3, figsize=(24, 6), subplot_kw={"projection": projection}
    )

    ax_true, ax_pred, ax_residual = axes

    # Add features to each subplot
    for ax in axes:
        # Add coastlines and features
        ax.coastlines(resolution="50m", linewidth=1, alpha=alpha)
        ax.add_feature(cfeature.BORDERS, linestyle=":", alpha=alpha * 0.6)
        ax.add_feature(cfeature.LAND, facecolor=(0.8, 0.8, 0.8, alpha))
        ax.add_feature(cfeature.OCEAN, facecolor=(1.0, 1.0, 1.0, alpha * 0.6))

        # Add gridlines
        gl = ax.gridlines(
            crs=projection,
            draw_labels=True,
            linewidth=0.5,
            color="gray",
            alpha=0.5,
            linestyle="--",
        )
        gl.top_labels = False
        gl.right_labels = False
        gl.xlines = True
        gl.ylines = True
        gl.xlabel_style = {"size": 8}
        gl.ylabel_style = {"size": 8}

        # Set extent based on input coordinates with a small buffer
        buffer = 1  # degrees
        ax.set_extent(
            [
                lon_180.min() - buffer,
                lon_180.max() + buffer,
                lat.min() - buffer,
                lat.max() + buffer,
            ],
            crs=projection,
        )

    # Create meshgrid for pcolormesh
    Lon, Lat = np.meshgrid(lon, lat)

    # Plot True Values
    im_true = ax_true.pcolormesh(
        Lon,
        Lat,
        y_true[time_step],
        cmap="jet",
        vmin=vmin_true_pred,
        vmax=vmax_true_pred,
        transform=projection,
        shading="auto",
    )
    # Plot Predicted Values
    im_pred = ax_pred.pcolormesh(
        Lon,
        Lat,
        samples[time_step],
        cmap="jet",
        vmin=vmin_true_pred,
        vmax=vmax_true_pred,
        transform=projection,
        shading="auto",
    )
    # Plot Residuals
    im_residual = ax_residual.pcolormesh(
        Lon,
        Lat,
        residual,
        cmap="viridis",
        vmin=vmin_residual,
        vmax=vmax_residual,
        transform=projection,
        shading="auto",
    )

    # Set titles
    time_label = f"Lead time: {int(time_step * 6)} hours"
    if var_name:
        ax_true.set_title(f"True {var_name}\n{time_label}", fontsize=14)
        ax_pred.set_title(f"Predicted {var_name}\n{time_label}", fontsize=14)
        ax_residual.set_title(f"L1 Residuals\n{time_label}", fontsize=14)
    else:
        ax_true.set_title(f"True\n{time_label}", fontsize=14)
        ax_pred.set_title(f"Predicted\n{time_label}", fontsize=14)
        ax_residual.set_title(f"L1 Residuals\n{time_label}", fontsize=14)

    # Add error metrics text if provided (only for True and Predicted)
    if err_metric is not None:
        # Position the text at the bottom center of each subplot with a semi-transparent background
        text_true = ax_true.text(
            0.5,
            -0.15,
            "",
            transform=ax_true.transAxes,
            fontsize=10,
            va="center",
            ha="center",
            bbox=dict(facecolor="white", alpha=0.7, boxstyle="round"),
        )
        text_pred = ax_pred.text(
            0.5,
            -0.15,
            "",
            transform=ax_pred.transAxes,
            fontsize=10,
            va="center",
            ha="center",
            bbox=dict(facecolor="white", alpha=0.7, boxstyle="round"),
        )
        # Initialize with current frame's metrics
        text_true.set_text(
            f"RMSE: {err_RMSE[time_step]:.4f}\nnRMSE: {err_nRMSE[time_step]:.4f}\nCSV: {err_CSV[time_step]:.4f}"
        )
        text_pred.set_text(
            f"RMSE: {err_RMSE[time_step]:.4f}\nnRMSE: {err_nRMSE[time_step]:.4f}\nCSV: {err_CSV[time_step]:.4f}"
        )

    # Create colorbars if required
    if cb:
        # Adjust layout to make space for colorbars
        plt.tight_layout(rect=[0, 0, 0.93, 1])

        # Colorbar for True and Predicted
        cbar_ax_true_pred = fig.add_axes(
            [0.935, 0.15, 0.01, 0.7]
        )  # [left, bottom, width, height]
        cbar_true_pred = fig.colorbar(im_true, cax=cbar_ax_true_pred)
        if unit:
            cbar_true_pred.set_label(str(unit), rotation=270, labelpad=15, fontsize=12)
        else:
            cbar_true_pred.set_label("Value", rotation=270, labelpad=15, fontsize=12)

        # Colorbar for Residuals
        cbar_ax_residual = fig.add_axes(
            [0.97, 0.15, 0.01, 0.7]
        )  # [left, bottom, width, height]
        cbar_residual = fig.colorbar(im_residual, cax=cbar_ax_residual, cmap="viridis")
        cbar_residual.set_label("L1 Residuals", rotation=270, labelpad=15, fontsize=12)

    # Save the plot if required
    if isinstance(save, str):
        plt.savefig(save, dpi=300)
        print(f"Plot saved as {save}")
    elif save:
        plt.savefig(f"{var_name}_time_{time_step}.png", dpi=300)
    plt.show()
