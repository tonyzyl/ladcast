import argparse
import json
import os
import time
import warnings
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
from functools import wraps
from typing import Optional

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import xarray as xr

from ladcast.dataloader.utils import precompute_mean_std
from ladcast.models.DCAE import AutoencoderDC
from ladcast.pipelines.utils import latent_ens_to_xarr

# Constants
GRID_RES = 1.5  # ERA5 grid resolution in degrees
NEIGHBOR_DEG = 1.5  # half-width of the local-min search box

VARIABLE_NAMES = [
    "geopotential",
    "specific_humidity",
    "temperature",
    "u_component_of_wind",
    "v_component_of_wind",
    "vertical_velocity",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "2m_temperature",
    "mean_sea_level_pressure",
    "sea_surface_temperature",
    "total_precipitation_6hr",
]


# Timing decorator
def timed(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        t0 = time.perf_counter()
        out = func(*args, **kwargs)
        print(f"[TIMING] {func.__name__!r} took {time.perf_counter() - t0:.3f} s")
        return out

    return wrapper


def load_ensemble_members(csv_path="ensemble_members.csv"):
    """
    Load ensemble-member tracks from CSV and return a dict:
      { member_name: [(time, lat, lon), …], … }
    """
    # Read CSV, parse the time column into Timestamps
    df = pd.read_csv(csv_path, parse_dates=["time"])

    ens_tracks = {}
    # Group by member and rebuild each track in step order
    for member, grp in df.groupby("member"):
        grp = grp.sort_values("step")
        # make list of (Timestamp, float, float)
        track = list(zip(grp["time"], grp["lat"], grp["lon"]))
        ens_tracks[member] = track
    return ens_tracks


def load_ensemble_mean(csv_path="ensemble_mean.csv"):
    """
    Load the ensemble-mean track from CSV and return a list:
      [(time, lat, lon), …]
    """
    df = pd.read_csv(csv_path, parse_dates=["time"])
    df = df.sort_values("step")
    return list(zip(df["time"], df["lat"], df["lon"]))


@timed
def load_hurdat(hurdat_file, storm_id):
    """Load HURDAT text and return DataFrame of time, lat, lon for one storm."""
    records = []
    with open(hurdat_file) as f:
        lines = f.readlines()
    i = 0
    while i < len(lines):
        header = [h.strip() for h in lines[i].split(",")]
        sid, n = header[0], int(header[2])
        if sid == storm_id:
            for j in range(i + 1, i + 1 + n):
                p = [x.strip() for x in lines[j].split(",")]
                dt = datetime.strptime(p[0] + p[1], "%Y%m%d%H%M")
                la = float(p[4][:-1]) * (-1 if p[4].endswith("S") else 1)
                lo = float(p[5][:-1]) * (-1 if p[5].endswith("W") else 1)
                if lo < 0:
                    lo += 360
                records.append({"time": dt, "lat": la, "lon": lo})
            break
        i += 1 + n
    if not records:
        raise ValueError(f"Storm {storm_id!r} not found in {hurdat_file}")
    return pd.DataFrame(records)


#@timed
def load_ibtracs(storm_id, ibtracs_file=None):
    """
    Load IBTrACS CSV and return DataFrame of time, lat, lon for one storm.

    Parameters:
    -----------
    ibtracs_file : str, optional
        Path to local IBTrACS CSV file or None to download from NOAA.
    storm_id : str
        Storm ID to extract from the IBTrACS dataset.
    """
    if ibtracs_file is None:
        # Use the direct URL if no file provided
        ibtracs_url = "https://www.ncei.noaa.gov/data/international-best-track-archive-for-climate-stewardship-ibtracs/v04r01/access/csv/ibtracs.ALL.list.v04r01.csv"
        print(f"Reading IBTrACS data directly from {ibtracs_url}")
        ibtracs_file = ibtracs_url

    df = pd.read_csv(
        ibtracs_file,
        skiprows=[1],  # drop the units-row
        parse_dates=["ISO_TIME"],
        na_values=["", " "],
        dtype={"SID": str},
    )

    storm = df[df.SID == storm_id].copy()
    if storm.empty:
        raise ValueError(f"{storm_id!r} not found in IBTrACS file")

    # rename and cast
    storm = storm.rename(columns={"ISO_TIME": "time", "LAT": "lat", "LON": "lon"})
    storm["lat"] = storm["lat"].astype(float)
    storm["lon"] = storm["lon"].astype(float)

    # convert any negative longitudes (degrees_east) into 0–360°
    storm.loc[storm["lon"] < 0, "lon"] += 360

    return storm[["time", "lat", "lon"]].reset_index(drop=True)


#@timed
def round_to_grid(val, resolution=GRID_RES):
    return float(np.round(val / resolution) * resolution)


def select_box(da2d, lat_lo, lat_hi, lon_start, lon_end):
    """Return subset of da2d within lat/lon box (handles 0–360 wrap)."""
    mask_lat = (da2d.latitude >= min(lat_lo, lat_hi)) & (
        da2d.latitude <= max(lat_lo, lat_hi)
    )
    if lon_start <= lon_end:
        mask_lon = (da2d.longitude >= lon_start) & (da2d.longitude <= lon_end)
    else:
        mask_lon = (da2d.longitude >= lon_start) | (da2d.longitude <= lon_end)
    return da2d.where(mask_lat & mask_lon, drop=True)


#@timed
def find_local_minimum(
    ds, varname, tgt_time, center, inner_deg, init_time=None, verbose=False
):
    """Search a small box around center for the local minimum of varname at tgt_time."""
    lat0, lon0 = center
    outer = inner_deg + NEIGHBOR_DEG * 2
    half_o = outer / 2
    half_i = inner_deg / 2

    lat_lo, lat_hi = lat0 - half_o, lat0 + half_o
    lon_s, lon_e = (lon0 - half_o) % 360, (lon0 + half_o) % 360

    if verbose:
        print(f"\n    [DEBUG] `{varname}` @ {tgt_time}")
        print(
            f"      outer box {outer}° → lat [{lat_lo:.3f}→{lat_hi:.3f}], "
            f"lon [{lon_s:.3f}→{lon_e:.3f}]"
        )

    # select the 2D field
    if "prediction_timedelta" in ds.dims:
        # lead = int((tgt_time - init_time) / np.timedelta64(1, 'ns'))
        lead = tgt_time - init_time
        # print(f"init_time {init_time}, type {type(init_time)}")
        # print(f"lead {lead}, type {type(lead)}")
        da = ds[varname].sel(time=init_time, prediction_timedelta=lead).compute()
    else:
        da = ds[varname].sel(time=tgt_time).compute()

    sub = select_box(da, lat_lo, lat_hi, lon_s, lon_e)
    if sub.size == 0:
        return None

    raw = []
    for la in sub.latitude.values:
        for lo in sub.longitude.values:
            v = float(da.sel(latitude=la, longitude=lo, method="nearest"))
            neigh = select_box(
                da, la - half_i, la + half_i, (lo - half_i) % 360, (lo + half_i) % 360
            )
            if neigh.size and v == float(neigh.min()):
                raw.append((la, lo, v))

    # drop edge points
    finals = [
        (la, lo, v)
        for la, lo, v in raw
        if not (
            abs(la - lat_lo) < 1e-6
            or abs(la - lat_hi) < 1e-6
            or abs((lo - lon_s) % 360) < 1e-6
            or abs((lo - lon_e) % 360) < 1e-6
        )
    ]
    if not finals:
        return None

    # pick closest to center
    best = min(
        finals,
        key=lambda t: (t[0] - lat0) ** 2 + (((t[1] - lon0 + 180) % 360 - 180) ** 2),
    )
    return best


# @timed
def track_first_n_steps(
    t0,
    raw_lat0,
    raw_lon0,
    zarr_path=None,
    zarr_path_w_lsm=None,
    ds=None,
    ens_member=None,
    ens_dim="idx",
    n_steps=3,
    inner_box_sizes=[7, 4, 1],
    enforce_msl=True,
    verbose=False,
):
    """
    Starting from (t0, raw_lat0, raw_lon0), find n_steps of local minima every 6h.
    """
    lat0 = round_to_grid(raw_lat0)
    lon0 = round_to_grid(raw_lon0)

    if ds is None:
        ds = xr.open_zarr(zarr_path, chunks={"time": 1})
    if ens_member is not None:
        ds = ds.sel({ens_dim: ens_member})
    if not enforce_msl:
        if zarr_path_w_lsm:
            land_mask = xr.open_zarr(zarr_path_w_lsm)["land_sea_mask"].load()
        else:
            land_mask = ds["land_sea_mask"].load()

    track = [(t0, lat0, lon0)]
    current = (lat0, lon0)

    for step in range(1, n_steps + 1):
        prev = current
        t_next = t0 + timedelta(hours=6 * step)
        if enforce_msl:
            mval = 0
        else:
            mval = land_mask.sel(
                latitude=current[0], longitude=current[1], method="nearest"
            ).values

        moved = False
        if mval < 0.5:
            for inner in inner_box_sizes:
                res = find_local_minimum(
                    ds,
                    "mean_sea_level_pressure",
                    t_next,
                    current,
                    inner,
                    init_time=t0,
                    verbose=verbose,
                )
                if res and ((prev[0] != res[0]) or (prev[1] != res[1])):
                    current = (res[0], res[1])
                    moved = True
                    break
            if verbose:
                print(f"  [DEBUG] prev {prev} -> current {current}, moved: {moved}")

        if not moved:
            if not enforce_msl:
                ds700 = ds.sel(level=700)
                for inner in inner_box_sizes:
                    res = find_local_minimum(
                        ds700,
                        "geopotential",
                        t_next,
                        current,
                        inner,
                        init_time=t0,
                        verbose=verbose,
                    )
                    if res and ((prev[0] != res[0]) or (prev[1] != res[1])):
                        current = (res[0], res[1])
                        moved = True
                        break
                if verbose:
                    print(f"  [DEBUG] prev {prev} -> current {current}, moved: {moved}")
            else:
                warnings.warn(
                    f"Enforce msl but no local min found at {t_next}, not moving"
                )

        if not moved and not enforce_msl:
            warnings.warn(
                f"Tried geopotential but no local min found at {t_next}, not moving"
            )

        track.append((t_next, *current))

    if verbose:
        print(
            f"\n=== Final track ({'ens=' + str(ens_member) if ens_member is not None else 'ERA5'}) ==="
        )
        for t, la, lo in track:
            print(f"  {t} → lat={la:.3f}, lon={lo:.3f}")

    return track


# @timed
def load_kml_tracks(url, valid_models=None, n_steps=None, interval=1):
    """
    Fetch a remote KML and return a dict mapping
      model → [(hour, lon, lat), …]
    then:
      • drop hour > n_steps*6   (if n_steps is not None)
      • thin by [::interval]
    """
    resp = requests.get(url)
    resp.raise_for_status()
    root = ET.fromstring(resp.text)
    ns = {"kml": "http://www.opengis.net/kml/2.2"}

    # 1) parse all forecast‐hour points
    raw = {}
    for pm in root.findall(".//kml:Placemark", ns):
        nm_el = pm.find("kml:name", ns)
        co_el = pm.find(".//kml:Point/kml:coordinates", ns)
        if nm_el is None or co_el is None:
            continue

        parts = nm_el.text.split()
        model = parts[0]
        if valid_models and model not in valid_models:
            continue

        # forecast hour is last token
        try:
            hour = int(parts[-1])
        except ValueError:
            continue

        lon, lat, *_ = map(float, co_el.text.strip().split(","))
        raw.setdefault(model, []).append((hour, lon, lat))

    # 2) sort, cap at max_hr, thin by interval
    tracks = {}
    max_hr = n_steps * 6 if n_steps is not None else None
    for model, pts in raw.items():
        pts = sorted(pts, key=lambda x: x[0])
        print(f"[DEBUG] model {model}, Pts {pts}")
        if max_hr is not None:
            pts = [p for p in pts if p[0] <= max_hr]
        if interval > 1:
            pts = pts[::interval]
        if pts:
            tracks[model] = pts

    return tracks


def plot_tracks(
    hurdat_df,  # df_obs: 6h-spaced IBTrACS or HURDAT (may be None)
    interval=1,
    era5_track=None,
    ens_tracks=None,
    ens_mean_track=None,
    hur_model_tracks=None,
    title="Storm Track Comparison",
    extent=None,  # (lon_min, lon_max, lat_min, lat_max)
    ensemble_plot_name="Ensemble",
    save_path=Optional[str],  # e.g. "storm_track.png"
):
    """
    Plot observed (IBTrACS/HURDAT) + ERA5 + ensemble tracks,
    annotating each at 0,24,48,… hours since its own start.
    """
    # 1) determine time window
    if era5_track:
        t_start, t_end = era5_track[0][0], era5_track[-1][0]
    elif ens_tracks:
        first = next(iter(ens_tracks.values()))
        t_start, t_end = first[0][0], first[-1][0]
    elif hurdat_df is not None:
        t_start, t_end = hurdat_df.time.min(), hurdat_df.time.max()
    else:
        raise ValueError("No track data available to plot")

    # 2) downsample observed track
    if hurdat_df is not None:
        df_win = hurdat_df[
            (hurdat_df.time >= t_start) & (hurdat_df.time <= t_end)
        ].reset_index(drop=True)
        df_plot = df_win.iloc[::interval]

    fig = plt.figure(figsize=(10, 7))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.coastlines("50m")
    ax.add_feature(cfeature.BORDERS)
    ax.gridlines(draw_labels=True)

    if extent is not None:
        # extent = (lon_min, lon_max, lat_min, lat_max)
        ax.set_extent(extent, crs=ccrs.PlateCarree())

    # 4) plot observed + annotate every 24h
    if hurdat_df is not None:
        ax.plot(
            df_plot.lon,
            df_plot.lat,
            "o-",
            transform=ccrs.PlateCarree(),
            label="IBTrACS",
            linewidth=2,
        )
        t0_obs = df_plot.time.iloc[0]
        for t, la, lo in zip(df_plot.time, df_plot.lat, df_plot.lon):
            hours = int((t - t0_obs).total_seconds() // 3600)
            if hours % 24 == 0:
                ax.text(
                    lo,
                    la,
                    f"{hours}",
                    transform=ccrs.PlateCarree(),
                    fontsize=6,
                    fontweight="bold",
                    ha="left",
                    va="bottom",
                )

    # 5) plot ERA5 + annotate
    if era5_track:
        era5_sub = era5_track[::interval]
        lons = [p[2] for p in era5_sub]
        lats = [p[1] for p in era5_sub]
        ax.plot(
            lons,
            lats,
            "s--",
            transform=ccrs.PlateCarree(),
            label="ERA5 (1.5°)",
            linewidth=2,
        )
        # t0_era5 = era5_track[0][0]
        # for t, la, lo in era5_track:
        # hours = int((t - t0_era5).total_seconds() // 3600)
        # if hours % 24 == 0:
        # ax.text(
        # lo, la, f"{hours}",
        # transform=ccrs.PlateCarree(),
        # fontsize=8, color='blue',
        # ha='center', va='bottom'
        # )

    # 6) plot ensemble members + annotate
    if ens_tracks:
        for idx, (member, track) in enumerate(ens_tracks.items()):
            sub = track[::interval]
            lons = [p[2] for p in sub]
            lats = [p[1] for p in sub]
            (line,) = ax.plot(
                lons,
                lats,
                transform=ccrs.PlateCarree(),
                color="green",
                linewidth=1,
                alpha=0.4,
                label=ensemble_plot_name if idx == 0 else "_nolegend_",
            )
            line.set_linestyle((idx * 3, (6, 4)))
            # annotate
            # t0_ens = track[0][0]
            # for t, la, lo in track:
            # hours = int((t - t0_ens).total_seconds() // 3600)
            # if hours % 24 == 0:
            # ax.text(
            # lo, la, f"{hours}",
            # transform=ccrs.PlateCarree(),
            # fontsize=7, color='darkgreen',
            # ha='center', va='bottom'
            # )
            # final marker
            ax.plot(
                lons[-1],
                lats[-1],
                "o",
                transform=ccrs.PlateCarree(),
                color=line.get_color(),
                markersize=2,
            )

    # 7) plot ensemble mean + annotate
    if ens_mean_track:
        mean_sub = ens_mean_track[::interval]
        lons = [p[2] for p in mean_sub]
        lats = [p[1] for p in mean_sub]
        ax.plot(
            lons,
            lats,
            "--",
            transform=ccrs.PlateCarree(),
            color="red",
            linewidth=2,
            label="Ensemble mean",
        )
        t0_mean = ens_mean_track[0][0]
        for t, la, lo in ens_mean_track:
            hours = int((t - t0_mean).total_seconds() // 3600)
            if hours % 24 == 0:
                ax.text(
                    lo,
                    la,
                    f"{hours}",
                    transform=ccrs.PlateCarree(),
                    fontsize=6,
                    color="black",
                    ha="center",
                    va="bottom",
                )
        ax.plot(
            lons[-1],
            lats[-1],
            "X",
            transform=ccrs.PlateCarree(),
            color="red",
            markersize=6,
        )

    # 8) plot KML model tracks (6 h, limited to n_steps, thinned by plot_interval)
    if hur_model_tracks:
        for model, track in hur_model_tracks.items():
            print("[DEBUG] model", model, "track", track)
            # track is [(hour, lon, lat), …]
            lons = [lon for _, lon, _ in track]
            lats = [lat for _, _, lat in track]

            ax.plot(lons, lats, "-^", transform=ccrs.PlateCarree(), label=model)

            # annotate every 24 h
            for hour, lon, lat in track:
                if hour % 24 == 0:
                    ax.text(
                        lon,
                        lat,
                        str(hour),
                        transform=ccrs.PlateCarree(),
                        fontsize=6,
                        color="red",
                        ha="center",
                        va="bottom",
                    )

    ax.legend()
    plt.title(title)
    plt.tight_layout()
    plt.savefig("storm_track_comparison.png", dpi=400)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compare storm track from IBTrACS/HURDAT vs ERA5 (and optional ensemble)."
    )
    parser.add_argument("storm_id", help="Storm ID for IBTrACS")
    parser.add_argument("--atcf_id", type=str, default=None, help="(optional) ATCF ID")
    parser.add_argument(
        "--encdec_model",
        type=str,
        help="Path to encdec model folder on huggingface Hub",
    )
    parser.add_argument(
        "--hurdat_file",
        type=str,
        default=None,
        help="(optional) Path to HURDAT text file",
    )
    parser.add_argument(
        "--ibtracs_file",
        type=str,
        default=None,
        help="(optional) Path to IBTrACS CSV file",
    )
    parser.add_argument(
        "--zarr_path",
        type=str,
        default="gs://weatherbench2/datasets/era5/1959-2023_01_10-1h-240x121_equiangular_with_poles_conservative.zarr/",
        help="Path to ERA5 Zarr store",
    )
    parser.add_argument(
        "--n_steps", type=int, default=None, help="Number of 6-h steps to compute"
    )
    parser.add_argument(
        "--startdate", type=str, help="Override start at YYYYMMDDHH (nearest)"
    )
    parser.add_argument(
        "--plot_interval", type=int, default=1, help="Plot every Nth 6-h step"
    )
    parser.add_argument(
        "--ens_zarr",
        type=str,
        default="gs://weatherbench2/datasets/ifs_ens/2018-2022-240x121_equiangular_with_poles_conservative.zarr",
        help="Optional: ensemble Zarr store",
    )
    parser.add_argument(
        "--latent_path",
        type=str,
        help="Path to prediction latent space sequence npy file",
    )
    parser.add_argument(
        "--normalization_json", type=str, help="Path to normalization json file"
    )
    parser.add_argument(
        "--ens_dim", type=str, default="idx", help="Name of the ensemble dimension"
    )
    parser.add_argument(
        "--ens_member",
        type=str,
        default=None,
        help='Comma-sep list of member of the ensemble dim (e.g. "0,2,5")',
    )
    parser.add_argument(
        "--plot_ens_mean",
        action="store_true",
        help="Plot ensemble mean instead of members",
    )
    parser.add_argument(
        "--show_hur_models",
        action="store_true",
        help="Fetch models from UW-Madison archive, https://web.uwm.edu/hurricane-models/models/models.html",
    )
    parser.add_argument(
        "--plot_lat_range",
        type=str,
        default=None,
        help='Plot latitude range (e.g. "10,30")',
    )
    parser.add_argument(
        "--plot_lon_range",
        type=str,
        default=None,
        help='Plot longitude range (e.g. "0,360")',
    )
    parser.add_argument(
        "--use_saved",
        action="store_true",
        help="Use saved ensemble members and mean tracks csv",
    )
    args = parser.parse_args()

    # 1) load IBTrACS or HURDAT for start+obs
    df_src_raw = load_ibtracs(args.storm_id, args.ibtracs_file)

    # elif args.hurdat_file:
    # df_src_raw = load_hurdat(args.hurdat_file, args.atcf_id)
    # else:
    # parser.error("Must supply at least one of --ibtracs_file or --hurdat_file")

    if args.plot_lat_range is not None:
        assert args.plot_lon_range is not None, "Must supply both lat and lon ranges"
        # formulate plot extent to be (lon_min, lon_max, lat_min, lat_max)
        extent = (
            float(args.plot_lon_range.split(",")[0]),
            float(args.plot_lon_range.split(",")[1]),
            float(args.plot_lat_range.split(",")[0]),
            float(args.plot_lat_range.split(",")[1]),
        )
    else:
        extent = None

    # 2) pick t0 (and allow --startdate override)
    df_src_raw = df_src_raw.sort_values("time").reset_index(drop=True)
    if args.startdate:
        forced = datetime.strptime(args.startdate, "%Y%m%d%H")
        if forced in df_src_raw.time.values:
            row0 = df_src_raw[df_src_raw.time == forced].iloc[0]
        else:
            idx_near = (df_src_raw.time - forced).abs().idxmin()
            row0 = df_src_raw.loc[idx_near]
            print(f"[INFO] startdate not exact, using nearest at {row0.time}")
    else:
        row0 = df_src_raw.iloc[0]
    t0, raw_lat0, raw_lon0 = row0.time, row0.lat, row0.lon

    # 3) enforce strict 6 h spacing on observed, and slice to n_steps if given
    dt_secs = (df_src_raw.time - t0).dt.total_seconds()
    mask6 = (dt_secs >= 0) & (dt_secs % (6 * 3600) == 0)
    df_obs = df_src_raw[mask6].reset_index(drop=True)
    if args.n_steps is None:
        n_steps = len(df_obs) - 1
    else:
        n_steps = args.n_steps
    df_obs = df_obs.iloc[: n_steps + 1].reset_index(drop=True)
    print(df_obs)

    # 4) run ERA5 tracker if requested
    era5_track = None
    if args.zarr_path:
        era5_track = track_first_n_steps(
            t0,
            raw_lat0,
            raw_lon0,
            zarr_path=args.zarr_path,
            zarr_path_w_lsm=None,
            n_steps=args.n_steps,
        )

    # 5) run ensemble members if requested
    ens_tracks = {}
    if not args.use_saved:
        if args.ens_zarr:
            ds_ens = xr.open_zarr(args.ens_zarr, chunks={"time": 1})
            members = ds_ens[args.ens_dim].values
            to_run = (
                [int(m) for m in args.ens_member.split(",")]
                if args.ens_member
                else list(members)
            )
            for m in to_run:
                trk = track_first_n_steps(
                    t0,
                    raw_lat0,
                    raw_lon0,
                    zarr_path=args.ens_zarr,
                    zarr_path_w_lsm=args.zarr_path,
                    ens_member=m,
                    ens_dim=args.ens_dim,
                    n_steps=args.n_steps,
                    # enforce_msl=False
                )
                ens_tracks[f"M{m}"] = trk

        if args.latent_path:
            with open(args.normalization_json, "r") as f:
                normalization_param_dict = json.load(f)
            mean_tensor, std_tensor = precompute_mean_std(
                normalization_param_dict, VARIABLE_NAMES
            )
            repo_name = "tonyzyl/ladcast"
            encdec_model = AutoencoderDC.from_pretrained(
                repo_name,
                subfolder=args.encdec_model,
            )
            encdec_model.eval()
            encdec_model.to("cuda")
            encdec_model.requires_grad_(False)
            print(f"[INFO] loading latent space from {args.latent_path}")
            tmp_to_run = (
                [int(m) for m in args.ens_member.split(",")]
                if args.ens_member
                else None
            )
            ds_ens = latent_ens_to_xarr(
                args.latent_path,
                encdec_model=encdec_model,
                mean_tensor=mean_tensor,
                std_tensor=std_tensor,
                timestamp=args.startdate,
                variable_names=VARIABLE_NAMES,
                # extract_variables=["mean_sea_level_pressure"],
                extract_variables=["mean_sea_level_pressure", "geopotential"],
                extract_first=args.n_steps + 1,  # account for t0
                extract_ens_member_idx=tmp_to_run,
            )
            print(f"[INFO] loaded decoded latent space from {args.latent_path}")
            members = ds_ens[args.ens_dim].values
            to_run = (
                [int(m) for m in args.ens_member.split(",")]
                if args.ens_member
                else list(members)
            )
            for m in to_run:
                trk = track_first_n_steps(
                    t0,
                    raw_lat0,
                    raw_lon0,
                    ds=ds_ens,
                    zarr_path_w_lsm=args.zarr_path,
                    ens_member=m,
                    ens_dim=args.ens_dim,
                    n_steps=args.n_steps,
                    # enforce_msl=False
                )
                ens_tracks[f"M{m}"] = trk

        # 6) ensemble-mean if requested
        ens_mean_track = None
        if args.ens_zarr and args.plot_ens_mean:
            ds_ens = xr.open_zarr(args.ens_zarr, chunks={"time": 1})
            ds_mean = ds_ens.mean(dim=args.ens_dim)
            ens_mean_track = track_first_n_steps(
                t0,
                raw_lat0,
                raw_lon0,
                ds=ds_mean,
                zarr_path_w_lsm=args.zarr_path,
                n_steps=args.n_steps,
                # enforce_msl=False
            )
        if args.latent_path and args.plot_ens_mean:
            ds_mean = ds_ens.mean(dim=args.ens_dim)
            ens_mean_track = track_first_n_steps(
                t0,
                raw_lat0,
                raw_lon0,
                ds=ds_mean,
                zarr_path_w_lsm=args.zarr_path,
                n_steps=args.n_steps,
                # enforce_msl=False
            )
    else:
        if args.ens_zarr:
            ens_tracks = load_ensemble_members("ifsens_members.csv")
            ens_mean_track = load_ensemble_mean("ifsens_mean.csv")
        else:
            ens_tracks = load_ensemble_members("ladcast_members.csv")
            ens_mean_track = load_ensemble_mean("ladcast_mean.csv")

    kml_tracks = None
    hur_models = ["HWRF", "AEMN"]
    if args.show_hur_models:
        # directory and filename use lowercase storm_id, e.g. "al132019"
        sid_dir = args.atcf_id.lower()
        # year folder comes from the init‐time
        year = t0.year
        # timestamp formatted as YYYYMMDDHH (same as your startdate)
        dt_str = t0.strftime("%Y%m%d%H")
        # assemble exactly:
        kml_url = (
            f"https://web.uwm.edu/hurricane-models/models/"
            f"archive/{year}/{sid_dir}/"
            f"{sid_dir}_{dt_str}_late.kml"
        )
        print(f"[INFO] fetching hurricane tracks from {kml_url}")
        try:
            hur_model_tracks = load_kml_tracks(
                kml_url, hur_models, n_steps=args.n_steps, interval=args.plot_interval
            )
        except Exception as e:
            print(f"[WARNING] could not load models from KML: {e}")
            hur_model_tracks = None
    else:
        hur_model_tracks = None

    # 7) plot everything
    ensemble_plot_name = "Ensemble"
    if args.ens_zarr:
        ensemble_plot_name = "IFS-ENS (1.5°)"
    if args.latent_path:
        ensemble_plot_name = "LaDCast (1.5°)"
    plot_tracks(
        df_obs,
        interval=args.plot_interval,
        era5_track=era5_track,
        ens_tracks=ens_tracks or None,
        ens_mean_track=ens_mean_track,
        hur_model_tracks=hur_model_tracks,
        title="Start: "
        + str(t0)
        + ", Hurricane Lorenzo 7‑day forecast, "
        + ensemble_plot_name,
        extent=extent,
        ensemble_plot_name=ensemble_plot_name,
    )

    # 8) save the tracks to CSV
    if args.ens_zarr or args.latent_path:
        rows = []
        for member, track in ens_tracks.items():
            # track is e.g. [(t0, lat0, lon0), (t1, lat1, lon1), …]
            df = pd.DataFrame(track, columns=["time", "lat", "lon"])
            df["member"] = member
            df["step"] = df.index  # new "step" dimension, 0…n_steps−1
            rows.append(df)

        ens_df = pd.concat(rows, ignore_index=True)

        mean_df = pd.DataFrame(ens_mean_track, columns=["time", "lat", "lon"])
        mean_df["step"] = mean_df.index
        mean_df["member"] = "mean"

        if args.latent_path:
            ens_df.to_csv("ladcast_members.csv", index=False)
            mean_df.to_csv("ladcast_mean.csv", index=False)
        else:
            ens_df.to_csv("ifsens_members.csv", index=False)
            mean_df.to_csv("ifsens_mean.csv", index=False)


"""

# Typhoon Kong-rey, figsize (9, 11), inner_box_sizes=[7,5,1]

python track.py 2018271N06154 \
    --ibtracs_file=ibtracs_all_list_v04r01.csv \
    --startdate 2018093000 \
    --n_steps 28 \
    --plot_interval=4 \
    --plot_lat_range=10,50 \
    --plot_lon_range=110,145 \
    --zarr_path=path_to_era5.zarr \
    --ens_zarr=path_to_ifsens.zarr \
    --ens_dim=number \
    --plot_ens_mean \

python track.py 2018271N06154 \
    --ibtracs_file=ibtracs_all_list_v04r01.csv \
    --startdate 2018093000 \
    --n_steps 28 \
    --plot_interval=4 \
    --plot_lat_range=10,50 \
    --plot_lon_range=110,145 \
    --zarr_path=path_to_era5.zarr \
    --normalization_json=../ERA5_normal_1979_2017.json \
    --latent_path=path_to_pred.npy \
    --ens_dim=idx \
    --plot_ens_mean \

        
# Hurricane Dorian, figsize (9, 11), inner_box_sizes=[6,3,0]

python track.py 2019236N10314 \
    --ibtracs_file=ibtracs_all_list_v04r01.csv \
    --startdate 2019082800 \
    --n_steps 40 \
    --plot_interval=4 \
    --plot_lat_range=10,50 \
    --plot_lon_range=265,300 \
    --zarr_path=path_to_era5.zarr \
    --ens_zarr=path_to_ifsens.zarr \
    --ens_dim=number \
    --plot_ens_mean \

python track.py 2019236N10314 \
    --ibtracs_file=ibtracs_all_list_v04r01.csv \
    --startdate 2019082800 \
    --n_steps 40 \
    --plot_interval=4 \
    --plot_lat_range=10,50 \
    --plot_lon_range=265,300 \
    --zarr_path=path_to_era5.zarr \
    --normalization_json=../ERA5_normal_1979_2017.json \
    --latent_path=path_to_pred.npy \
    --ens_dim=idx \
    --plot_ens_mean \

# Hurricane Lorenzo #figsize=(10, 7), inner_box_sizes=[7,4,1]

python track.py 2019266N11341 \
    --ibtracs_file=ibtracs_all_list_v04r01.csv \
    --startdate 2019092400 \
    --n_steps 28 \
    --plot_interval=4 \
    --plot_lat_range=10,33 \
    --plot_lon_range=302,338 \
    --zarr_path=path_to_era5.zarr \
    --ens_zarr=path_to_ifsens.zarr \
    --ens_dim=number \
    --plot_ens_mean \

python track.py 2019266N11341 \
    --ibtracs_file=ibtracs_all_list_v04r01.csv \
    --startdate 2019092400 \
    --n_steps 28 \
    --plot_interval=4 \
    --plot_lat_range=10,33 \
    --plot_lon_range=302,338 \
    --zarr_path=path_to_era5.zarr \
    --normalization_json=../ERA5_normal_1979_2017.json \
    --latent_path=path_to_pred.npy \
    --ens_dim=idx \
    --plot_ens_mean \

"""
