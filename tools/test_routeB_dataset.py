import os
import torch

from ladcast.dataloader.routeB_dataset import ERA5RouteBDataset


def check_tensor(name: str, x: torch.Tensor) -> None:
    print(f"\n{name}")
    print("  shape:", tuple(x.shape))
    print("  dtype:", x.dtype)
    print("  min  :", float(x.min()))
    print("  max  :", float(x.max()))
    print("  mean :", float(x.mean()))
    print("  std  :", float(x.std()))
    print("  has_nan:", bool(torch.isnan(x).any()))
    print("  has_inf:", bool(torch.isinf(x).any()))


def main():
    ds_path = os.path.expanduser("~/data/ERA5_ladcast_routeB_1979_2024.zarr")
    norm_path = os.path.expanduser("~/ladcast/static/ERA5_routeB_normal_1979_2017.json")

    print("==== Test dataset without normalization ====")
    ds_raw = ERA5RouteBDataset(
        ds_path=ds_path,
        norm_path=norm_path,
        start_time="1979-01-01T05:00:00",
        end_time="1979-01-03T05:00:00",
        normalize=False,
        return_time=True,
    )
    print("len(ds_raw) =", len(ds_raw))

    sample0 = ds_raw[0]
    print("sample0 keys:", sample0.keys())
    print("sample0 time:", sample0["time"])
    check_tensor("raw sample0 x", sample0["x"])

    assert sample0["x"].shape == (70, 121, 240), f"Unexpected shape: {sample0['x'].shape}"
    assert sample0["x"].dtype == torch.float32, f"Unexpected dtype: {sample0['x'].dtype}"
    assert not torch.isnan(sample0["x"]).any(), "Found NaN in raw sample"
    assert not torch.isinf(sample0["x"]).any(), "Found Inf in raw sample"

    print("\n==== Test dataset with normalization ====")
    ds_norm = ERA5RouteBDataset(
        ds_path=ds_path,
        norm_path=norm_path,
        start_time="1979-01-01T05:00:00",
        end_time="1979-01-03T05:00:00",
        normalize=True,
        return_time=True,
    )
    print("len(ds_norm) =", len(ds_norm))

    sample1 = ds_norm[0]
    print("sample1 keys:", sample1.keys())
    print("sample1 time:", sample1["time"])
    check_tensor("normalized sample1 x", sample1["x"])

    assert sample1["x"].shape == (70, 121, 240), f"Unexpected shape: {sample1['x'].shape}"
    assert sample1["x"].dtype == torch.float32, f"Unexpected dtype: {sample1['x'].dtype}"
    assert not torch.isnan(sample1["x"]).any(), "Found NaN in normalized sample"
    assert not torch.isinf(sample1["x"]).any(), "Found Inf in normalized sample"

    print("\n==== Compare first few timestamps ====")
    for i in range(min(3, len(ds_norm))):
        s = ds_norm[i]
        print(f"idx={i}, time={s['time']}, shape={tuple(s['x'].shape)}")

    print("\nAll checks passed.")


if __name__ == "__main__":
    main()