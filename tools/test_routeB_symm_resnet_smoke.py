import argparse
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ladcast.models.routeB_latent_ar import RouteBNonSymmResNet, RouteBSymmResNet


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Independent smoke test for RouteB ResNet AR predictors.")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--input_seq_len", type=int, default=2)
    parser.add_argument("--return_seq_len", type=int, default=3)
    parser.add_argument("--channels", type=int, default=8)
    parser.add_argument("--height", type=int, default=16)
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--hidden_channels", type=int, default=32)
    parser.add_argument("--num_blocks", type=int, default=3)
    parser.add_argument("--kernel_size", type=int, default=3)
    parser.add_argument("--max_lon_shift", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    expected = (
        args.batch_size,
        args.return_seq_len,
        args.channels,
        args.height,
        args.width,
    )
    x = torch.randn(
        args.batch_size,
        args.input_seq_len,
        args.channels,
        args.height,
        args.width,
        device=device,
    )

    non_symm = RouteBNonSymmResNet(
        args.input_seq_len,
        args.return_seq_len,
        args.channels,
        hidden_channels=args.hidden_channels,
        num_blocks=args.num_blocks,
        kernel_size=args.kernel_size,
    ).to(device)
    non_symm.train()
    pred_non_symm = non_symm(x)
    assert pred_non_symm.shape == expected, pred_non_symm.shape
    pred_non_symm.square().mean().backward()

    symm = RouteBSymmResNet(
        args.input_seq_len,
        args.return_seq_len,
        args.channels,
        hidden_channels=args.hidden_channels,
        num_blocks=args.num_blocks,
        kernel_size=args.kernel_size,
        max_lon_shift=args.max_lon_shift,
    ).to(device)
    symm.train()
    pred_train = symm(x)
    assert pred_train.shape == expected, pred_train.shape
    pred_train.square().mean().backward()

    symm.eval()
    pred_deterministic = symm(x, inference_mode="deterministic")
    assert pred_deterministic.shape == expected, pred_deterministic.shape

    pred_random = symm(x, inference_mode="random_single")
    assert pred_random.shape == expected, pred_random.shape

    pred_group4 = symm(x, inference_mode="group_mean", num_symmetry_samples=4)
    assert pred_group4.shape == expected, pred_group4.shape

    pred_group8 = symm(x, inference_mode="group_mean", num_symmetry_samples=8)
    assert pred_group8.shape == expected, pred_group8.shape

    print("routeB resnet ablation smoke passed")


if __name__ == "__main__":
    main()
