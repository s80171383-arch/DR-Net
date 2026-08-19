"""Benchmark exact hierarchy backends on the Stage 6D ISPRS batch shape."""

import argparse
import time

import torch

from .utils.knn import build_hierarchy


def time_backend(points, backend, ratios, repeats):
    durations = []
    for _ in range(repeats):
        start = time.perf_counter()
        hierarchy = build_hierarchy(points, ratios, backend=backend)
        durations.append(time.perf_counter() - start)
        del hierarchy
    return min(durations)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-points", type=int, default=65536)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    points = torch.randn(args.batch_size, args.num_points, 3,
                         generator=torch.Generator().manual_seed(args.seed))
    ratios = (4, 4, 4, 4, 2)
    reference = time_backend(points, "torch", ratios, args.repeats)
    optimized = time_backend(points, "ckdtree", ratios, args.repeats)
    print(f"torch:   {reference:.3f} s")
    print(f"ckdtree: {optimized:.3f} s")
    print(f"speedup: {reference / optimized:.2f}x")


if __name__ == "__main__":
    main()
