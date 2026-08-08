"""Validate an on-disk DALES preprocessing tree without running training."""
import argparse

from .dales import DALESCloudStore


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True, help="DALES_ROOT containing original_ply/")
    parser.add_argument("--sub-grid-size", type=float, default=0.32)
    args = parser.parse_args(argv)
    store = DALESCloudStore(args.data_root, args.sub_grid_size, labeled_point="100%")
    for split in ("training", "validation"):
        for name, tree in zip(store.input_names[split], store.input_trees[split]):
            print(f"OK {split}: {name} ({len(tree.data)} subsampled points)")
    print(f"OK validation projections: {len(store.val_proj)}")


if __name__ == "__main__":
    main()
