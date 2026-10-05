"""
make_test_data.py — generates a small set of fake i11-1-<n>.nxs / pixium_<n>.hdf
pairs so you can test app_ssa_V3_beamline.py end-to-end WITHOUT needing real
beamtime data or waiting for real files to land.

Creates files compatible with both:
  - load_hdf_frame()  (Visual triage)   -> reads "entry/data/data"
  - load_nxs_frame()  (Inline loop etc) -> reads "/entry1/pixium_hdf/data"

Some frames get a few bright synthetic "peaks" added (roughly what real
diffraction spots would look like against noise) so Visual triage has
something meaningful to classify — most frames are plain noise
("background"), a few have peaks ("diffraction").

Usage:
    python make_test_data.py --out "C:\\Users\\<you>\\test_raw_dump" --n 20 --start 100

Then point the Data directories tab's "Raw beamline directory" at --out.
"""

import argparse
import os
import numpy as np
import h5py


def make_frame(shape, add_peaks):
    frame = np.random.poisson(lam=5.0, size=shape).astype(np.float32)
    if add_peaks:
        n_peaks = np.random.randint(3, 8)
        for _ in range(n_peaks):
            cy, cx = np.random.randint(0, shape[0]), np.random.randint(0, shape[1])
            yy, xx = np.ogrid[:shape[0], :shape[1]]
            dist2 = (yy - cy) ** 2 + (xx - cx) ** 2
            frame += (300 * np.exp(-dist2 / 8.0)).astype(np.float32)
    return frame


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, help="Output directory (the fake raw beamline dump)")
    parser.add_argument("--n", type=int, default=20, help="Number of collections to generate")
    parser.add_argument("--start", type=int, default=100, help="Starting collection number")
    parser.add_argument("--shape", type=int, nargs=2, default=[200, 200],
                         help="Frame height/width (small on purpose — real detector is 2881x2880, "
                              "this is just for testing the app's plumbing, not real integration)")
    parser.add_argument("--diffraction-fraction", type=float, default=0.3,
                         help="Fraction of frames that get synthetic peaks added (rest are plain noise)")
    parser.add_argument("--orphan-every", type=int, default=0,
                         help="If >0, every Nth collection gets ONLY an .hdf or ONLY an .nxs "
                              "written (not both) — useful for testing the 'missing pair' / "
                              "late-arrival handling in Data sorting.")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    shape = tuple(args.shape)
    rng = np.random.default_rng()

    written_nxs = written_hdf = orphaned = 0

    for i in range(args.n):
        cid = args.start + i
        is_diffraction = rng.random() < args.diffraction_fraction
        frame = make_frame(shape, add_peaks=is_diffraction)

        write_nxs = write_hdf = True
        if args.orphan_every and (i + 1) % args.orphan_every == 0:
            orphaned += 1
            if rng.random() < 0.5:
                write_nxs = False
            else:
                write_hdf = False

        if write_hdf:
            hdf_path = os.path.join(args.out, f"pixium_{cid}.hdf")
            with h5py.File(hdf_path, "w") as f:
                f.create_dataset("entry/data/data", data=frame)
            written_hdf += 1

        if write_nxs:
            nxs_path = os.path.join(args.out, f"i11-1-{cid}.nxs")
            with h5py.File(nxs_path, "w") as f:
                f.create_dataset("entry1/pixium_hdf/data", data=frame[np.newaxis, :, :])
            written_nxs += 1

        tag = "diffraction-like" if is_diffraction else "background-like"
        pair_note = "" if (write_nxs and write_hdf) else " (ORPHANED — only one half written)"
        print(f"  collection {cid}: {tag}{pair_note}")

    print(f"\nDone. {written_nxs} .nxs + {written_hdf} .hdf file(s) written to {args.out}")
    if orphaned:
        print(f"{orphaned} collection(s) deliberately orphaned (missing one half) for testing.")
    print(f"~{int(args.diffraction_fraction * args.n)} frame(s) have synthetic diffraction-like peaks.")


if __name__ == "__main__":
    main()
