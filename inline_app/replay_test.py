"""
replay_test.py  -  rehearse the inline chain without a beamtime
================================================================
Copies frames you already have into a "watched" folder one at a time, at beamline pace, named the way
the engineer's subtraction script names them (I11-1-<collection number>_subtracted.npy). Run it in one
terminal and inline_worker.py in another, then watch the app fill up.

    python replay_test.py --src E:/2D_sub_appV4_trial/AP39_CBZ/<corrected frames> ^
        --dest E:/AP40_rehearsal/subtracted --interval 5

Source frames can be .npy, .hdf, .h5 or .nxs (2D corrected frames, e.g. the offline app's 01_corrected_2D).
--format npy  writes I11-1-<number>_subtracted.npy          (default)
--format hdf  writes I11-1-<number>.hdf, an HDF5 file with the frame at entry1/pixium_hdf/data, like the corrected files the
              subtraction script writes (use --suffix "" for this)
Never touches the source folder. Writes only into --dest.

--mode atomic   write to a temporary name, then rename (a file appears complete)         [default]
--mode slow     write straight to the final name in two halves with a pause between them,
                like a script that is still writing; this checks the worker waits for it
--mode mixed    every third frame is written slowly, the rest atomically
"""
import argparse
import io
import os
import time
from pathlib import Path

import numpy as np


def find_sources(src: Path, recursive: bool):
    exts = (".npy", ".hdf", ".h5", ".hdf5", ".nxs")
    it = src.rglob("*") if recursive else src.iterdir()
    return sorted(p for p in it if p.is_file() and p.suffix.lower() in exts)


def load_frame(path: Path):
    if path.suffix.lower() == ".npy":
        a = np.load(path, allow_pickle=False)
        return a[0] if a.ndim == 3 else a
    import h5py
    with h5py.File(path, "r") as f:
        for p in ["entry1/pixium_hdf/data", "entry/data/data", "entry1/data/data",
                  "entry1/Pixium10:detector/data", "entry/detector/data"]:
            if p in f:
                d = f[p][()]
                return d[0] if d.ndim == 3 else d
        found = []
        f.visititems(lambda n, o: found.append(n) if hasattr(o, "ndim") and o.ndim in (2, 3) else None)
        if found:
            d = f[max(found, key=lambda n: int(np.prod(f[n].shape[-2:])))][()]
            return d[0] if d.ndim == 3 else d
    raise RuntimeError(f"No 2D dataset found in {path}")


def _serialise(dest: Path, arr):
    """File content as bytes, in the format the destination name asks for."""
    buf = io.BytesIO()
    if dest.suffix.lower() == ".npy":
        np.save(buf, arr, allow_pickle=False)
    else:
        import h5py
        with h5py.File(buf, "w") as f:
            f.create_dataset("entry1/pixium_hdf/data", data=arr[None].astype(np.float32))
    return buf.getvalue()


def write_atomic(dest: Path, arr):
    tmp = dest.with_name(".part_" + dest.name)
    tmp.write_bytes(_serialise(dest, arr))
    os.replace(tmp, dest)


def write_slow(dest: Path, arr, stall_s):
    data = _serialise(dest, arr)
    half = len(data) // 2
    with open(dest, "wb") as f:
        f.write(data[:half])
        f.flush()
        time.sleep(stall_s)
        f.write(data[half:])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True, help="folder of existing 2D corrected frames")
    ap.add_argument("--dest", required=True, help="the folder the worker watches (created if missing)")
    ap.add_argument("--interval", type=float, default=5.0, help="seconds between frames")
    ap.add_argument("--start-number", type=int, default=900000, help="first fake collection number")
    ap.add_argument("--prefix", default="I11-1-")
    ap.add_argument("--suffix", default="_subtracted")
    ap.add_argument("--format", choices=["npy", "hdf"], default="npy", help="file type to write")
    ap.add_argument("--n", type=int, default=0, help="stop after this many frames (0 = all)")
    ap.add_argument("--mode", choices=["atomic", "slow", "mixed"], default="atomic")
    ap.add_argument("--stall", type=float, default=3.0, help="pause (s) in the middle of a slow write")
    ap.add_argument("--recursive", action="store_true")
    a = ap.parse_args()

    src, dest = Path(a.src), Path(a.dest)
    if not src.is_dir():
        raise SystemExit(f"Source folder not found: {src}")
    if src.resolve() == dest.resolve():
        raise SystemExit("Source and destination are the same folder.")
    dest.mkdir(parents=True, exist_ok=True)
    files = find_sources(src, a.recursive)
    if a.n:
        files = files[: a.n]
    if not files:
        raise SystemExit("No frames found in the source folder.")
    print(f"Replaying {len(files)} frames into {dest} every {a.interval:g} s ({a.mode}).")
    for k, p in enumerate(files):
        t0 = time.time()
        arr = np.ascontiguousarray(load_frame(p), dtype=np.float32)
        name = f"{a.prefix}{a.start_number + k}{a.suffix}.{a.format}"
        slow = a.mode == "slow" or (a.mode == "mixed" and k % 3 == 2)
        (write_slow(dest / name, arr, a.stall) if slow else write_atomic(dest / name, arr))
        print(f"{k + 1}/{len(files)}  {name}  <- {p.name}{'  (slow write)' if slow else ''}", flush=True)
        time.sleep(max(0.0, a.interval - (time.time() - t0)))
    print("Replay finished.")


if __name__ == "__main__":
    main()
