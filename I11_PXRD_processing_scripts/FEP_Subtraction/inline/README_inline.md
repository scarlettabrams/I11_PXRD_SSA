# Inline 2D FEP subtraction (fep2d_inline.py): notes for the software engineer

Status 07/10/26: working and tested on real I11 frames (AP39, carbamazepine in EtOH:water, Pixium detector, 2881 x 2880). Not yet run on a live beamline feed.

## The one job

Raw 2D frame in, background-subtracted 2D frame out.

1. Before the crystal solution runs, the user collects reference frames (solvent in the FEP tube, no crystals) and merges them into one reference file, `merged_reference.npz`, using the offline app.
2. During the experiment this script takes each new raw frame, fits one scale factor `s` so that `s * reference` matches it, and writes `frame - s * reference` as a 2D file with the same name as the raw frame.
3. Diffraction sorting, integration and baselining are done afterwards in the offline app, on the corrected 2D files. This script does none of that.

The maths is the offline app's "Subtract frames" tab, unchanged (`fep2d_v3.subtract_frame`): robust scale fit on the whole detector (Bragg spots rejected), no offset, negative values kept, masked pixels set to 0. The output is pixel-for-pixel identical to what the offline app writes (checked on real frames, difference 0.0), and it writes the same `subtraction_diagnostics.csv`, so the app reads it with no change.

## Files you need

```
fep2d_inline.py      the subtractor, folder watcher and command line
fep2d_pipeline.py    shared settings and helpers (also used by the offline app)
fep2d_v3.py          the 2D maths and file reading (tested code, unchanged)
```
Python packages: numpy, scipy, h5py, pyFAI. No GPU. (`fep2d_inline_1d_preview.py` is an unused earlier design that goes straight to a 1D pattern; ignore it.)

Inputs:
- `merged_reference.npz` made in the offline app (tab 1, about 46 or more reference frames);
- `.poni` calibration and the detector mask `.npy` (True = bad pixel), from the beamtime calibration. The calibration is needed only to build the extra 2theta mask (5.30 to 6.10 deg), which hides the strong FEP line.

## Use from your own code

```python
import numpy as np, pyFAI
import fep2d_v3 as fm
from fep2d_inline import InlineSubtractor

sub = InlineSubtractor(fm.load_reference("merged_reference.npz"),
                       pyFAI.load("calib.poni"), np.load("mask.npy"))   # once, under a second
res = sub.process(frame_2d, name="pixium_146588")     # numpy array in, never raises
res.ok, res.flags     # not ok, or any flag = look at this frame
res.corrected         # float32 2D array, same shape as the raw frame
res.scale             # fitted scale factor

sub.process_file("pixium_146588.hdf", "E:/out/Run7/01_corrected_2D")   # reads, subtracts, writes
```

## Folder watcher and command line

```
python fep2d_inline.py --ref merged_reference.npz --poni calib.poni --mask mask.npy \
       --watch E:/run/frames --out E:/out --run Run7
```
Writes under `<out>/<run>/`:
- `01_corrected_2D/<same name as the raw frame>` (.hdf, self-contained, about 59 MB per frame; plan disk space accordingly);
- `01_corrected_2D/subtraction_diagnostics.csv` (scale factor per frame etc., same columns as the app);
- `inline_log.csv` (one line per frame including flags and failures);
- `inline_manifest.json` (settings, reference and calibration used).

Only `.hdf` files are watched (each `.nxs` links to its `.hdf`, so watching both counts every frame twice). A file is picked up once its size stops changing and it opens as HDF5. A file that still will not open after 60 s is logged as failed and the watcher carries on. Each output is written under a hidden temporary name and then renamed, so a reader never sees a half-written file. Tested with a slowly written file and a corrupt file.

## Speed (2 CPU cores, one frame at a time)

About 4.2 s per frame including writing the 59 MB output (about 3.3 s inside the watcher). Frames arrive about 27 s apart. Start-up is under a second.

## Checks on every frame

Not written, logged as failed: wrong shape, NaN or infinite pixels, total counts under 30% of the reference total (shutter closed or dark), file that will not open. A closed-shutter or corrupt frame therefore never reaches the sorting step.
Written but flagged in `inline_log.csv`: scale outside 0.3 to 3.0, robust fit kept under half the pixels, scale more than 5 robust sigma from the running median (after 10 frames). On 96 R007 frames: 1 drift flag, scales 0.88 to 1.17.

## Things to know

- The reference must match the setup (same tube, solvent and geometry). Rebuild it after any change. A drifting scale is the main warning that it no longer matches.
- Everything between 5.30 and 6.10 deg 2theta is set to 0 in the corrected frame (it hides the strong FEP line). A crystal peak inside that window cannot be seen. Treat it as missing data, not as zero signal.
- This script does not decide hit or non-hit and does not integrate.

## Open decisions for you

- Raw frames are watched in a folder and read as files. Is that how your system will hand them over, or should it pass arrays? Both are supported.
- A failed frame is skipped and logged. It is never replaced by the previous frame.
