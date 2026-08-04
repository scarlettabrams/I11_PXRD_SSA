# Copilot Instructions for I11 PXRD Processing Scripts

## Project Overview
This is a **Jupyter notebook-based data processing pipeline** for Powder X-ray Diffraction (PXRD) data from Diamond Light Source Beamline I11. The pipeline transforms 2D detector images into 1D diffraction patterns through sequential processing stages (calibration → sorting → thresholding → baseline correction → merging → analysis).

## Architecture & Data Flow

### Sequential Processing Pipeline
Each notebook processes data in a strict sequence (01 → 07):
1. **01_Calibration**: Detector geometry calibration using pyFAI (outputs `.poni` geometry file)
2. **02_Diffraction_Sorting**: Identifies good diffraction frames by scoring and sorting raw detector images
3. **03A_pyFAI_Thresholding**: Integrates 2D detector images → 1D I(2θ) patterns with noise thresholding
4. **03B_Threshold_Merging**: Merges multiple thresholded frames into composite 1D patterns
5. **04A_Pre_Merging_Baselining**: Applies baseline correction before frame merging
6. **04B_Merging_BaselineCorr**: Combines datasets and applies merged baseline routines
7. **05_Post_Merging_Baselining**: Final baseline correction on merged patterns
8. **06_FEP_Subtraction**: Removes fluoroelastomer (FEP) tubing background signature
9. **07_PXRD_Pattern_Stacking**: Generates stacked plots for condition comparison

Each notebook saves intermediate results in clearly labeled subfolders (e.g., `05_merged_patterns/`, `06_fep_subtraction_results/`).

### Key Data Formats
- **Input**: HDF5/NeXus files from I11 (`.nxs`, `.hdf`) containing detector frames in `/entry/data/data` or `/entry1/pixium_hdf/data`
- **Intermediate**: 1D patterns as numpy arrays or CSV/Excel files
- **Output**: 1D patterns (`.xy` or `.xlsx` format) for scientific analysis

## Developer Workflows

### Loading Detector Data
Two robust loader patterns are used across notebooks:
```python
# Pattern 1: pyFAI GUI workflow (01_Calibration)
# Run calibration interactively: pyfai-calib2 "path/to/calibration.hdf"
# Then load generated .poni file for geometry

# Pattern 2: Fallback loader with fabio → h5py (02_Diffraction_Sorting, 03A_Thresholding)
def load_detector_frame(filepath):
    ext = filepath.lower()
    try:
        if ext.endswith((".hdf", ".h5", ".nxs")):
            with h5py.File(filepath, "r") as f:
                data = f["entry/data/data"][()]  # or /entry1/pixium_hdf/data
                return data[0] if data.ndim == 3 else data
        elif ext.endswith((".tif", ".tiff", ".cbf")):
            return fabio.open(filepath).data
    except:
        # Fall back to h5py inspection if fabio fails
```

### Running Notebooks
```bash
conda create -n i11_pxrd_env python=3.10
conda activate i11_pxrd_env
pip install -r requirements.txt
jupyter notebook
# Execute cells sequentially, adjusting USER INPUTS section in each notebook
```

## Code Patterns & Conventions

### Notebook Structure
- **Comment header**: `# Cell N: brief description`
- **Configuration section**: `# ===== USER INPUTS =====` at top of notebook with hard-coded paths/parameters
- **Function definitions**: Documented with docstrings explaining physics/method
- **Visualization**: Matplotlib plots at key decision points (e.g., showing threshold effects, baseline fits)
- **tqdm progress bars**: Used for long-running file loops

### Scientific Conventions
- **Physical quantities**: Use 2-theta (2θ) for diffraction angle, I for intensity
- **Radial scoring** (02_Diffraction_Sorting): Detects ring-like diffraction structure by computing `std(radial_mean) × mean(radial_std)` — insensitive to absolute brightness, physically aligned with powder diffraction
- **Baseline correction**: Applied at multiple stages (pre/post merging) to remove instrument drift and background
- **Pixel masking**: Hot pixel masks generated in calibration step and reused across all integration steps

### File Path Conventions
- Use forward slashes `/` even on Windows: `Path = r"D:/data/folder/"`
- Append trailing `/` to base directories for consistency
- Use `pathlib.Path` or `os.path.join()` for cross-platform compatibility
- Raw data typically in `RAW_2D/` subdirectory; backgrounds in `Potential backgrounds/` subfolder

## Dependencies & External Tools

### Core Scientific Stack
- **numpy/scipy**: Numerical computations, filtering
- **pyFAI**: Detector calibration and 2D→1D integration (critical dependency)
- **h5py**: Reading NeXus/HDF5 detector files from I11
- **pandas**: Tabular data handling
- **matplotlib**: All visualization
- **pybaselines**: Baseline correction (installed dynamically in notebooks with `pip install`)

### Beamline-Specific
- **pyFAI calibration workflow**: Generate `.poni` geometry file via interactive `pyfai-calib2` GUI tool
- **NeXus/HDF5 structure**: I11 data uses `/entry/data/data` or `/entry1/pixium_hdf/data` for detector frames
- **FEP background**: I11-specific tubing material that must be subtracted (notebook 06)

## Integration Points & Cross-Notebook Patterns

- **Calibration file**: 01 outputs `.poni` file → reused in all thresholding/integration steps (03A+)
- **Frame sorting**: 02 produces a list of good frames → 03A uses this for selective integration
- **Baseline models**: 04A baseline coefficients → used in 04B merging
- **Output format flexibility**: 07 reads patterns in `.xy` or `.xlsx` format for stacking

## Testing & Debugging Strategy

- **Interactive validation**: Each notebook includes matplotlib plots at key stages to visually inspect data quality
- **Sample data inspection**: Print statements and conditional visualization for selected frames
- **Fallback loaders**: When files fail to load, system tries multiple formats (fabio → h5py) with informative error messages
- **Version trials folder**: `02_Diffraction_Sorting_trials/` contains iterative development versions (v2-v6) for reference and testing alternative scoring metrics

---

**Key References**: [README.md](../../README.md) | [requirements.txt](../../requirements.txt) | Related publication: "Optimising Pharmaceutical Forms through Flow Crystallisation with in situ PXRD"
