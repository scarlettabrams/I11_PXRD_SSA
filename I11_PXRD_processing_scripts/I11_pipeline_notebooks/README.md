# I11 PXRD Data Processing Scripts

Scripts 02-05 are based off of / enhancements on scripts developed by Beth Galtry for data processing on I11. All other scripts are developed by Scarlett Abrams

This repository contains Jupyter notebooks for the processing and analysis of **Powder X-ray Diffraction (PXRD)** data collected on **Beamline I11** at Diamond Light Source.  
These scripts were developed as part of ongoing PhD research into *“Optimising Pharmaceutical Forms through Flow Crystallisation with in situ Powder X-ray Diffraction and Inline Visual Analysis.”*

---

## Repository structure

All notebooks are standalone Jupyter notebooks intended to be run sequentially as part of the I11 PXRD workflow.

| Notebook | Description |
|-----------|--------------|
| `01_Calibration_SSA.ipynb` | Detector and geometry calibration using pyFAI; sets up integration parameters. |
| `02_Diffraction_Sorting_SSA.ipynb` | Sorting raw diffraction frames and extracting initial 1D patterns. |
| `03A_pyFAI_Thresholding_SSA.ipynb` | Applies pixel thresholding to remove noise and artefacts. |
| `03B_Threshold_Merging_SSA.ipynb` | Merges thresholded frames into 1D patterns. |
| `04A_Pre_Merging_Baselining_SSA.ipynb` | Baseline correction prior to merging. |
| `04B_merging_baselineCorr_patterns_SSA.ipynb` | Combines corrected datasets and applies baseline merging routines. |
| `05_Post_Merging_Baselining_SSA.ipynb` | baseline correction on merged PXRD patterns. |
| `06_FEP_Subtraction_SSA.ipynb` | FEP tubing subtraction for I11 data. |
| `07_PXRD_Pattern_Stacking_SSA.ipynb` | Generates stacked plots for condition comparison. |
| `PXRD_Viewer_SSA.ipynb` | Interactive viewer for processed `.xy` or `.xlsx` patterns. |

---

## Setup

### 1. Clone the repository

git clone https://github.com/scarlettabrams/I11_PXRD_processing_scripts.git
cd I11_PXRD_processing_scripts

### 2. Create and activate a conda environment
conda create -n i11_pxrd_env python=3.10
conda activate i11_pxrd_env

### 3. Install dependencies 
pip install -r requirements.txt

### 4. Running the notebooks
Open Anaconda Prompt.
Activate your environment:
  conda activate i11_pxrd_env
Launch Jupyter Notebook:
  jupyter notebook
Open each notebook (01_... → 07_...) in sequence.
Each notebook saves intermediate and final data in clearly labelled subfolders (e.g. 05_merged_patterns, 06_fep_subtraction_results).

---

### Manual

For detailed workflow explanations, refer to the companion document:
“I11 Data Processing Manual” — this provides full context, example outputs, and scientific rationale for each stage.

Author: Scarlett Abrams and Beth Galtry 
Affiliation: Diamond Light Source & University of Nottingham


original scripts from Beth Galtry availble at: https://github.com/ESGaltry/I11_PXRD_processing_scripts
