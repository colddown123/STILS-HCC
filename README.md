# DASI-HCC 

**cell Density, spatial Aggregation, and Spatial Interaction (DASI)**

This repository contains the codebase for analyzing spatial Tumor Microenvironment partten in Hepatocellular Carcinoma (HCC), including feature extraction from cell segmentation results, feature normalization, and survival analysis using machine learning.

## Overview

The DASI-HCC pipeline consists of three main modules:
1.  **Spatial Feature Extraction**: Extracts complex biological spatial statistics from cell detection results (HoverNet JSON format).
2.  **Feature Normalization**: Normalizes spatial features based on cell density to ensure comparability across ROI/patches.
3.  **Survival Analysis**: A robust machine learning pipeline involving Bootstrap Feature Selection and Random Survival Forests (RSF) to predict patient prognosis.

## Installation

Ensure you have Python 3.8+ installed. Install dependencies utilizing:

```bash
pip install -r requirements.txt
```

### Dependencies
- `numpy`, `pandas`, `scipy`
- `scikit-learn`
- `scikit-survival` (for survival analysis models)
- `lifelines` (for KM curves)
- `optuna` (for hyperparameter optimization)

## Usage

### 1. Spatial Feature Extraction

Extracts spatial metrics (metrics within radius, nearest neighbor distances, entropies) from cell segmentation files.

**Input Format**:
The script expects JSON files (HoverNet output format).
Structure:
```json
{
  "nuc": {
    "1": { "type": 1, "centroid": [x, y], ... },
    "2": { "type": 3, "centroid": [x, y], ... },
    ...
  }
}
```
*   Type 1: Neoplasm
*   Type 3: Stroma
*   Type 4: Necrosis
*   Type 5+: Immune cell subtypes (mapped internally)

**Command**:
```bash
python spatial_extraction.py --input_dir ./data/jsons --output_dir ./results/features
```

### 2. Feature Normalization

Normalizes the raw extracted features. 'Within-radius' counts are divided by cell density, and 'Nearest-neighbor' distances are multiplied by the square root of cell density to create dimensionless indices.

**Command**:
```bash
python feature_normalization.py -i ./results/features/spatial_features.csv -o ./results/features/normalized_features.csv
```

### 3. Survival Analysis & Modeling

Runs the complete survival analysis pipeline:
1.  Splits data into Train (70%), Internal Test (30%), and a fixed External Validation set (specified by Center ID).
2.  Performs Hyperparameter Optimization using Optuna.
3.  Selects features using Bootstrap aggregation of RSF importance.
4.  Trains final model and evaluates C-index.
5.  Plots Kaplan-Meier curves.

**Command**:
```bash
python survival_analysis.py \
  --data ./data/clinical_and_features.csv \
  --output_dir ./results/model_output \
  --center_id_ext 4 \
  --trials 50
```

**Input Data for Analysis**:
The CSV must contain:
- `filename` (ID)
- `survival_months` (Time)
- `event` (Status: 0/1)
- `center` (Center ID for splitting)
- Feature columns (generated from step 2)

## Acknowledgements

Designed for HCC immune microenvironment research.
