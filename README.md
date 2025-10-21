STILS-HCC
Feature extraction, model pre-training, training, inference scripts, and trained model files for HCC spatial analysis.

Project Overview
This project provides a complete pipeline for spatial feature extraction, batch effect correction, feature selection, survival modeling, and inference evaluation in hepatocellular carcinoma (HCC), along with trained model files.

Directory Structure
(Please fill in your actual directory structure here.)

1. Spatial Feature Extraction
Script: spatial_features.py
Function: Batch processing of cell segmentation JSON files to compute multi-radius spatial neighborhood features (e.g., neighborhood counts, nearest neighbor distances, immune cell spatial relationships), outputting results as CSV.
Input: JSON (from cell segmentation result via hovernet/deepliif) directories configured in path_pairs.
Output: Per-sample spatial statistics CSV.

2. Batch Effect Correction & Univariate Analysis
Script: combat.py
Function:
Fill missing values, filter, and standardize raw features.
Perform batch effect correction using ComBat and visualize PCA before/after correction.
Support applying ComBat parameters from the training set to external data.
Conduct univariate Cox regression and binary statistical tests.
Input: Training/external CSV files.
Output: Batch-corrected feature CSV, PCA plots, univariate analysis results, etc.

3. Full-Feature Pre-training & Feature Importance Evaluation
Script: COX_pretrain.py
Function:
Train multiple survival models (LassoCox, RSF, BoostingCox, SVM) with all features and perform hyperparameter tuning.
Automatically save best parameters, model files, and feature importance rankings (including cross-validation robustness).
Input: Batch-corrected training/external CSV.
Output: Best model files, parameters, feature importance CSVs.

4. Feature Selection & Model Training (SFS)
Script: COX_train.py
Function:
Build a feature pool from pre-training feature importance rankings.
Use Sequential Forward Selection (SFS) to iteratively select optimal feature combinations and train RSF models at each step.
Save the best model, feature set, cross-validation results, and final model at each step.
Input: Feature pool CSV, training/external CSV.
Output: Best models per step, final models, feature importance, training logs, etc.

5. Inference & Evaluation
Script: COX_infer.py
Function:
Load trained models and feature sets to predict risk scores and survival probabilities on new data.
Automatically plot KM curves by risk group, compute logrank test, c-index, time-dependent AUC, calibration curves, and other metrics.
Output inference results and evaluation plots.
Input: Trained model, inference data CSV.
Output: Inference results CSV, KM curves, calibration curves, evaluation metrics CSV.

6. Trained Model Files
SFS_GBS/final_best_models/BEST_BoostingCox_top_3.pkl
BoostingCox model trained with the top 3 SFS-selected features.
SFS_RSF/final_best_models/BEST_RSF_top_4.pkl
RSF model trained with the top 4 SFS-selected features.
Dependencies
Python >= 3.7
numpy, pandas, scikit-learn, scikit-survival, lifelines, matplotlib, pycombat, scipy

Example Usage
1. Batch Effect Correction: python combat.py --input train_batch.csv --output_dir ./batch_effect --external chenzhou_batch.csv --apply_combat
2. Spatial Feature Extraction: python spatial_features.py
3. Full-Feature Pre-training: python COX_pretrain.py
4. SFS Feature Selection & Model Training: python COX_train.py
5. Inference & Evaluation:python COX_infer.py

Notes
Please adjust paths and parameters according to your actual data locations.
Intermediate and final results will be automatically saved to the specified output directories during training and inference.
For detailed parameters and workflow, refer to the comments in each script.
