import pandas as pd
import re
import numpy as np
import argparse
import os

def normalize_features(input_filepath, output_filepath):
    try:
        # 1. Load Data
        df = pd.read_csv(input_filepath)
        print(f"Successfully loaded file: {input_filepath}")
        print(f"DataFrame shape: {df.shape}")

        # Rename Features for consistency
        print("\n--- Renaming Features ---")
        rename_dict = {}
        for col in df.columns:
            parts = col.split('_')
            # Rule 1: xxx_within_yy_zz -> neoplasm_within_yy_xxx_zz (Implicit neoplasm-centric)
            if len(parts) == 4 and parts[1] == 'within':
                # Example: necrosis_within_10_std -> neoplasm_within_10_necrosis_std
                new_name = f"neoplasm_within_{parts[2]}_{parts[0]}_{parts[3]}"
                rename_dict[col] = new_name
            # Rule 2: nearest_xxx_zz -> neoplasm_nearest_xxx_zz
            elif len(parts) == 3 and parts[0] == 'nearest':
                # Example: nearest_stroma_q25 -> neoplasm_nearest_stroma_q25
                new_name = f"neoplasm_nearest_{parts[1]}_{parts[2]}"
                rename_dict[col] = new_name
        
        if rename_dict:
            df.rename(columns=rename_dict, inplace=True)
            print(f"Renamed {len(rename_dict)} columns.")

        # 2. Identify 'within' features for normalization (divide by density)
        print("\n--- Processing 'within' features (Normalization by Density) ---")
        # Pattern: matches '..._within_..._..._...' but excluding entropy
        pattern_within = re.compile(r'(.+)_within_.+_(.+)_(?!entropy$)[^_]+$')
        columns_to_normalize = [col for col in df.columns if pattern_within.match(col)]

        if not columns_to_normalize:
            print("No columns found matching format 'xxx_within_yyy_zzz_aaa'.")
        else:
            print(f"Found {len(columns_to_normalize)} columns to normalize.")

        for col_name in columns_to_normalize:
            match = pattern_within.match(col_name)
            if match:
                # Extract target cell type
                # Assumes structure: [source]_within_[radius]_[target]_[stat]
                cell_type = col_name.split('_')[3] 
                density_col_name = f"{cell_type}_mean_density"
                
                new_col_name = f"{col_name}_norm"

                if density_col_name in df.columns:
                    # Normalize: feature / density
                    density_values = df[density_col_name]
                    df[new_col_name] = df[col_name].div(density_values).replace([np.inf, -np.inf], 0).fillna(0)
                # Note: Silent skip if density column not exists, to avoid log spam

        # 3. Identify 'nearest' features for normalization (multiply by sqrt(density))
        print("\n--- Processing 'nearest' features (Adjustment by Density) ---")
        pattern_nearest = re.compile(r'(.+)_nearest_(.+)_(?!entropy$)[^_]+$')
        columns_to_multiply = [col for col in df.columns if pattern_nearest.match(col)]

        if not columns_to_multiply:
            print("No columns found matching format 'xxx_nearest_yyy_zzz'.")
        else:
            print(f"Found {len(columns_to_multiply)} columns to adjust.")

        for col_name in columns_to_multiply:
            match = pattern_nearest.match(col_name)
            if match:
                # Extract target cell type
                # Assumes structure: [source]_nearest_[target]_[stat]
                cell_type = col_name.split('_')[2]
                density_col_name = f"{cell_type}_mean_density"

                new_col_name = f"{col_name}_norm"

                if density_col_name in df.columns:
                    # Adjustment: feature * sqrt(density) -> dimensionless distance index
                    df[new_col_name] = df[col_name] * np.sqrt(df[density_col_name])

        # 4. Save
        df.to_csv(output_filepath, index=False)
        print(f"\nProcessing complete.")
        print(f"Saved to: {output_filepath}")
        print(f"Final shape: {df.shape}")

    except FileNotFoundError:
        print(f"Error: File not found at '{input_filepath}'.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Normalize spatial features based on cell density.")
    parser.add_argument("-i", "--input", required=True, help="Path to input features CSV.")
    parser.add_argument("-o", "--output", required=True, help="Path to save normalized features CSV.")
    
    args = parser.parse_args()
    normalize_features(args.input, args.output)
