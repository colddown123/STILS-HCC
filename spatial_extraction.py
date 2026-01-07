import json
import os
import glob
import numpy as np
import csv
import argparse
from scipy.spatial import cKDTree
from scipy.stats import entropy

# Cell type mapping (HoverNet standard)
# 1: Neoplasm, 2: Inflammatory, 3: Connective/Stroma, 4: Necrosis, 5+: Immune Subtypes
TYPE_MAP = {
    "1": "neoplasm",
    "2": "inflam",
    "3": "stroma",
    "4": "necrosis",
    "5": "CD3",
    "6": "CD8",
    "7": "FOXP3",
    "8": "CD20",
    "9": "CD68",
    "10": "CD163",
    "11": "CD66B",
    "12": "CD57",
    "13": "PD-L1",
    "14": "CAIX"
}

RADIUS_LIST = [10, 50]

# Immune cell subtypes
IMMUNE_TYPES = ["CD3", "CD8", "FOXP3", "CD20", "CD68", "CD163", "CD66B", "CD57", "PD-L1", "CAIX"]
IMMUNE_TYPE_IDS = {
    "CD3": "5", "CD8": "6", "FOXP3": "7", "CD20": "8",
    "CD68": "9", "CD163": "10", "CD66B": "11", "CD57": "12",
    "PD-L1": "13", "CAIX": "14"
}

def calc_entropy(arr, bins=10):
    """Calculate entropy of a distribution."""
    if arr.size == 0:
        return 0
    hist, _ = np.histogram(arr, bins=bins)
    hist = hist.astype(float)
    if hist.sum() == 0:
        return 0
    return entropy(hist, base=2)

def generate_header():
    """Generate CSV header for spatial features."""
    header = ["filename"]
    
    # 1. Neoplasm neighborhood statistics (Stroma/Necrosis)
    for r in RADIUS_LIST:
        for t in ["stroma", "necrosis"]:
            header += [
                f"{t}_within_{r}_mean", f"{t}_within_{r}_std", f"{t}_within_{r}_median",
                f"{t}_within_{r}_q10", f"{t}_within_{r}_q25", f"{t}_within_{r}_q75",
                f"{t}_within_{r}_q90", f"{t}_within_{r}_iqr", f"{t}_within_{r}_entropy"
            ]

    # 2. Nearest neighbor distances for Stroma/Necrosis
    for t in ["stroma", "necrosis"]:
        header += [
            f"nearest_{t}_mean", f"nearest_{t}_std", f"nearest_{t}_median",
            f"nearest_{t}_q10", f"nearest_{t}_q25", f"nearest_{t}_q75",
            f"nearest_{t}_q90", f"nearest_{t}_iqr", f"nearest_{t}_entropy"
        ]

    # 3. Interactions between Main Types (Neoplasm/Stroma/Necrosis) and Immune Cells
    for main_type in ["neoplasm", "stroma", "necrosis"]:
        for immune in IMMUNE_TYPES:
            for r in RADIUS_LIST:
                header += [
                    f"{main_type}_within_{r}_{immune}_mean", f"{main_type}_within_{r}_{immune}_std",
                    f"{main_type}_within_{r}_{immune}_median", f"{main_type}_within_{r}_{immune}_q10",
                    f"{main_type}_within_{r}_{immune}_q25", f"{main_type}_within_{r}_{immune}_q75",
                    f"{main_type}_within_{r}_{immune}_q90", f"{main_type}_within_{r}_{immune}_iqr",
                    f"{main_type}_within_{r}_{immune}_entropy"
                ]
            header += [
                f"{main_type}_nearest_{immune}_mean", f"{main_type}_nearest_{immune}_std",
                f"{main_type}_nearest_{immune}_median", f"{main_type}_nearest_{immune}_q10",
                f"{main_type}_nearest_{immune}_q25", f"{main_type}_nearest_{immune}_q75",
                f"{main_type}_nearest_{immune}_q90", f"{main_type}_nearest_{immune}_iqr",
                f"{main_type}_nearest_{immune}_entropy"
            ]

    # 4. Immune-Immune Interactions
    immune_pairs = [
        ("CD8", "FOXP3"), ("CD8", "CD20"),
        ("FOXP3", "CD8"), ("FOXP3", "CD20"),
        ("CD20", "CD8"), ("CD20", "FOXP3"),
        ("CD3", "CD20"), ("CD20", "CD3"),
        ("CD68", "CD163"), ("CD163", "CD68") # Macrophage interactions
    ]

    for main, target in immune_pairs:
        for r in RADIUS_LIST:
            header += [
                f"{main}_within_{r}_{target}_mean", f"{main}_within_{r}_{target}_std",
                f"{main}_within_{r}_{target}_median", f"{main}_within_{r}_{target}_q10",
                f"{main}_within_{r}_{target}_q25", f"{main}_within_{r}_{target}_q75",
                f"{main}_within_{r}_{target}_q90", f"{main}_within_{r}_{target}_iqr",
                f"{main}_within_{r}_{target}_entropy"
            ]
        header += [
            f"{main}_nearest_{target}_mean", f"{main}_nearest_{target}_std",
            f"{main}_nearest_{target}_median", f"{main}_nearest_{target}_q10",
            f"{main}_nearest_{target}_q25", f"{main}_nearest_{target}_q75",
            f"{main}_nearest_{target}_q90", f"{main}_nearest_{target}_iqr",
            f"{main}_nearest_{target}_entropy"
        ]
    return header

def calculate_stats(arr):
    if len(arr) == 0:
        return [0] * 9
    arr = np.array(arr)
    iqr = np.percentile(arr, 75) - np.percentile(arr, 25)
    ent = calc_entropy(arr)
    return [
        np.mean(arr), np.std(arr), np.median(arr),
        np.percentile(arr, 10), np.percentile(arr, 25),
        np.percentile(arr, 75), np.percentile(arr, 90),
        iqr, ent
    ]

def process_files(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    json_files = glob.glob(os.path.join(input_dir, '*.json'))
    
    output_csv = os.path.join(output_dir, 'spatial_features.csv')
    processed_txt = os.path.join(output_dir, 'processed_files.txt')

    processed_files = set()
    if os.path.exists(processed_txt):
        with open(processed_txt, 'r') as f:
            processed_files = set(line.strip() for line in f)

    write_header = not os.path.exists(output_csv)
    
    with open(output_csv, 'a', newline='', encoding='utf-8-sig') as f_csv, \
         open(processed_txt, 'a') as f_txt:
        
        writer = csv.writer(f_csv)
        if write_header:
            writer.writerow(generate_header())

        print(f"Found {len(json_files)} JSON files. processing...")

        for json_path in json_files:
            fname = os.path.basename(json_path)
            if fname in processed_files:
                continue

            try:
                with open(json_path, 'r') as f:
                    data = json.load(f)
            except Exception as e:
                print(f"Error reading {fname}: {e}")
                continue

            # Extract centroids by type
            coords_neoplasm = []
            coords_stroma = []
            coords_necrosis = []
            coords_immune = {k: [] for k in IMMUNE_TYPES}

            # Handle HoverNet format (dictionary of cells)
            cells = data.get('nuc', {})
            for cell in cells.values():
                t = str(cell.get('type'))
                c = cell.get('centroid')
                
                if t == "1":
                    coords_neoplasm.append(c)
                elif t == "3":
                    coords_stroma.append(c)
                elif t == "4":
                    coords_necrosis.append(c)
                
                for immune in IMMUNE_TYPES:
                    if t == IMMUNE_TYPE_IDS[immune]:
                        coords_immune[immune].append(c)

            # Convert to numpy arrays
            coords_neoplasm = np.array(coords_neoplasm)
            coords_stroma = np.array(coords_stroma)
            coords_necrosis = np.array(coords_necrosis)
            for k in coords_immune:
                coords_immune[k] = np.array(coords_immune[k])

            # Build KD-Trees
            tree_stroma = cKDTree(coords_stroma) if len(coords_stroma) > 0 else None
            tree_necrosis = cKDTree(coords_necrosis) if len(coords_necrosis) > 0 else None
            trees_immune = {k: cKDTree(v) if len(v) > 0 else None for k, v in coords_immune.items()}

            row = [fname]

            # --- Analysis 1: Neoplasm Neighborhood (Stroma/Necrosis) ---
            # Within Distance
            stats_within = {f"stroma_{r}": [] for r in RADIUS_LIST}
            stats_within.update({f"necrosis_{r}": [] for r in RADIUS_LIST})

            if len(coords_neoplasm) > 0:
                for center in coords_neoplasm:
                    for r in RADIUS_LIST:
                        count_stroma = len(tree_stroma.query_ball_point(center, r)) if tree_stroma else 0
                        count_necrosis = len(tree_necrosis.query_ball_point(center, r)) if tree_necrosis else 0
                        stats_within[f"stroma_{r}"].append(count_stroma)
                        stats_within[f"necrosis_{r}"].append(count_necrosis)
            
            for r in RADIUS_LIST:
                for t in ["stroma", "necrosis"]:
                    row += calculate_stats(stats_within[f"{t}_{r}"])

            # Nearest Distance
            nearest_stroma = []
            nearest_necrosis = []
            if len(coords_neoplasm) > 0:
                if tree_stroma:
                    dists, _ = tree_stroma.query(coords_neoplasm, k=1)
                    nearest_stroma = dists
                if tree_necrosis:
                    dists, _ = tree_necrosis.query(coords_neoplasm, k=1)
                    nearest_necrosis = dists
            
            row += calculate_stats(nearest_stroma)
            row += calculate_stats(nearest_necrosis)

            # --- Analysis 2: Interactions between Main Types and Immune Cells ---
            main_types_data = [
                ("neoplasm", coords_neoplasm),
                ("stroma", coords_stroma),
                ("necrosis", coords_necrosis)
            ]

            for main_name, coords_main in main_types_data:
                for immune in IMMUNE_TYPES:
                    tree_target = trees_immune[immune]
                    
                    # Within Radius
                    stats = {r: [] for r in RADIUS_LIST}
                    if len(coords_main) > 0:
                        for center in coords_main:
                            for r in RADIUS_LIST:
                                count = len(tree_target.query_ball_point(center, r)) if tree_target else 0
                                stats[r].append(count)
                    
                    for r in RADIUS_LIST:
                        row += calculate_stats(stats[r])

                    # Nearest Neighbor
                    dists = []
                    if len(coords_main) > 0 and tree_target:
                        dists, _ = tree_target.query(coords_main, k=1)
                    
                    row += calculate_stats(dists)

            # --- Analysis 3: Immune-Immune Interactions ---
            # Define specific pairs (including macrophages)
            immune_pairs = [
                ("CD8", "FOXP3"), ("CD8", "CD20"),
                ("FOXP3", "CD8"), ("FOXP3", "CD20"),
                ("CD20", "CD8"), ("CD20", "FOXP3"),
                ("CD3", "CD20"), ("CD20", "CD3"),
                ("CD68", "CD163"), ("CD163", "CD68")
            ]

            for main, target in immune_pairs:
                coords_main = coords_immune[main]
                tree_target = trees_immune[target]

                # Within Radius
                stats = {r: [] for r in RADIUS_LIST}
                if len(coords_main) > 0:
                    for center in coords_main:
                        for r in RADIUS_LIST:
                            count = len(tree_target.query_ball_point(center, r)) if tree_target else 0
                            stats[r].append(count)
                
                for r in RADIUS_LIST:
                    row += calculate_stats(stats[r])

                # Nearest Neighbor
                dists = []
                if len(coords_main) > 0 and tree_target:
                    dists, _ = tree_target.query(coords_main, k=1)
                
                row += calculate_stats(dists)

            # Write row
            writer.writerow(row)
            f_txt.write(fname + '\n')

    print(f"Done. Output saved to {output_csv}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract spatial features from cellular data (HoverNet format JSONs).")
    parser.add_argument("-i", "--input_dir", required=True, help="Path to the directory containing JSON files.")
    parser.add_argument("-o", "--output_dir", required=True, help="Path to the output directory.")
    args = parser.parse_args()

    process_files(args.input_dir, args.output_dir)
