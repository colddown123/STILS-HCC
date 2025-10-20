import json
import os
import glob
import numpy as np
import csv
from scipy.spatial import cKDTree

# 输入输出路径
path_pairs = [
    (r'M:\HCC\Cellpoint\chenzhou\HE_Tcell_json', r'M:\HCC\Cellpoint\chenzhou\HE_Tcell_json\neigh_out'),
    #(r'M:\HCC\Cellpoint\zhuhai\HE_Tcell_json', r'M:\HCC\Cellpoint\zhuhai\HE_Tcell_json\neigh_out'),
    #(r'M:\HCC\Cellpoint\hunan\HE_Tcell_json', r'M:\HCC\Cellpoint\hunan\HE_Tcell_json\neigh_out')
    # 可以继续添加更多路径对
]

type_map = {
    "1": "neopla",
    "2": "inflam",
    "3": "connec",
    "4": "necros",
    "5": "CD3",
    "6": "CD8",
    "7": "FOXp3",
    "8": "CD20"
}
radius_list = [10, 50, 100, 300, 500, 1000]

# 新增免疫细胞类型
immune_types = ["CD3", "CD8", "FOXp3", "CD20"]
immune_type_ids = {"CD3": "5", "CD8": "6", "FOXp3": "7", "CD20": "8"}

# 结果表头
header = ["filename"]
for r in radius_list:
    for t in ["connec", "necros"]:
        header += [
            f"{t}_within_{r}_mean",
            f"{t}_within_{r}_std",
            f"{t}_within_{r}_median",
            f"{t}_within_{r}_q10",
            f"{t}_within_{r}_q25",
            f"{t}_within_{r}_q75",
            f"{t}_within_{r}_q90"
        ]

# 结果表头增加最近邻距离统计
for t in ["connec", "necros"]:
    header += [
        f"nearest_{t}_mean",
        f"nearest_{t}_std",
        f"nearest_{t}_median",
        f"nearest_{t}_q10",
        f"nearest_{t}_q25",
        f"nearest_{t}_q75",
        f"nearest_{t}_q90"
    ]

# 结果表头扩展
for main_type in ["neopla", "connec", "necros"]:
    for immune in immune_types:
        for r in radius_list:
            header += [
                f"{main_type}_within_{r}_{immune}_mean",
                f"{main_type}_within_{r}_{immune}_std",
                f"{main_type}_within_{r}_{immune}_median",
                f"{main_type}_within_{r}_{immune}_q10",
                f"{main_type}_within_{r}_{immune}_q25",
                f"{main_type}_within_{r}_{immune}_q75",
                f"{main_type}_within_{r}_{immune}_q90"
            ]
        header += [
            f"{main_type}_nearest_{immune}_mean",
            f"{main_type}_nearest_{immune}_std",
            f"{main_type}_nearest_{immune}_median",
            f"{main_type}_nearest_{immune}_q10",
            f"{main_type}_nearest_{immune}_q25",
            f"{main_type}_nearest_{immune}_q75",
            f"{main_type}_nearest_{immune}_q90"
        ]

# 新增免疫细胞间空间关系表头
immune_pairs = [
    ("CD8", "FOXp3"), ("CD8", "CD20"),
    ("FOXp3", "CD8"), ("FOXp3", "CD20"),
    ("CD20", "CD8"), ("CD20", "FOXp3"),
    ("CD3", "CD20"), ("CD20", "CD3")   # 新增
]
for main, target in immune_pairs:
    for r in radius_list:
        header += [
            f"{main}_within_{r}_{target}_mean",
            f"{main}_within_{r}_{target}_std",
            f"{main}_within_{r}_{target}_median",
            f"{main}_within_{r}_{target}_q10",
            f"{main}_within_{r}_{target}_q25",
            f"{main}_within_{r}_{target}_q75",
            f"{main}_within_{r}_{target}_q90"
        ]
    header += [
        f"{main}_nearest_{target}_mean",
        f"{main}_nearest_{target}_std",
        f"{main}_nearest_{target}_median",
        f"{main}_nearest_{target}_q10",
        f"{main}_nearest_{target}_q25",
        f"{main}_nearest_{target}_q75",
        f"{main}_nearest_{target}_q90"
    ]

for json_dir, output_dir in path_pairs:
    os.makedirs(output_dir, exist_ok=True)
    json_files = glob.glob(os.path.join(json_dir, '*.json'))

    intermediate_csv = os.path.join(output_dir, 'intermediate.csv')
    processed_txt = os.path.join(output_dir, 'processed.txt')

    # 读取已处理文件名
    if os.path.exists(processed_txt):
        with open(processed_txt, 'r') as f:
            processed_files = set(line.strip() for line in f)
    else:
        processed_files = set()

    # 如果中间csv不存在，写入表头
    if not os.path.exists(intermediate_csv):
        with open(intermediate_csv, 'w', newline='', encoding='utf-8-sig') as f:
            writer = csv.writer(f)
            writer.writerow(header)

    for json_path in json_files:
        fname = os.path.basename(json_path)
        if fname in processed_files:
            continue  # 跳过已处理文件

        with open(json_path, 'r') as f:
            data = json.load(f)

        coords_neopla = []
        coords_connec = []
        coords_necros = []
        for cell in data['nuc'].values():
            t = str(cell['type'])
            c = cell['centroid']
            if t == "1":
                coords_neopla.append(c)
            elif t == "3":
                coords_connec.append(c)
            elif t == "4":
                coords_necros.append(c)

        coords_neopla = np.array(coords_neopla)
        coords_connec = np.array(coords_connec)
        coords_necros = np.array(coords_necros)

        tree_connec = cKDTree(coords_connec) if len(coords_connec) > 0 else None
        tree_necros = cKDTree(coords_necros) if len(coords_necros) > 0 else None

        # 提取免疫细胞坐标
        coords_immune = {k: [] for k in immune_types}
        for cell in data['nuc'].values():
            t = str(cell['type'])
            c = cell['centroid']
            for immune in immune_types:
                if t == immune_type_ids[immune]:
                    coords_immune[immune].append(c)

        for immune in immune_types:
            coords_immune[immune] = np.array(coords_immune[immune])

        trees_immune = {k: cKDTree(v) if len(v) > 0 else None for k, v in coords_immune.items()}

        # 最紧邻距离分析
        nearest_connec = []
        nearest_necros = []
        if len(coords_neopla) > 0:
            if tree_connec and len(coords_connec) > 0:
                dists_connec, _ = tree_connec.query(coords_neopla, k=1)
                nearest_connec = dists_connec
            else:
                nearest_connec = np.array([])
            if tree_necros and len(coords_necros) > 0:
                dists_necros, _ = tree_necros.query(coords_neopla, k=1)
                nearest_necros = dists_necros
            else:
                nearest_necros = np.array([])
        else:
            nearest_connec = np.array([])
            nearest_necros = np.array([])

        # 统计每个neopla细胞的邻域计数
        stats = {}
        for r in radius_list:
            stats[f"connec_{r}"] = []
            stats[f"necros_{r}"] = []

        for center in coords_neopla:
            for r in radius_list:
                if tree_connec:
                    count_connec = len(tree_connec.query_ball_point(center, r))
                else:
                    count_connec = 0
                if tree_necros:
                    count_necros = len(tree_necros.query_ball_point(center, r))
                else:
                    count_necros = 0
                stats[f"connec_{r}"].append(count_connec)
                stats[f"necros_{r}"].append(count_necros)

        row = [os.path.basename(json_path)]
        for r in radius_list:
            for t in ["connec", "necros"]:
                arr = np.array(stats[f"{t}_{r}"])
                if arr.size > 0:
                    row += [
                        np.mean(arr),
                        np.std(arr),
                        np.median(arr),
                        np.percentile(arr, 10),
                        np.percentile(arr, 25),
                        np.percentile(arr, 75),
                        np.percentile(arr, 90)
                    ]
                else:
                    row += [0, 0, 0, 0, 0, 0, 0]

        # 增加最近邻距离统计
        for arr in [nearest_connec, nearest_necros]:
            if len(arr) > 0:
                row += [
                    np.mean(arr),
                    np.std(arr),
                    np.median(arr),
                    np.percentile(arr, 10),
                    np.percentile(arr, 25),
                    np.percentile(arr, 75),
                    np.percentile(arr, 90)
                ]
            else:
                row += [0, 0, 0, 0, 0, 0, 0]

        # 针对每类主细胞，分别做邻域计数和最近邻距离
        for main_type, coords_main in zip(
            ["neopla", "connec", "necros"],
            [coords_neopla, coords_connec, coords_necros]
        ):
            for immune in immune_types:
                tree_immune = trees_immune[immune]
                # 邻域计数
                stats = {r: [] for r in radius_list}
                for center in coords_main:
                    for r in radius_list:
                        if tree_immune:
                            count = len(tree_immune.query_ball_point(center, r))
                        else:
                            count = 0
                        stats[r].append(count)
                for r in radius_list:
                    arr = np.array(stats[r])
                    if arr.size > 0:
                        row += [
                            np.mean(arr),
                            np.std(arr),
                            np.median(arr),
                            np.percentile(arr, 10),
                            np.percentile(arr, 25),
                            np.percentile(arr, 75),
                            np.percentile(arr, 90)
                        ]
                    else:
                        row += [0, 0, 0, 0, 0, 0, 0]
                # 最近邻距离
                if tree_immune and len(coords_main) > 0:
                    dists, _ = tree_immune.query(coords_main, k=1)
                    arr = dists
                else:
                    arr = np.array([])
                if len(arr) > 0:
                    row += [
                        np.mean(arr),
                        np.std(arr),
                        np.median(arr),
                        np.percentile(arr, 10),
                        np.percentile(arr, 25),
                        np.percentile(arr, 75),
                        np.percentile(arr, 90)
                    ]
                else:
                    row += [0, 0, 0, 0, 0, 0, 0]

        # 免疫细胞间空间关系统计
        for main, target in immune_pairs:
            coords_main = coords_immune[main]
            coords_target = coords_immune[target]
            tree_target = trees_immune[target]
            # 邻域计数
            stats = {r: [] for r in radius_list}
            for center in coords_main:
                for r in radius_list:
                    if tree_target:
                        count = len(tree_target.query_ball_point(center, r))
                    else:
                        count = 0
                    stats[r].append(count)
            for r in radius_list:
                arr = np.array(stats[r])
                if arr.size > 0:
                    row += [
                        np.mean(arr),
                        np.std(arr),
                        np.median(arr),
                        np.percentile(arr, 10),
                        np.percentile(arr, 25),
                        np.percentile(arr, 75),
                        np.percentile(arr, 90)
                    ]
                else:
                    row += [0, 0, 0, 0, 0, 0, 0]
            # 最近邻距离
            if tree_target and len(coords_main) > 0:
                dists, _ = tree_target.query(coords_main, k=1)
                arr = dists
            else:
                arr = np.array([])
            if len(arr) > 0:
                row += [
                    np.mean(arr),
                    np.std(arr),
                    np.median(arr),
                    np.percentile(arr, 10),
                    np.percentile(arr, 25),
                    np.percentile(arr, 75),
                    np.percentile(arr, 90)
                ]
            else:
                row += [0, 0, 0, 0, 0, 0, 0]

        # 追加写入中间csv
        with open(intermediate_csv, 'a', newline='', encoding='utf-8-sig') as f:
            writer = csv.writer(f)
            writer.writerow(row)

        # 记录已处理文件
        with open(processed_txt, 'a') as f:
            f.write(fname + '\n')

    # 最终整理
    csv_path = os.path.join(output_dir, 'neopla_neigh_stats.csv')
    with open(intermediate_csv, 'r', encoding='utf-8-sig') as fin, \
         open(csv_path, 'w', newline='', encoding='utf-8-sig') as fout:
        lines = fin.readlines()
        fout.writelines(lines)