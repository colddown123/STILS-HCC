import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.decomposition import PCA
from pycombat import Combat
import argparse
from matplotlib.patches import Ellipse
from scipy import stats
import pickle
#from lifelines import CoxPHFitter
import scipy.stats as stats

def filter_features_no_na(df, features):
    """只保留没有缺失值的特征"""
    return [feat for feat in features if not df[feat].isnull().any()]

def filter_zero_ratio_features(df, features, threshold=0.5):
    """去除0值比例大于threshold的特征"""
    filtered = []
    for feat in features:
        zero_count = (df[feat] == 0).sum()
        ratio = zero_count / len(df)
        if ratio <= threshold:
            filtered.append(feat)
    return filtered

def standardize_features(df, features):
    for feat in features:
        mean = df[feat].mean()
        std = df[feat].std()
        if std > 0:
            df[feat] = (df[feat] - mean) / std
        else:
            df[feat] = 0
    return df

def plot_pca(X, batch, title, outpath):
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    plt.figure(figsize=(7,6))
    colors = plt.cm.tab10.colors
    for i, b in enumerate(np.unique(batch)):
        idx = (batch == b)
        plt.scatter(X_pca[idx,0], X_pca[idx,1], label=f'batch {b}', alpha=0.7, color=colors[i % len(colors)])
        # 画椭圆
        if np.sum(idx) > 2:
            points = X_pca[idx]
            mean = points.mean(axis=0)
            cov = np.cov(points, rowvar=False)
            vals, vecs = np.linalg.eigh(cov)
            order = vals.argsort()[::-1]
            vals, vecs = vals[order], vecs[:,order]
            theta = np.degrees(np.arctan2(*vecs[:,0][::-1]))
            # 2sigma椭圆覆盖约95%数据
            width, height = 2 * 2 * np.sqrt(vals)
            ellipse = Ellipse(xy=mean, width=width, height=height, angle=theta, 
                              edgecolor=colors[i % len(colors)], fc='None', lw=2, ls='--')
            plt.gca().add_patch(ellipse)
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()

def replace_extreme_values_with_thresholds(df, features, feature_quantiles):
    """使用预先计算的阈值替换极端值"""
    df_copy = df.copy()
    for feat in features:
        if feat in feature_quantiles:
            lower, upper = feature_quantiles[feat]
            df_copy.loc[df_copy[feat] < lower, feat] = lower
            df_copy.loc[df_copy[feat] > upper, feat] = upper
    return df_copy

def apply_combat_to_external(external_csv, combat_model_pkl, output_csv, filtered_features, train_batches, output_dir, quantiles_path):
    df_ext = pd.read_csv(external_csv)
    df_ext = df_ext.rename(columns={'filename': 'filename', 'event': 'event', 'survival_months': 'survival_months'})
    
    # 确保验证集只包含训练时筛选出的特征
    all_needed_cols = filtered_features + ['batch', 'filename', 'event', 'survival_months']
    df_ext = df_ext[[col for col in all_needed_cols if col in df_ext.columns]]

    # 使用训练集的阈值处理验证集的极端值
    with open(quantiles_path, 'rb') as f:
        feature_quantiles = pickle.load(f)
    df_ext = replace_extreme_values_with_thresholds(df_ext, filtered_features, feature_quantiles)
    print("已使用训练集的分位数阈值处理验证集的极端值。")

    df_ext['batch'] = pd.Categorical(df_ext['batch'], categories=train_batches)
    missing_batches = set(train_batches) - set(df_ext['batch'].dropna().unique())
    if missing_batches:
        dummy = df_ext.iloc[0:1].copy()
        for b in missing_batches:
            dummy['batch'] = b
            df_ext = pd.concat([df_ext, dummy], ignore_index=True)
        df_ext['batch'] = pd.Categorical(df_ext['batch'], categories=train_batches)
    batch = df_ext['batch'].cat.codes
    covars = None # 不使用协变量
    X_ext = df_ext[filtered_features].values

    # 校正前PCA
    plot_pca(
        X_ext,
        df_ext['batch'].values,
        'External PCA before ComBat',
        os.path.join(output_dir, 'external_pca_before_combat.png')
    )

    # 加载 combat 对象
    with open(combat_model_pkl, 'rb') as f:
        combat = pickle.load(f)
    X_ext_combat = combat.transform(Y=X_ext, b=batch, X=covars, C=None)
    if missing_batches:
        X_ext_combat = X_ext_combat[:-len(missing_batches)]
        df_ext = df_ext.iloc[:-len(missing_batches)]
    df_ext[filtered_features] = X_ext_combat

    # 校正后PCA
    plot_pca(
        X_ext_combat,
        df_ext['batch'].values,
        'External PCA after ComBat',
        os.path.join(output_dir, 'external_pca_after_combat.png')
    )

    df_ext.to_csv(output_csv, index=False)
    print(f'外部数据批次校正完成，结果已保存到 {output_csv}')

def batch_effect_anova(df, features):
    """对每个特征做单因素方差分析，输出p值"""
    results = []
    for feat in features:
        groups = [df[df['batch'] == b][feat].dropna().values for b in df['batch'].unique()]
        if all(len(g) > 1 for g in groups):
            fval, pval = stats.f_oneway(*groups)
            results.append({'feature': feat, 'F': fval, 'p': pval})
        else:
            results.append({'feature': feat, 'F': None, 'p': None})
    return pd.DataFrame(results)

def replace_extreme_values(df, features, lower_q=0.01, upper_q=0.99):
    """将极端值替换为紧邻分位数的值"""
    df_copy = df.copy()
    for feat in features:
        lower = df_copy[feat].quantile(lower_q)
        upper = df_copy[feat].quantile(upper_q)
        df_copy.loc[df_copy[feat] < lower, feat] = lower
        df_copy.loc[df_copy[feat] > upper, feat] = upper
    return df_copy

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default=r'M:\HCC\HCC_immune_project\split_data\train_set.csv')
    parser.add_argument('--output_dir', type=str, default=r'M:\HCC\HCC_immune_project\batch_effect')
    parser.add_argument('--external', type=str, default=r'M:\HCC\HCC_immune_project\split_data\val_set.csv', help='外部数据csv路径')
    parser.add_argument('--apply_combat', action='store_true', help='是否对外部数据应用combat参数')
    parser.add_argument('--use_covars', action='store_true', help='是否在ComBat校正时使用协变量')
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # 训练集处理
    df = pd.read_csv(args.input)
    df = df.rename(columns={'filename': 'filename', 'event': 'event', 'survival_months': 'survival_months'})
    features = [col for col in df.columns if col not in ['case_id', 'filename', 'event', 'survival_months', 'batch']]

    # 1. 特征过滤
    # 1.1 只保留无缺失值特征
    filtered_features = filter_features_no_na(df, features)
    # 1.2 去除0值比例大于50%的特征
    filtered_features = filter_zero_ratio_features(df, filtered_features, threshold=0.5)
    
    # 2. 计算并保存极端值处理的阈值
    feature_quantiles = {}
    for feat in filtered_features:
        lower = df[feat].quantile(0.01)
        upper = df[feat].quantile(0.99)
        feature_quantiles[feat] = (lower, upper)
    
    quantiles_path = os.path.join(args.output_dir, 'feature_quantiles.pkl')
    with open(quantiles_path, 'wb') as f:
        pickle.dump(feature_quantiles, f)
    print(f"特征分位数阈值已保存到 {quantiles_path}")

    # 3. 对训练集应用极端值处理
    df = replace_extreme_values(df, filtered_features)
    print("已处理训练集的极端值。")

    # 保存最终使用的特征列表
    filtered_features_path = os.path.join(args.output_dir, 'filtered_features.txt')
    with open(filtered_features_path, 'w') as f:
        for feat in filtered_features:
            f.write(feat+'\n')

    # 根据过滤后的特征列表，筛选DataFrame的列
    meta_cols = ['case_id', 'filename', 'event', 'survival_months', 'batch']
    existing_meta_cols = [col for col in meta_cols if col in df.columns]
    df = df[existing_meta_cols + filtered_features]

    # 打印包含缺失值的行和列
    na_mask = df[filtered_features].isnull()
    rows_with_na = na_mask.any(axis=1)
    if rows_with_na.sum() > 0:
        print("包含缺失值的行索引：", df.index[rows_with_na].tolist())
        print("对应缺失的特征列：")
        for idx in df.index[rows_with_na]:
            na_cols = df.loc[idx, filtered_features][df.loc[idx, filtered_features].isnull()].index.tolist()
            print(f"行 {idx}: {na_cols}")
    else:
        print("无缺失值样本。")

    # 新增：批次效应显著性检验
    anova_results = batch_effect_anova(df, filtered_features)
    anova_results_path = os.path.join(args.output_dir, 'batch_effect_anova.csv')
    anova_results.to_csv(anova_results_path, index=False)
    print(f'批次效应显著性检验结果已保存到 {anova_results_path}')

    # 校正前PCA可视化
    plot_pca(
        df[filtered_features].values,
        df['batch'].values,
        'PCA before ComBat',
        os.path.join(args.output_dir, 'pca_before_combat.png')
    )

    # ComBat校正
    combat = Combat()
    batch = df['batch'].values
    covars = None # 不使用协变量
    X = df[filtered_features].values
    combat.fit(Y=X, b=batch, X=covars, C=None)
    X_combat = combat.transform(Y=X, b=batch, X=covars, C=None)
    df_combat = df.copy()
    df_combat[filtered_features] = X_combat

    # 保存combat对象
    combat_model_pkl = os.path.join(args.output_dir, 'combat_model.pkl')
    with open(combat_model_pkl, 'wb') as f:
        pickle.dump(combat, f)

    # 校正后PCA可视化
    plot_pca(
        df_combat[filtered_features].values,
        df_combat['batch'].values,
        'PCA after ComBat',
        os.path.join(args.output_dir, 'pca_after_combat.png')
    )

    # 保存中间数据
    df.to_csv(os.path.join(args.output_dir, 'filtered_filled.csv'), index=False)
    df_combat.to_csv(os.path.join(args.output_dir, 'train_combat_corrected.csv'), index=False)
    print('批次效应可视化和校正已完成，结果已保存。')

    # 外部数据处理（如有）
    if args.external:
        # 校正外部数据（如指定apply_combat）
        if args.apply_combat:
            train_batches = pd.unique(df['batch'])
            output_csv = os.path.join(args.output_dir, 'val_combat_corrected.csv')
            def get_covars(df_ext):
                return df_ext[['survival_months', 'event']].values if args.use_covars else None
            # 修改apply_combat_to_external调用
            apply_combat_to_external(args.external, combat_model_pkl, output_csv, filtered_features, train_batches, args.output_dir, quantiles_path)
            ext_combat = pd.read_csv(output_csv)
        ext_raw = pd.read_csv(args.external)
        ext_raw = ext_raw.rename(columns={'filename': 'filename', 'event': 'event', 'survival_months': 'survival_months'})
