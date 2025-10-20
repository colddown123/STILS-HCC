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
from lifelines import CoxPHFitter
import scipy.stats as stats

def filter_features(df, features, threshold=0.60):
    filtered = []
    for feat in features:
        vals = df[feat]
        zero_or_na = ((vals == 0) | (vals.isna())).mean()
        if zero_or_na <= threshold:
            filtered.append(feat)
    return filtered

def fillna_median(df, features):
    for feat in features:
        median = df[feat].replace(0, np.nan).median()
        df[feat] = df[feat].replace(0, np.nan).fillna(median)
    return df

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

def apply_combat_to_external(external_csv, combat_model_pkl, output_csv, filtered_features, train_batches, output_dir):
    df_ext = pd.read_csv(external_csv)
    df_ext = df_ext.rename(columns={'slide': 'slide', 'event': 'event', 'survival_months': 'survival_months'})
    df_ext = fillna_median(df_ext, filtered_features)
    df_ext['batch'] = pd.Categorical(df_ext['batch'], categories=train_batches)
    missing_batches = set(train_batches) - set(df_ext['batch'].dropna().unique())
    if missing_batches:
        dummy = df_ext.iloc[0:1].copy()
        for b in missing_batches:
            dummy['batch'] = b
            df_ext = pd.concat([df_ext, dummy], ignore_index=True)
        df_ext['batch'] = pd.Categorical(df_ext['batch'], categories=train_batches)
    batch = df_ext['batch'].cat.codes
    covars = df_ext[['survival_months', 'event']].values
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

def univariate_cox(df, features, time_col='survival_months', event_col='event', out_csv=None):
    results = []
    for feat in features:
        cph = CoxPHFitter()
        data = df[[time_col, event_col, feat]].copy()
        # 若特征全为常数或有缺失，跳过
        if data[feat].nunique() < 2 or data[feat].isnull().any():
            continue
        try:
            cph.fit(data, duration_col=time_col, event_col=event_col)
            summary = cph.summary.loc[feat]
            results.append({
                'feature': feat,
                'p': summary['p'],
                'HR': summary['exp(coef)']
            })
        except Exception as e:
            continue
    res_df = pd.DataFrame(results)
    if out_csv:
        res_df.to_csv(out_csv, index=False)
    return res_df

def univariate_binary_test(df, features, time_col='survival_months', event_col='event', out_csv=None):
    # 构造二分类目标
    df_bin = df.copy()
    # 仅保留24月内有事件和24月后或未发生事件的样本
    df_bin = df_bin[(df_bin[time_col] >= 24) | ((df_bin[time_col] < 24) & (df_bin[event_col] == 1))]
    df_bin['target_24m'] = np.where((df_bin[time_col] < 24) & (df_bin[event_col] == 1), 1, 0)
    results = []
    for feat in features:
        x1 = df_bin[df_bin['target_24m'] == 1][feat]
        x0 = df_bin[df_bin['target_24m'] == 0][feat]
        # 跳过全常数或缺失
        if x1.nunique() < 2 and x0.nunique() < 2:
            continue
        try:
            stat, p = stats.mannwhitneyu(x1, x0, alternative='two-sided')
            results.append({
                'feature': feat,
                'p': p,
                'median_1': x1.median(),
                'median_0': x0.median(),
                'n_1': len(x1),
                'n_0': len(x0)
            })
        except Exception as e:
            continue
    res_df = pd.DataFrame(results)
    if out_csv:
        res_df.to_csv(out_csv, index=False)
    return res_df

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default=r'M:\HCC\Cellpoint\train_batch.csv')
    parser.add_argument('--output_dir', type=str, default=r'M:\HCC\Cellpoint\hua\batch_effect')
    parser.add_argument('--external', type=str, default=r'M:\HCC\Cellpoint\chenzhou_batch.csv', help='外部数据csv路径')
    parser.add_argument('--apply_combat', action='store_true', help='是否对外部数据应用combat参数')
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # 训练集处理
    df = pd.read_csv(args.input)
    df = df.rename(columns={'slide': 'slide', 'event': 'event', 'survival_months': 'survival_months'})
    features = [col for col in df.columns if col not in ['case_id', 'slide', 'event', 'survival_months', 'batch']]

    # 1. 特征过滤
    filtered_features = filter_features(df, features)
    # 保存filtered_features以便外部数据复用
    filtered_features_path = os.path.join(args.output_dir, 'filtered_features.txt')
    with open(filtered_features_path, 'w') as f:
        for feat in filtered_features:
            f.write(feat+'\n')
    df = fillna_median(df, filtered_features)

    # 2. 校正前PCA可视化
    plot_pca(
        df[filtered_features].values,
        df['batch'].values,
        'PCA before ComBat',
        os.path.join(args.output_dir, 'pca_before_combat.png')
    )

    # 3. ComBat校正
    combat = Combat()
    batch = df['batch'].values
    covars = df[['survival_months', 'event']].values
    X = df[filtered_features].values
    combat.fit(Y=X, b=batch, X=covars, C=None)
    X_combat = combat.transform(Y=X, b=batch, X=covars, C=None)
    df_combat = df.copy()
    df_combat[filtered_features] = X_combat

    # 保存combat对象
    combat_model_pkl = os.path.join(args.output_dir, 'combat_model.pkl')
    with open(combat_model_pkl, 'wb') as f:
        pickle.dump(combat, f)

    # 4. 校正后PCA可视化
    plot_pca(
        df_combat[filtered_features].values,
        df_combat['batch'].values,
        'PCA after ComBat',
        os.path.join(args.output_dir, 'pca_after_combat.png')
    )

    # 5. 保存中间数据
    df.to_csv(os.path.join(args.output_dir, 'filtered_filled.csv'), index=False)
    df_combat.to_csv(os.path.join(args.output_dir, 'train_combat_corrected.csv'), index=False)
    print('批次效应可视化和校正已完成，结果已保存。')

    # 6. 训练集单因素Cox
    df_std = standardize_features(df.copy(), filtered_features)
    df_combat_std = standardize_features(df_combat.copy(), filtered_features)
    univariate_cox(df_std, filtered_features, out_csv=os.path.join(args.output_dir, 'train_raw_unicox.csv'))
    univariate_cox(df_combat_std, filtered_features, out_csv=os.path.join(args.output_dir, 'train_combat_unicox.csv'))
    # 新增：二分类统计检验
    univariate_binary_test(df_std, filtered_features, out_csv=os.path.join(args.output_dir, 'train_raw_binarytest.csv'))
    univariate_binary_test(df_combat_std, filtered_features, out_csv=os.path.join(args.output_dir, 'train_combat_binarytest.csv'))

    # 外部数据处理（如有）
    if args.external:
        # 校正外部数据（如指定apply_combat）
        if args.apply_combat:
            # 读取训练集batch类别
            train_batches = pd.unique(df['batch'])
            output_csv = os.path.join(args.output_dir, 'chenzhou_combat_corrected.csv')
            apply_combat_to_external(args.external, combat_model_pkl, output_csv, filtered_features, train_batches, args.output_dir)
            ext_combat = pd.read_csv(output_csv)
        # 读取外部数据原始
        ext_raw = pd.read_csv(args.external)
        ext_raw = ext_raw.rename(columns={'slide': 'slide', 'event': 'event', 'survival_months': 'survival_months'})
        ext_raw = fillna_median(ext_raw, filtered_features)
        # 外部数据单因素Cox
        univariate_cox(ext_raw, filtered_features, out_csv=os.path.join(args.output_dir, 'chenzhou_raw_unicox.csv'))
        if args.apply_combat:
            univariate_cox(ext_combat, filtered_features, out_csv=os.path.join(args.output_dir, 'chenzhou_combat_unicox.csv'))
        # 新增：外部数据二分类统计检验
        univariate_binary_test(ext_raw, filtered_features, out_csv=os.path.join(args.output_dir, 'chenzhou_raw_binarytest.csv'))
        if args.apply_combat:
            univariate_binary_test(ext_combat, filtered_features, out_csv=os.path.join(args.output_dir, 'chenzhou_combat_binarytest.csv'))