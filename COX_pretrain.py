import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.base import BaseEstimator, TransformerMixin
# 新增导入
from sklearn.inspection import permutation_importance
from sksurv.linear_model import CoxnetSurvivalAnalysis
from sksurv.ensemble import RandomSurvivalForest, GradientBoostingSurvivalAnalysis
from sksurv.svm import FastSurvivalSVM
from sksurv.util import Surv
import joblib
import os

# Early stopping回调
class EarlyStoppingMonitor:
    def __init__(self, window_size=10, max_iter_without_improvement=5):
        self.window_size = window_size
        self.max_iter_without_improvement = max_iter_without_improvement
        self._best_step = -1

    def __call__(self, iteration, estimator, args):
        if iteration < self.window_size:
            return False
        start = iteration - self.window_size + 1
        end = iteration + 1
        improvement = np.mean(estimator.oob_improvement_[start:end])
        if improvement > 1e-6:
            self._best_step = iteration
            return False
        diff = iteration - self._best_step
        return diff >= self.max_iter_without_improvement

# 相关性过滤自定义变换器 (已优化)
class CorrelationFilter(BaseEstimator, TransformerMixin):
    def __init__(self, threshold=0.90):
        self.threshold = threshold
        self.to_drop_ = []
        self.feature_names_in_ = []

    def fit(self, X, y=None):
        # 确保输入是DataFrame以保留列名
        df = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        self.feature_names_in_ = df.columns.tolist()
        
        corr_matrix = df.corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        self.to_drop_ = [column for column in upper.columns if any(upper[column] > self.threshold)]
        return self

    def transform(self, X):
        df = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        return df.drop(columns=self.to_drop_)

    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            input_features = self.feature_names_in_
        return [col for col in input_features if col not in self.to_drop_]

# 1. 训练集数据读取与初步处理
df = pd.read_csv(r'M:\HCC\Cellpoint\train_combat_corrected.csv')
id_cols = ['case_id', 'slide']
time_col = 'survival_months'
event_col = 'event'
drop_cols = id_cols + [time_col, event_col]
feature_cols = [c for c in df.columns if c not in drop_cols]
feature_cols = [c for c in feature_cols if df[c].nunique() > 1 and (df[c].value_counts(normalize=True).iloc[0] < 0.90)]
X = df[feature_cols]
y = Surv.from_dataframe(event=event_col, time=time_col, data=df)

# 2. 外部测试集读取与处理
df_ext = pd.read_csv(r'M:\HCC\Cellpoint\inter_combat_corrected.csv')
X_ext = df_ext[feature_cols]
y_ext = Surv.from_dataframe(event=event_col, time=time_col, data=df_ext)

# 3. pipeline定义
pipe_lasso = make_pipeline(
    VarianceThreshold(threshold=0.01),
    CorrelationFilter(threshold=0.90),
    StandardScaler(),
    CoxnetSurvivalAnalysis(l1_ratio=1.0, fit_baseline_model=True)  # 增加fit_baseline_model=True
)
pipe_rsf = make_pipeline(
    VarianceThreshold(threshold=0.01),
    CorrelationFilter(threshold=0.90),
    StandardScaler(),
    RandomSurvivalForest(n_jobs=-1, random_state=5)
)
pipe_gb = make_pipeline(
    VarianceThreshold(threshold=0.01),
    CorrelationFilter(threshold=0.90),
    StandardScaler(),
    GradientBoostingSurvivalAnalysis(random_state=42)
)
pipe_svm = make_pipeline(
    VarianceThreshold(threshold=0.01),
    CorrelationFilter(threshold=0.90),
    StandardScaler(),
    FastSurvivalSVM(random_state=0)
)

# 4. LASSO自动估算alphas
pipe_tmp = make_pipeline(StandardScaler(), CoxnetSurvivalAnalysis(l1_ratio=1.0, alpha_min_ratio=0.01, max_iter=100, fit_baseline_model=True))
pipe_tmp.fit(X, y)
estimated_alphas = pipe_tmp.named_steps["coxnetsurvivalanalysis"].alphas_
param_lasso = {'coxnetsurvivalanalysis__alphas': [[float(a)] for a in estimated_alphas]}

# 5. 其他参数网格
param_rsf = {
    'randomsurvivalforest__n_estimators': [3, 5, 10, 20, 50, 100],
    'randomsurvivalforest__min_samples_split': [15, 20, 30, 40]
}
param_gb = {
    'gradientboostingsurvivalanalysis__n_estimators': [3, 5, 10, 20, 50, 100],
    'gradientboostingsurvivalanalysis__learning_rate': [0.01, 0.05, 0.1, 0.2, 0.5],
    'gradientboostingsurvivalanalysis__max_depth': [2, 3, 4, 5],
    'gradientboostingsurvivalanalysis__dropout_rate': [0.3, 0.4, 0.5, 0.6],
    'gradientboostingsurvivalanalysis__subsample': [1.0, 0.8, 0.6, 0.5],
}
param_svm = {
    'fastsurvivalsvm__alpha': np.logspace(-4, 1, 10),
    'fastsurvivalsvm__rank_ratio': [1.0, 0],  # 新增rank_ratio搜索
}

# 6. 并行搜索
monitor = EarlyStoppingMonitor(window_size=10, max_iter_without_improvement=5)

# 指定输出目录
output_dir = r'D:/HCC_wsi/PyPathomics/预训练9_21'
os.makedirs(output_dir, exist_ok=True)

searches = {
    'LassoCox': GridSearchCV(pipe_lasso, param_lasso, cv=5, n_jobs=-1),
    'RSF': GridSearchCV(pipe_rsf, param_rsf, cv=5, n_jobs=-1),
    'BoostingCox': GridSearchCV(pipe_gb, param_gb, cv=5, n_jobs=-1),
    'SVM': GridSearchCV(pipe_svm, param_svm, cv=5, n_jobs=-1),
}

results = {}
for name, search in searches.items():
    if name == 'BoostingCox':
        search.fit(X, y, **{'gradientboostingsurvivalanalysis__monitor': monitor})
    else:
        search.fit(X, y)
    train_cindex = search.score(X, y)
    test_cindex = search.score(X_ext, y_ext)
    results[name] = {
        'best_params': search.best_params_,
        'best_cv_score': search.best_score_,
        'train_cindex': train_cindex,
        'test_cindex': test_cindex,
        'cv_results': pd.DataFrame(search.cv_results_).to_dict()  # 可选：如太大可删
    }
    # 保存最佳模型
    joblib.dump(search.best_estimator_, os.path.join(output_dir, f'best_model_{name}.pkl'))
    # 保存pipeline参数
    params = search.best_estimator_.get_params(deep=True)
    with open(os.path.join(output_dir, f'best_model_{name}_params.txt'), 'w', encoding='utf-8') as f:
        for k, v in params.items():
            f.write(f"{k}: {v}\n")


df_result = pd.DataFrame(results).T
pd.set_option('display.max_colwidth', 200)
print(df_result)

# 可选：保存为csv（不含cv_results，避免文件过大）
df_result.drop(columns=['cv_results']).to_csv(os.path.join(output_dir, 'model_compare_results.csv'), encoding='utf-8-sig')

# 保存训练用的特征名顺序
with open(os.path.join(output_dir, 'feature_cols.txt'), 'w', encoding='utf-8') as f:
    for col in feature_cols:
        f.write(col + '\n')

# ========== 修改：优化特征重要性提取逻辑 ==========
print("\nExtracting and saving feature importance rankings for best models...")

for name in searches.keys():
    print(f"Processing feature importance for {name}...")
    model_path = os.path.join(output_dir, f'best_model_{name}.pkl')
    if not os.path.exists(model_path):
        print(f"  - Model file not found for {name}, skipping.")
        continue

    model = joblib.load(model_path)
    
    # 对于RSF，使用置换重要性评估初始特征对整个pipeline的影响
    if name == 'RSF':
        # 置换重要性在整个pipeline上运行，评估的是初始特征的重要性
        print("  - Calculating permutation importance for RSF (this may take a while)...")
        result = permutation_importance(model, X, y, n_repeats=10, random_state=0, n_jobs=-1)
        df_importance = pd.DataFrame({
            'feature': X.columns, 
            'importance_mean': result.importances_mean,
            'importance_std': result.importances_std
        })
        df_importance = df_importance.sort_values('importance_mean', ascending=False)
        output_filename = f'{name}_feature_importance_ranking.csv'

    # 对于其他模型，提取其系数或内置重要性，并匹配预处理后的特征名
    else:
        # 步骤 1: 获取通过预处理步骤的最终特征名
        try:
            vt_step = model.named_steps['variancethreshold']
            cf_step = model.named_steps['correlationfilter']
            
            # 获取方差阈值过滤后的特征
            features_after_vt = X.columns[vt_step.get_support()]
            # CorrelationFilter的fit方法需要DataFrame来获取列名
            cf_step.fit(X[features_after_vt])
            # 获取相关性过滤后的最终特征
            final_feature_names = cf_step.get_feature_names_out(features_after_vt)
        except Exception as e:
            print(f"  - Error getting final feature names for {name}: {e}")
            continue

        # 步骤 2: 根据模型类型提取重要性
        if name == 'LassoCox':
            # 修改：将coefs从 (n, 1) 展平为 (n,)
            coefs = model.named_steps['coxnetsurvivalanalysis'].coef_.flatten()
            df_importance = pd.DataFrame({'feature': final_feature_names, 'coefficient': coefs})
            df_importance['abs_coefficient'] = df_importance['coefficient'].abs()
            df_importance = df_importance.sort_values('abs_coefficient', ascending=False).drop(columns=['abs_coefficient'])
            output_filename = f'{name}_feature_coef_ranking.csv'

        elif name == 'BoostingCox':
            importances = model.named_steps['gradientboostingsurvivalanalysis'].feature_importances_
            df_importance = pd.DataFrame({'feature': final_feature_names, 'importance': importances})
            df_importance = df_importance.sort_values('importance', ascending=False)
            output_filename = f'{name}_feature_importance_ranking.csv'

        elif name == 'SVM':
            # 修改：将coefs从 (n, 1) 展平为 (n,)
            coefs = model.named_steps['fastsurvivalsvm'].coef_.flatten()
            df_importance = pd.DataFrame({'feature': final_feature_names, 'coefficient': coefs})
            df_importance['abs_coefficient'] = df_importance['coefficient'].abs()
            df_importance = df_importance.sort_values('abs_coefficient', ascending=False).drop(columns=['abs_coefficient'])
            output_filename = f'{name}_feature_coef_ranking.csv'
        
        else:
            continue

    # 保存到CSV
    if 'df_importance' in locals() and not df_importance.empty:
        output_path = os.path.join(output_dir, output_filename)
        df_importance.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"  - Saved feature ranking to {output_path}")

print("\nAll processing complete.")
# =================================================================


# ========== 修正：通过交叉验证获取更稳健的特征重要性 ==========
from sklearn.model_selection import KFold
print("\nExtracting and saving robust feature importance rankings using 5-fold cross-validation...")

# 定义交叉验证策略
cv = KFold(n_splits=5, shuffle=True, random_state=42)

# 遍历每个已经过网格搜索的模型
for name, search in searches.items():
    # RSF的置换重要性计算成本高，跳过CV，在整个数据集上评估
    if name == 'RSF':
        print(f"Processing permutation importance for {name} on the full training set (skipping CV)...")
        model_path = os.path.join(output_dir, f'best_model_{name}.pkl')
        if not os.path.exists(model_path):
            print(f"  - Model file not found for {name}, skipping.")
            continue
        model = joblib.load(model_path)
        try:
            # 对于RSF，置换重要性在整个pipeline上运行，评估的是初始特征的重要性
            result = permutation_importance(model, X, y, n_repeats=10, random_state=0, n_jobs=-1)
            df_importance = pd.DataFrame({
                'feature': X.columns, 
                'importance_mean': result.importances_mean,
                'importance_std': result.importances_std
            }).sort_values('importance_mean', ascending=False)
            
            output_filename = f'{name}_feature_importance_ranking_permutation.csv'
            output_path = os.path.join(output_dir, output_filename)
            df_importance.to_csv(output_path, index=False, encoding='utf-8-sig')
            print(f"  - Saved permutation importance to {output_path}")
        except Exception as e:
            print(f"  - Failed to calculate permutation importance for {name}: {e}")
        continue

    # 对于其他模型，执行交叉验证
    print(f"Processing robust feature importance for {name}...")
    best_pipeline = search.best_estimator_
    fold_importances = []

    for fold, (train_idx, val_idx) in enumerate(cv.split(X, y)):
        print(f"  - Processing fold {fold+1}/5...")
        X_train, y_train = X.iloc[train_idx], y[train_idx]
        
        try:
            # --- 修正：手动追踪特征名通过pipeline的每一步 ---
            
            # 复制一份pipeline，避免修改原始的最佳模型
            from sklearn.base import clone
            current_pipeline = clone(best_pipeline)

            # 步骤 1: 方差过滤
            vt = current_pipeline.named_steps['variancethreshold']
            vt.fit(X_train)
            features_after_vt = X_train.columns[vt.get_support()]
            X_train_vt = X_train[features_after_vt]

            # 步骤 2: 相关性过滤
            cf = current_pipeline.named_steps['correlationfilter']
            cf.fit(X_train_vt)
            final_feature_names = cf.get_feature_names_out() # 现在这里是正确的字符串名称

            # 步骤 3: 训练整个pipeline以获取模型系数/重要性
            current_pipeline.fit(X_train, y_train)

            # 步骤 4: 根据模型类型提取重要性
            importance_values = None
            if name in ['LassoCox', 'SVM']:
                model_step_name = 'coxnetsurvivalanalysis' if name == 'LassoCox' else 'fastsurvivalsvm'
                importance_values = current_pipeline.named_steps[model_step_name].coef_.flatten()
            elif name == 'BoostingCox':
                importance_values = current_pipeline.named_steps['gradientboostingsurvivalanalysis'].feature_importances_

            # 步骤 5: 检查并创建DataFrame
            if importance_values is not None:
                if len(final_feature_names) != len(importance_values):
                    print(f"    - WARNING: Length mismatch in fold {fold+1}. Features: {len(final_feature_names)}, Importances: {len(importance_values)}. Skipping fold.")
                    continue
                
                df_fold_importance = pd.DataFrame({
                    'feature': final_feature_names, 
                    'importance': np.abs(importance_values)
                })
                fold_importances.append(df_fold_importance)

        except Exception as e:
            print(f"    - An error occurred in fold {fold+1}: {e}")
            continue

    # --- 汇总所有折的结果 ---
    if not fold_importances:
        print(f"  - No importance results generated for {name}, skipping.")
        continue

    df_all_folds = pd.concat(fold_importances)
    df_robust_importance = df_all_folds.groupby('feature')['importance'].agg(['mean', 'std']).reset_index()
    df_robust_importance = df_robust_importance.rename(columns={'mean': 'importance_mean', 'std': 'importance_std'})
    df_robust_importance = df_robust_importance.sort_values('importance_mean', ascending=False)
    
    output_filename = f'{name}_feature_importance_ranking_CV.csv'
    output_path = os.path.join(output_dir, output_filename)
    df_robust_importance.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"  - Saved robust feature ranking to {output_path}")

print("\nRobust importance extraction complete.")