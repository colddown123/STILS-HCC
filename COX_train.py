import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sksurv.linear_model import CoxnetSurvivalAnalysis
from sksurv.ensemble import RandomSurvivalForest, GradientBoostingSurvivalAnalysis
from sksurv.svm import FastSurvivalSVM
from sksurv.util import Surv
from sklearn.inspection import permutation_importance
import joblib
import os
import ast
import random  # 导入random模块

# --- 1. 数据和预计算的特征排名加载 ---

# 假设您有一个名为 'XXX_ranking.csv' 的文件，其中包含一列 'feature'，按重要性降序排列
# 请将 'XXX_ranking.csv' 替换为您的实际文件名
ranking_file_path = r'D:\HCC_wsi\PyPathomics\预训练3\RSF_feature_importance_ranking_permutation.csv' # 示例：使用之前运行得到的RSF重要性排序
df_ranking = pd.read_csv(ranking_file_path)
# --- 修改：定义一个固定的顶层特征池 ---
# 我们将从这个池中进行随机抽样
TOP_N_POOL = 15 
feature_pool = df_ranking['feature'].tolist()[:TOP_N_POOL]
print(f"特征抽样池 (Top {TOP_N_POOL}): {feature_pool}")

# 训练集数据读取
df = pd.read_csv(r'M:\HCC\Cellpoint\train_combat_corrected.csv')
time_col = 'survival_months'
event_col = 'event'
y = Surv.from_dataframe(event=event_col, time=time_col, data=df)

# 外部测试集读取
df_ext = pd.read_csv(r'M:\HCC\Cellpoint\zhu_combat_corrected.csv')
y_ext = Surv.from_dataframe(event=event_col, time=time_col, data=df_ext)

# --- 2. 定义模型、参数和输出目录 ---

# 定义不含特征选择器的pipeline
pipe_rsf = make_pipeline(
    StandardScaler(),
    RandomSurvivalForest(n_jobs=-1, random_state=0)
)
pipe_gb = make_pipeline(
    StandardScaler(),
    GradientBoostingSurvivalAnalysis(random_state=0, n_iter_no_change=10)
)

# 定义不含特征选择器参数的参数网格
param_rsf = {
    'randomsurvivalforest__n_estimators': [1, 2, 3, 4, 5, 6],#, 3, 4, 5, 6, 7, 8
    'randomsurvivalforest__min_samples_split': [30, 40, 50, 60]# , 40, 50, 60 
}
param_gb = {
    'gradientboostingsurvivalanalysis__n_estimators': [1, 2, 3, 4, 5],#, 8
    'gradientboostingsurvivalanalysis__learning_rate': [0.01, 0.03, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5],
    'gradientboostingsurvivalanalysis__max_depth': [2, 3, 4]
}


# 要遍历的特征数量
N_FEATURES_OPTIONS = [1, 2, 3, 4, 5, 6, 7, 8]



# 创建输出目录
output_dir = r'D:/HCC_wsi/PyPathomics/SFS_RSF'

# --- 3. 主循环：遍历特征数量，执行GridSearchCV ---

# 用于存储每个模型在不同特征数量下的最佳结果
all_models_summary = []

# SFS 使用的代理模型，选择一个较快的
sfs_estimator = GridSearchCV(pipe_gb, param_gb, cv=5, n_jobs=-1)

# SFS 过程中的变量
selected_features = []
remaining_features = feature_pool.copy()
sfs_log = [] # 记录SFS过程

print(f"\n{'='*20} 开始序列前向选择 (SFS) {'='*20}")

for n_features in N_FEATURES_OPTIONS:
    print(f"\n--- 正在为 {n_features} 个特征寻找最佳组合 ---")
    
    best_score_for_this_n = -1
    best_feature_to_add = None
    
    # a. SFS核心：找到下一个最佳特征
    for feature in remaining_features:
        current_features = selected_features + [feature]
        current_features.sort()
        
        X_subset = df[current_features].values
        
        try:
            sfs_estimator.fit(X_subset, y)
            current_score = sfs_estimator.best_score_
            print(f"  - 测试组合: {current_features}, CV C-index: {current_score:.4f}")

            if current_score > best_score_for_this_n:
                best_score_for_this_n = current_score
                best_feature_to_add = feature
        except Exception as e:
            print(f"!!! SFS 步骤失败，组合: {current_features}, 错误: {e}")

    if best_feature_to_add is None:
        print(f"!!! 无法为 {n_features} 个特征找到有效的改进特征，SFS终止。")
        break

    # b. 更新特征集
    selected_features.append(best_feature_to_add)
    selected_features.sort()
    remaining_features.remove(best_feature_to_add)
    
    print(f"\n>>> {n_features} 特征下的最佳组合已确定: {selected_features} (CV C-index: {best_score_for_this_n:.4f})")
    sfs_log.append({'n_features': n_features, 'selected_features': selected_features.copy(), 'sfs_cv_cindex': best_score_for_this_n})

    # c. 使用这个确定的最佳特征集来训练所有模型
    print("--- 使用此最佳组合训练所有模型 ---")
    X_subset = df[selected_features].values
    X_ext_subset = df_ext[selected_features].values

    searches = {
        #'BoostingCox': GridSearchCV(pipe_gb, param_gb, cv=5, n_jobs=-1),
        'RSF': GridSearchCV(pipe_rsf, param_rsf, cv=5, n_jobs=-1),
        #'SVM': GridSearchCV(pipe_gb, param_gb, cv=5, n_jobs=-1),
    }

    for name, search in searches.items():
        try:
            print(f"  - 训练模型: {name}")
            search.fit(X_subset, y)

            train_cindex = search.score(X_subset, y)
            test_cindex = search.score(X_ext_subset, y_ext)

            result_dict = {
                'model': name,
                'n_features': n_features,
                'best_cv_cindex': search.best_score_,
                'train_cindex': train_cindex,
                'test_cindex': test_cindex,
                'best_params': search.best_params_,
                'selected_features': selected_features.copy(),
            }
            all_models_summary.append(result_dict)

            # 保存当前模型
            n_features_dir = os.path.join(output_dir, f'top_{n_features}_features')
            os.makedirs(n_features_dir, exist_ok=True)
            joblib.dump(search.best_estimator_, os.path.join(n_features_dir, f'best_model_{name}.pkl'))

        except Exception as e:
            print(f"!!! {name} 模型训练失败，特征集: {selected_features}, 错误: {e}")

# 保存SFS过程日志
pd.DataFrame(sfs_log).to_csv(os.path.join(output_dir, 'sfs_selection_log.csv'))

# --- 4. 结果汇总与保存 ---

# 将汇总结果转换为DataFrame
df_summary = pd.DataFrame(all_models_summary)

# 找到每个模型的最佳表现（基于交叉验证C-index）
if not df_summary.empty:
    best_results = df_summary.loc[df_summary.groupby('model')['best_cv_cindex'].idxmax()]
    best_results = best_results.set_index('model').sort_values('best_cv_cindex', ascending=False)

    print("\n\n--- Overall Best Performers (based on CV C-index) ---")
    pd.set_option('display.max_colwidth', 200)
    print(best_results)

    # 保存详细的每次运行结果和最终的最佳结果
    df_summary.to_csv(os.path.join(output_dir, 'full_cv_summary.csv'), index=False, encoding='utf-8-sig')
    best_results.to_csv(os.path.join(output_dir, 'best_model_per_type_summary.csv'), encoding='utf-8-sig')

    # --- 5. 保存最佳模型的特征系数/重要性并归档最佳模型 ---
    print("\n--- Saving feature importances and archiving final best models ---")
    
    # 创建一个专门存放最终模型的目录
    final_models_dir = os.path.join(output_dir, 'final_best_models')
    os.makedirs(final_models_dir, exist_ok=True)

    for model_name, row in best_results.iterrows():
        n_features = row['n_features']
        features_obj = row['selected_features']
        
        # --- 修正：健壮地处理特征列表（无论是字符串还是列表对象）---
        features = None
        if isinstance(features_obj, str):
            try:
                features = ast.literal_eval(features_obj)
            except (ValueError, SyntaxError):
                print(f"警告：无法解析特征列表字符串 '{features_obj}'。跳过此模型。")
                continue
        elif isinstance(features_obj, list):
            features = features_obj # 它已经是列表了，直接使用
        
        if not isinstance(features, list):
            print(f"警告：未能从 '{features_obj}' 获取有效的特征列表。跳过此模型。")
            continue
        # --- 修正结束 ---

        # 加载对应的最佳模型
        model_path = os.path.join(output_dir, f'top_{n_features}_features', f'best_model_{model_name}.pkl')
        if not os.path.exists(model_path):
            print(f"Model file not found for {model_name}, skipping.")
            continue
        
        # 将此最佳模型文件复制到最终目录
        final_model_path = os.path.join(final_models_dir, f'BEST_{model_name}_top_{n_features}.pkl')
        joblib.dump(joblib.load(model_path), final_model_path)
        print(f"Archived best model for {model_name} to {final_model_path}")

        model = joblib.load(model_path)

        df_out = None
        if model_name == 'BoostingCox':
            importances = model.named_steps['gradientboostingsurvivalanalysis'].feature_importances_
            df_out = pd.DataFrame(list(zip(features, importances)), columns=['feature', 'importance'])
            df_out = df_out.sort_values('importance', ascending=False)
            
        elif model_name == 'RSF':
            # 对于RSF，使用置换重要性来计算
            print(f"Calculating permutation importance for RSF with {n_features} features...")
            # 需要使用与训练时相同的数据子集
            X_subset_best = df[features].values
            result = permutation_importance(model, X_subset_best, y, n_repeats=15, random_state=0, n_jobs=-1)
            importances = result.importances_mean
            df_out = pd.DataFrame(list(zip(features, importances)), columns=['feature', 'importance'])
            df_out = df_out.sort_values('importance', ascending=False)

        # 保存文件
        if df_out is not None:
            output_filename = os.path.join(output_dir, f'BEST_{model_name}_top_{n_features}_features_importance.csv')
            df_out.to_csv(output_filename, index=False)
            print(f"Saved importance for {model_name} to {output_filename}")
else:
    print("\nNo models were successfully trained. Summary cannot be generated.")

print("\n处理完成，结果已保存至:", output_dir)