import numpy as np
import pandas as pd
import joblib
import sys
import os
import ast # <-- 1. 导入 ast 模块
from sksurv.metrics import concordance_index_censored, cumulative_dynamic_auc
from sksurv.util import Surv
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
import matplotlib.pyplot as plt

# 参数
model_path = r'D:\HCC_wsi\PyPathomics\SFS_GBS\final_best_models\BEST_BoostingCox_top_3.pkl'
data_path = r'M:\HCC\Cellpoint\chenzhou_combat_corrected.csv'  # 用于推理的数据集路径

# 从模型文件名解析信息, e.g., 'BEST_RSF_top_11.pkl'
model_basename_no_ext = os.path.basename(model_path).replace('.pkl', '') # -> 'BEST_RSF_top_11'
parts = model_basename_no_ext.split('_')
model_name = parts[1]      # -> 'RSF'
n_features = parts[3]      # -> '11'

# 训练脚本的根输出目录
TRAINING_OUTPUT_DIR = r'D:\HCC_wsi\PyPathomics\SFS_GBS'

# 统一输出目录，放到一个专门的推理结果文件夹下
BASE_OUTPUT_DIR = os.path.join(TRAINING_OUTPUT_DIR, 'inferences', f'infer_{model_basename_no_ext}_on_chenzhou')
os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)

# 自动找到对应的特征列表文件
# --- 修正：不再依赖独立的特征重要性文件，而是从训练摘要中获取 ---
training_summary_path = os.path.join(TRAINING_OUTPUT_DIR, r'best_model_per_type_summary.csv')
# --- 修正结束 ---

output_path = os.path.join(BASE_OUTPUT_DIR, 'result.csv')
km_curve_path = os.path.join(BASE_OUTPUT_DIR, 'km_curve.png')
summary_metrics_path = os.path.join(BASE_OUTPUT_DIR, 'summary_metrics.csv')

# 1. 加载模型
print(f"正在加载模型: {model_path}")
model = joblib.load(model_path)

# 2. 读取新数据
print(f"正在读取数据: {data_path}")
df = pd.read_csv(data_path)

# 3. 准备输入数据：必须使用与训练时完全相同的特征子集
# --- 修改：从训练摘要文件加载模型使用的特征列 ---
try:
    print(f"正在加载训练摘要: {training_summary_path}")
    df_summary = pd.read_csv(training_summary_path)
    # model_name 是从文件名解析出来的，例如 'RSF'
    model_info = df_summary[df_summary['model'] == model_name]
    if model_info.empty:
        raise ValueError(f"在摘要文件 {training_summary_path} 中找不到模型 '{model_name}' 的信息。")
    
    # 获取特征列表字符串并安全地转换为Python列表
    features_str = model_info['selected_features'].iloc[0]
    used_feature_cols = ast.literal_eval(features_str)

except FileNotFoundError:
    raise FileNotFoundError(f"必需的训练摘要文件未找到: {training_summary_path}。\n请确保该文件存在于训练输出目录中。")
except Exception as e:
    raise RuntimeError(f"从摘要文件解析特征列表时出错: {e}")

print(f"模型使用了 {len(used_feature_cols)} 个特征: {used_feature_cols[:5]}...")

# 确保新数据中包含所有需要的特征
missing_cols = set(used_feature_cols) - set(df.columns)
if missing_cols:
    raise ValueError(f"新数据中缺少以下必需的特征列: {missing_cols}")

# 使用加载的、顺序固定的特征列表来准备 X_new
X_new = df[used_feature_cols].values
# --- 结束修改 ---

# 4. 推理
print("开始推理...")
# 加载的model是一个包含标准化和预测器的pipeline
if hasattr(model, "predict"):
    risk_score = model.predict(X_new)
    df['risk_score'] = risk_score
else:
    raise AttributeError("加载的模型没有 'predict' 方法。")

# 对于支持生存函数的模型（如Cox, GBS, RSF），计算生存概率
if hasattr(model, "predict_survival_function"):
    surv_funcs = model.predict_survival_function(X_new)
    # 输出每个样本在12、24、36个月的生存概率
    df['survival_prob_12m'] = [fn(12) for fn in surv_funcs]
    df['survival_prob_24m'] = [fn(24) for fn in surv_funcs]
    df['survival_prob_36m'] = [fn(36) for fn in surv_funcs]
    print("已计算风险评分和生存概率。")
else:
    df['survival_prob_12m'] = np.nan
    df['survival_prob_24m'] = np.nan
    df['survival_prob_36m'] = np.nan
    print("模型不支持 predict_survival_function，仅计算风险评分。")


# 基于risk_score中位数分层
median_score = df['risk_score'].median()
df['risk_group'] = df['risk_score'].apply(lambda x: 'High' if x > median_score else 'Low')

# 绘制KM曲线
print("正在生成KM曲线...")
kmf = KaplanMeierFitter()
plt.figure(figsize=(8, 6))
for name, grouped in df.groupby('risk_group'):
    kmf.fit(grouped['survival_months'], grouped['event'], label=f'{name} Risk (n={len(grouped)})')
    kmf.plot_survival_function()
plt.title(f'KM Curve by Risk Group ({model_name} Top {n_features})')
plt.xlabel('Survival Months')
plt.ylabel('Survival Probability')
plt.legend()
plt.tight_layout()
plt.savefig(km_curve_path, dpi=300)
plt.close()

# logrank检验
high = df[df['risk_group'] == 'High']
low = df[df['risk_group'] == 'Low']
results = logrank_test(
    high['survival_months'], low['survival_months'],
    event_observed_A=high['event'], event_observed_B=low['event']
)
logrank_p = results.p_value

print(f"KM曲线已保存至 {km_curve_path}，logrank检验p值为: {logrank_p:.4f}")

# ====== 校准曲线绘制（如果支持） ======
if 'surv_funcs' in locals():
    print("正在生成校准曲线...")
    calib_times = [12, 24, 36]
    n_bins = 3

    plt.figure(figsize=(7,7))
    for calib_time in calib_times:
        try:
            # 获取每个样本在该时间点的生存概率
            df[f'pred_prob_{calib_time}m'] = [fn(calib_time) for fn in surv_funcs]
            # 分箱
            df['prob_bin'] = pd.qcut(df[f'pred_prob_{calib_time}m'], q=n_bins, duplicates='drop', labels=False)
            
            bin_means = df.groupby('prob_bin')[f'pred_prob_{calib_time}m'].mean()
            
            bin_actuals = []
            for bin_label, group in df.groupby('prob_bin'):
                kmf_calib = KaplanMeierFitter()
                kmf_calib.fit(group['survival_months'], group['event'])
                # 预测在calib_time的生存概率，如果时间超出观察范围，则取最后一个观察点的生存率
                if calib_time <= group['survival_months'].max():
                    actual_prob = kmf_calib.predict(calib_time)
                else:
                    actual_prob = kmf_calib.survival_function_.iloc[-1, 0]
                bin_actuals.append(actual_prob)
            
            plt.plot(bin_means, bin_actuals, marker='o', linestyle='-', label=f'Calibration {calib_time}m')
        except Exception as e:
            print(f"无法为 {calib_time}m 生成校准曲线: {e}")
            continue

    plt.plot([0,1], [0,1], 'k--', label='Perfect Calibration')
    plt.xlabel('Predicted Survival Probability')
    plt.ylabel('Observed Survival Probability (Kaplan-Meier)')
    plt.title('Calibration Curve at Multiple Time Points')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(BASE_OUTPUT_DIR, 'calibration_curve_multi_time.png'), dpi=300)
    plt.close()
    print(f"多时间点校准曲线已保存至 {os.path.join(BASE_OUTPUT_DIR, 'calibration_curve_multi_time.png')}")


# 只保留指定列
keep_cols = ['case_id', 'survival_months', 'event', 'risk_score', 'risk_group', 
             'survival_prob_12m', 'survival_prob_24m', 'survival_prob_36m']
# 确保slide列存在时才保留
if 'slide' in df.columns:
    keep_cols.insert(1, 'slide')
df_to_save = df[[col for col in keep_cols if col in df.columns]]


# 5. 保存结果
df_to_save.to_csv(output_path, index=False, encoding='utf-8-sig')
print(f"推理完成，结果已保存至 {output_path}")

# 构造生存对象
y_true = Surv.from_dataframe('event', 'survival_months', df)

# 计算c-index
cindex = concordance_index_censored(df['event'].astype(bool), df['survival_months'], df['risk_score'])[0]

# 计算时间依赖AUC
auc_scores = []
mean_auc = np.nan
if 'surv_funcs' in locals():
    times = np.array([6, 12, 18, 24, 30, 36])
    # 1-生存概率即为风险
    risk_scores_at_times = np.row_stack([1 - fn(times) for fn in surv_funcs])
    
    try:
        auc_scores_raw, mean_auc_raw = cumulative_dynamic_auc(y_true, y_true, risk_scores_at_times, times)
        auc_scores = list(auc_scores_raw)
        mean_auc = mean_auc_raw
    except Exception as e:
        print(f"计算AUC失败: {e}")
        # 如果计算失败，用NaN填充所有预期的AUC值
        auc_scores = [np.nan] * len(times)

# 保存到csv
summary_data = {
    'metric': ['c-index', 'logrank_p', 'risk_score_median'],
    'value': [cindex, logrank_p, median_score]
}
if auc_scores:
    # 确保metric和value列表长度一致
    auc_metrics = [f'AUC@{t}' for t in times] + ['mean_AUC']
    auc_values = auc_scores + [mean_auc]
    
    summary_data['metric'].extend(auc_metrics)
    summary_data['value'].extend(auc_values)

summary = pd.DataFrame(summary_data)
summary.to_csv(summary_metrics_path, index=False, encoding='utf-8-sig')
print(f"汇总性参数已保存至 {summary_metrics_path}")
print("\n--- 汇总指标 ---")
print(summary)