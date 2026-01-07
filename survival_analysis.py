import numpy as np
import pandas as pd
import joblib
import os
import optuna
import argparse
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold, SelectFromModel
from sklearn.base import BaseEstimator, TransformerMixin
from sksurv.linear_model import CoxnetSurvivalAnalysis
from sksurv.util import Surv
from sksurv.ensemble import GradientBoostingSurvivalAnalysis, RandomSurvivalForest
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.statistics import logrank_test
from sklearn.utils import resample
from sklearn.impute import SimpleImputer
import sklearn
import matplotlib.pyplot as plt

# Sklearn config
sklearn.set_config(transform_output="pandas")

# --- Custom Transformers ---

class CorrelationFilter(BaseEstimator, TransformerMixin):
    def __init__(self, threshold=0.95):
        self.threshold = threshold
        self.to_drop_ = []
        self.feature_names_in_ = []

    def fit(self, X, y=None):
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

class RSFSelector(BaseEstimator, TransformerMixin):
    """
    Feature selector based on Random Survival Forest variable importance.
    Supports Bootstrap aggregation for stability.
    """
    def __init__(self, n_estimators=100, n_features=10, random_state=42, 
                 max_features=None, n_bootstrap=20, subsample=0.8):
        self.n_estimators = n_estimators
        self.n_features = n_features
        self.random_state = random_state
        self.max_features = max_features
        self.n_bootstrap = n_bootstrap
        self.subsample = subsample
        self.selector_ = None
        self.selected_features_ = []
        self.feature_names_in_ = None
        self.avg_importances_ = None

    def fit(self, X, y):
        if not isinstance(X, pd.DataFrame): X = pd.DataFrame(X)
        self.feature_names_in_ = X.columns.tolist()
        n_samples = X.shape[0]
        n_subsamples = int(n_samples * self.subsample)
        
        accumulated_importances = np.zeros(X.shape[1])
        
        # 1. Bootstrap importance calculation
        if self.n_bootstrap > 1:
            print(f"  [RSFSelector] Running Bootstrap Feature Selection ({self.n_bootstrap} iters)...")
            for i in range(self.n_bootstrap):
                X_res, y_res = resample(
                    X, y, 
                    n_samples=n_subsamples, 
                    random_state=self.random_state + i,
                    replace=True
                )
                
                gbs = GradientBoostingSurvivalAnalysis(
                    n_estimators=self.n_estimators, 
                    random_state=self.random_state + i,
                    max_features=self.max_features
                )
                gbs.fit(X_res, y_res)
                accumulated_importances += gbs.feature_importances_
            
            self.avg_importances_ = accumulated_importances / self.n_bootstrap
            
        else:
            gbs = GradientBoostingSurvivalAnalysis(
                n_estimators=self.n_estimators, 
                random_state=self.random_state,
                max_features=self.max_features
            )
            gbs.fit(X, y)
            self.avg_importances_ = gbs.feature_importances_

        # 2. Select top N features
        indices = np.argsort(self.avg_importances_)[::-1]
        top_n = min(self.n_features, len(indices))
        self.selected_features_ = [self.feature_names_in_[i] for i in indices[:top_n]]

        # 3. Fallback if empty (variance based)
        if len(self.selected_features_) == 0:
            variances = X.var()
            self.selected_features_ = variances.nlargest(5).index.tolist()
            
        return self

    def transform(self, X):
        if not isinstance(X, pd.DataFrame): X = pd.DataFrame(X, columns=self.feature_names_in_)
        return X[self.selected_features_]

    def get_feature_names_out(self, input_features=None):
        return self.selected_features_

# --- Helper Functions ---

def load_data(path, center_col):
    df = pd.read_csv(path)
    if center_col in df.columns:
        df[center_col] = df[center_col].astype(int)
    return df

def get_stratified_split(df, time_col, event_col, center_col, test_size=0.3, random_state=42):
    """
    Stratified split based on Center, Event and Time bins.
    """
    temp_df = df.copy()
    
    # Time binning for stratification
    try:
        temp_df['time_bin'] = pd.qcut(temp_df[time_col], q=4, labels=False, duplicates='drop')
    except ValueError:
        temp_df['time_bin'] = pd.qcut(temp_df[time_col], q=2, labels=False, duplicates='drop')

    temp_df['stratify_key'] = (
        temp_df[center_col].astype(str) + "_" +
        temp_df[event_col].astype(str) + "_" +
        temp_df['time_bin'].astype(str)
    )
    
    # Function to check min samples
    min_samples = temp_df['stratify_key'].value_counts().min()
    if min_samples < 2:
        print("  Warning: Full stratification failed. Fallback to Center+Event.")
        temp_df['stratify_key'] = (
            temp_df[center_col].astype(str) + "_" +
            temp_df[event_col].astype(str)
        )
        min_samples = temp_df['stratify_key'].value_counts().min()
        if min_samples < 2:
            print("  Warning: Center+Event stratification failed. Fallback to Event only.")
            temp_df['stratify_key'] = temp_df[event_col]

    train_df, test_df = train_test_split(
        df, 
        test_size=test_size, 
        stratify=temp_df['stratify_key'], 
        random_state=random_state
    )
    return train_df, test_df

def prepare_xy(df, feature_cols, event_col, time_col):
    X = df[feature_cols]
    y = Surv.from_dataframe(event_col, time_col, df)
    return X, y

def create_pipeline(params=None):
    pipeline_steps = [
        ('imputer', SimpleImputer(strategy="median")), 
        ('scaler', StandardScaler()),
        ('variancethreshold', VarianceThreshold(threshold=0)),
        ('correlationfilter', CorrelationFilter(threshold=0.95)),
        # Feature Selector (RSF + Bootstrap)
        ('rsfselector', RSFSelector(n_estimators=100, n_features=10, n_bootstrap=20)),
        # Final Estimator (RSF)
        ('randomsurvivalforest', RandomSurvivalForest(n_estimators=100, random_state=42))
    ]
    pipe = make_pipeline(*[step for name, step in pipeline_steps])
    if params:
        pipe.set_params(**params)
    return pipe

def run_optuna_optimization(X, y, n_trials=30):
    def objective(trial):
        params = {
            'rsfselector__n_estimators': trial.suggest_int('rsfselector__n_estimators', 10, 50, step=10),
            'rsfselector__n_features': trial.suggest_int('rsfselector__n_features', 5, 20, step=1),
            'randomsurvivalforest__n_estimators': trial.suggest_int('randomsurvivalforest__n_estimators', 50, 150, step=25),
            'randomsurvivalforest__max_features': trial.suggest_categorical('randomsurvivalforest__max_features', ["log2", "sqrt", None]),
            'randomsurvivalforest__min_samples_leaf': trial.suggest_int('randomsurvivalforest__min_samples_leaf', 10, 50, step=5),
            'randomsurvivalforest__max_depth': trial.suggest_int('randomsurvivalforest__max_depth', 3, 8),
        }
        pipe = create_pipeline(params)
        score = cross_val_score(pipe, X, y, cv=5, n_jobs=1).mean()
        return score

    optuna.logging.set_verbosity(optuna.logging.INFO)
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials, n_jobs=1)
    return study.best_params, study.best_value

def evaluate_model(model, X, y, dataset_name):
    from sksurv.metrics import concordance_index_censored
    try:
        prediction = model.predict(X)
        c_index = concordance_index_censored(y['event'], y['survival_months'], prediction)[0]
        print(f"  [{dataset_name}] C-index: {c_index:.3f}")
        return prediction, c_index
    except Exception as e:
        print(f"Error evaluating {dataset_name}: {e}")
        return np.zeros(len(X)), 0.5

def plot_km_curve(time, event, risk_score, title, save_path=None, cutoff=None):
    if cutoff is None:
        cutoff = np.median(risk_score)
        
    group = risk_score > cutoff
    
    kmf_high = KaplanMeierFitter()
    kmf_low = KaplanMeierFitter()
    
    plt.figure(figsize=(6, 5))
    try:
        kmf_high.fit(time[group], event[group], label='High Risk')
        ax = kmf_high.plot_survival_function(ci_show=True, color='red')
        
        kmf_low.fit(time[~group], event[~group], label='Low Risk')
        kmf_low.plot_survival_function(ax=ax, ci_show=True, color='blue')
        
        results = logrank_test(time[group], time[~group], event_observed_A=event[group], event_observed_B=event[~group])
        p_value = results.p_value
        
        plt.title(f"{title}\nP-value = {p_value:.4e}")
        plt.xlabel("Time (Months)")
        plt.ylabel("Survival Probability")
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"  Saved KM curve to {save_path}")
        else:
            plt.show()
    except Exception as e:
        print(f"Error plotting KM: {e}")

# --- Main ---

def main():
    parser = argparse.ArgumentParser(description="Survival Analysis Pipeline with RSF and Feature Selection.")
    parser.add_argument("--data", required=True, help="Path to the input CSV containing features and clinical data.")
    parser.add_argument("--output_dir", required=True, help="Directory to save models and results.")
    parser.add_argument("--center_id_ext", type=int, default=4, help="Center ID to use as fixed External Validation set.")
    parser.add_argument("--trials", type=int, default=30, help="Number of Optuna trials.")
    
    # Column configuration
    parser.add_argument("--col_id", default="filename", help="Column name for Sample ID")
    parser.add_argument("--col_time", default="survival_months", help="Column name for Time to Event")
    parser.add_argument("--col_event", default="event", help="Column name for Event (0/1)")
    parser.add_argument("--col_center", default="center", help="Column name for Center ID")

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"Processing data from: {args.data}")
    print(f"External Validation Center: {args.center_id_ext}")

    full_df = load_data(args.data, args.col_center)
    
    # Identify feature columns (exclude metadata)
    exclude_cols = [args.col_id, args.col_event, args.col_time, args.col_center, "stratify_key", "time_bin"]
    feature_cols = [c for c in full_df.columns if c not in exclude_cols]

    # Split Data: Ext Val vs Remaining
    ext_val_df = full_df[full_df[args.col_center] == args.center_id_ext].copy()
    remaining_df = full_df[full_df[args.col_center] != args.center_id_ext].copy()

    # Split Remaining: Train vs Internal Test
    try:
        train_df, int_test_df = get_stratified_split(
            remaining_df, 
            time_col=args.col_time, 
            event_col=args.col_event, 
            center_col=args.col_center,
            test_size=0.3
        )
    except Exception as e:
        print(f"Stratification error: {e}. Fallback to simple split.")
        train_df, int_test_df = train_test_split(remaining_df, test_size=0.3, random_state=42)

    print(f"Shapes -> Train: {train_df.shape}, IntTest: {int_test_df.shape}, ExtVal: {ext_val_df.shape}")

    # Prepare specific X, y (Rename columns locally for sksurv if needed, but sksurv Surv.from_dataframe uses names)
    # Just need to ensure the DF passed to prepare_xy has the right columns
    X_train, y_train = prepare_xy(train_df, feature_cols, args.col_event, args.col_time)
    X_int_test, y_int_test = prepare_xy(int_test_df, feature_cols, args.col_event, args.col_time)
    X_ext_val, y_ext_val = prepare_xy(ext_val_df, feature_cols, args.col_event, args.col_time)

    # Optimization
    print("\nStarting Optuna Optimization...")
    best_params, best_cv_score = run_optuna_optimization(X_train, y_train, n_trials=args.trials)
    print(f"Best CV Score: {best_cv_score:.4f}")

    # Final Training
    print("\nTraining final model...")
    best_pipe = create_pipeline(best_params)
    best_pipe.fit(X_train, y_train)

    # Evaluation
    print("\nEvaluating...")
    train_pred, train_cindex = evaluate_model(best_pipe, X_train, y_train, "Train")
    int_test_pred, int_test_cindex = evaluate_model(best_pipe, X_int_test, y_int_test, "Internal Test")
    ext_pred, ext_val_cindex = evaluate_model(best_pipe, X_ext_val, y_ext_val, "External Val")

    print(f"  >>> Ext Val C-Index: {ext_val_cindex:.4f} <<<")

    # Save Results
    rsf_selector = best_pipe.named_steps['rsfselector']
    if rsf_selector.avg_importances_ is not None:
        full_feat_df = pd.DataFrame({
            'feature': rsf_selector.feature_names_in_,
            'importance_avg': rsf_selector.avg_importances_
        })
        selected_feat_df = full_feat_df[full_feat_df['feature'].isin(rsf_selector.selected_features_)].sort_values('importance_avg', ascending=False)
        selected_feat_df.to_csv(os.path.join(args.output_dir, 'selected_features.csv'), index=False)

    # Save predictions
    pd.DataFrame({'id': train_df[args.col_id].values, 'risk_score': train_pred}).to_csv(os.path.join(args.output_dir, 'risk_train.csv'), index=False)
    pd.DataFrame({'id': int_test_df[args.col_id].values, 'risk_score': int_test_pred}).to_csv(os.path.join(args.output_dir, 'risk_int_test.csv'), index=False)
    pd.DataFrame({'id': ext_val_df[args.col_id].values, 'risk_score': ext_pred}).to_csv(os.path.join(args.output_dir, 'risk_ext_val.csv'), index=False)

    # Save Model
    joblib.dump(best_pipe, os.path.join(args.output_dir, 'final_model.pkl'))

    # Plot KM
    plot_km_curve(
        ext_val_df[args.col_time].values,
        ext_val_df[args.col_event].values,
        ext_pred,
        f"External Validation\nC-index: {ext_val_cindex:.3f}",
        save_path=os.path.join(args.output_dir, 'km_curve_ext.png')
    )

    print("Done.")

if __name__ == "__main__":
    main()
