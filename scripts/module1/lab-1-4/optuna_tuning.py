import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
import lightgbm as lgb
import optuna
import time
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import roc_auc_score, average_precision_score
import warnings
warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)
from pathlib import Path
# Project root
ROOT = Path(__file__).resolve().parents[3]


DATA_DIR = ROOT / "data"

# Output folder
OUTPUT_DIR = ROOT / "output/Lab1"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
 
print("=" * 60)
print("  MODULE 1 | LAB 1.4")
print("  LightGBM + Optuna Tuning")
print("  Pareto Tradeoff: AUC vs Training Time")
print("=" * 60)
 
# ---------------------------------------------------------------------------
# SECTION 1: Load Engineered Features from Lab 1.3
# ---------------------------------------------------------------------------
print("\n[1] Loading engineered feature matrix from Lab 1.3...")
 
df = pd.read_csv(DATA_DIR / "user_features_engineered.csv")
 
# TODO: Extract the list of feature column names (all columns except 'visitorid' and 'purchased')
FEATURE_COLS = [c for c in df.columns if c not in ['visitorid', 'purchased']]
TARGET = 'purchased'
 
X = df[FEATURE_COLS]
y = df[TARGET]
 
print(f"    Features loaded : {len(FEATURE_COLS) if FEATURE_COLS is not None else 0}")
print(f"    Feature names   : {FEATURE_COLS}")
print(f"    Samples         : {len(df):,}")
print(f"    Purchase rate   : {y.mean()*100:.2f}%")
 
# TODO: Create train/test splits using an 80/20 ratio, setting random_state=42 and stratifying across y
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)
 
# TODO: Compute the negative-to-positive class weight ratio to handle background label imbalance
scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
 
print(f"\n    Train: {len(X_train) if X_train is not None else 0:,}  |  Test: {len(X_test) if X_test is not None else 0:,}")
print(f"    scale_pos_weight: {scale_pos_weight}")
 
# ---------------------------------------------------------------------------
# SECTION 2: XGBoost Reference (Default Params — from Lab 1.3 result)
# ---------------------------------------------------------------------------
print("\n[2] Training XGBoost reference model (default params)...")
 
# TODO: Measure training time and fit an XGBClassifier model to set up your baseline comparison.
# Hyperparameters: n_estimators=200, max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
# scale_pos_weight=scale_pos_weight, eval_metric='auc', early_stopping_rounds=20, random_state=42, verbosity=0
t0 = time.time()
xgb_model = xgb.XGBClassifier(
    n_estimators=200,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=scale_pos_weight,
    eval_metric='auc',
    early_stopping_rounds=20,
    random_state=42,
    verbosity=0
)

# TODO: Fit the model using X_train and y_train while passing eval_set=[(X_test, y_test)]
xgb_model.fit(
    X_train,
    y_train,
    eval_set=[(X_test, y_test)],
    verbose=False
)

xgb_time = time.time() - t0
 
# TODO: Calculate predictions and metrics (ROC-AUC and Average Precision) for the reference model
xgb_proba = xgb_model.predict_proba(X_test)[:, 1]
xgb_auc = roc_auc_score(y_test, xgb_proba)
xgb_ap = average_precision_score(y_test, xgb_proba)
 
print(f"    XGBoost AUC-ROC      : {xgb_auc}")
print(f"    XGBoost Avg-PR       : {xgb_ap}")
print(f"    XGBoost training time: {xgb_time:.1f}s")
 
# ---------------------------------------------------------------------------
# SECTION 3: LightGBM — Default Params (Warm-up)
# ---------------------------------------------------------------------------
print("\n[3] Training LightGBM default (before Optuna tuning)...")
 
# TODO: Initialize and time a default lgb.LGBMClassifier to see out-of-the-box algorithmic performance differences.
# Hyperparameters: n_estimators=200, max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
# scale_pos_weight=scale_pos_weight, random_state=42, verbose=-1
t0 = time.time()
lgb_default = lgb.LGBMClassifier(
    n_estimators=200,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=scale_pos_weight,
    random_state=42,
    verbose=-1
)

# TODO: Fit your lgb_default instance using early stopping and log evaluation callbacks
# Hint: Pass callbacks=[lgb.early_stopping(20, verbose=False), lgb.log_evaluation(-1)] during .fit()
lgb_default.fit(
    X_train,
    y_train,
    eval_set=[(X_test, y_test)],
    callbacks=[
        lgb.early_stopping(20, verbose=False),
        lgb.log_evaluation(-1)
    ]
)
lgb_default_time = time.time() - t0
 
# TODO: Evaluate performance metrics across test boundaries for your baseline LightGBM run
lgb_default_proba = lgb_default.predict_proba(X_test)[:, 1]
lgb_default_auc = roc_auc_score(y_test, lgb_default_proba)
lgb_default_ap = average_precision_score(y_test, lgb_default_proba)
 
print(f"    LightGBM Default AUC      : {lgb_default_auc}")
print(f"    LightGBM Default Avg-PR   : {lgb_default_ap}")
print(f"    LightGBM training time    : {lgb_default_time:.1f}s")
 
# ---------------------------------------------------------------------------
# SECTION 4: Optuna Objective for LightGBM
# ---------------------------------------------------------------------------
print("\n[4] Defining Optuna objective for LightGBM...")
 
def lgb_objective(trial):
    """
    Optuna calls this repeatedly to sample parameter intervals.
    Each call = one experiment with a different set of hyperparameter samples.
    """
    # TODO: Define hyperparameter suggestion boundaries using trial.suggest_* routines
    params = {
    'n_estimators': trial.suggest_int('n_estimators', 100, 600),
    'max_depth': trial.suggest_int('max_depth', 3, 8),
    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
    'num_leaves': trial.suggest_int('num_leaves', 20, 150),
    'subsample': trial.suggest_float('subsample', 0.5, 1.0),
    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
    'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
    'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 2.0),
    'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 2.0),
    'scale_pos_weight': scale_pos_weight,
    'random_state': 42,
    'verbose': -1
}
 
    # TODO: Initialize an lgb.LGBMClassifier unpacking your dynamically suggested trial params dictionary
    model = lgb.LGBMClassifier(**params)
    
    # TODO: Create a StratifiedKFold cross-validator with 3 splits, enabled shuffling, and random_state=42
    cv = StratifiedKFold(
    n_splits=3,
    shuffle=True,
    random_state=42
)
 
    # TODO: Calculate stratified evaluation cross-validation score vectors using cross_val_score
    # Target criteria: use X_train, y_train, scoring='roc_auc', and fix n_jobs=1
    scores = cross_val_score(
    model,
    X_train,
    y_train,
    cv=cv,
    scoring='roc_auc',
    n_jobs=1
)
    
    # Return the mean of your generated cross-validation arrays
    return scores.mean() if scores is not None else 0.0
 
# ---------------------------------------------------------------------------
# SECTION 5: Run Optuna Study
# ---------------------------------------------------------------------------
print("\n[5] Running Optuna on LightGBM (30 trials)...")
 
# TODO: Initialize an optuna study optimized to target maximum values
study = optuna.create_study(direction='maximize')

# TODO: Optimize the study running the lgb_objective routine across 30 distinct trials
t0 = time.time()
study.optimize(
    lgb_objective,
    n_trials=30
)
optuna_search_time = time.time() - t0
 
print(f"\n    Best CV AUC  : {study.best_value if study is not None else 0:.4f}")
print(f"    Search time  : {optuna_search_time:.1f}s")
print(f"    Best params  :")
if study is not None:
    for k, v in study.best_params.items():
        print(f"      {k:<25}: {v}")
 
# ---------------------------------------------------------------------------
# SECTION 6: Retrain Final LightGBM with Best Params
# ---------------------------------------------------------------------------
print("\n[6] Retraining final LightGBM with best params...")
 
# TODO: Extract and append background parameters onto the best configuration output found by your study
best_params = {}
if study is not None:
    best_params = study.best_params.copy()
best_params['scale_pos_weight'] = scale_pos_weight
best_params['random_state']     = 42
best_params['verbose']          = -1
 
# TODO: Instantiate an LGBMClassifier with your optimized structural mapping and measure fit speeds
t0 = time.time()
lgb_tuned = lgb.LGBMClassifier(**best_params)

# TODO: Fit the tuned model configuration using training splits

lgb_tuned.fit(X_train, y_train)
lgb_tuned_time = time.time() - t0
 
# TODO: Generate testing boundary calculations and compute ROC-AUC and Average Precision outputs
lgb_tuned_proba = lgb_tuned.predict_proba(X_test)[:, 1]
lgb_tuned_auc = roc_auc_score(y_test, lgb_tuned_proba)
lgb_tuned_ap = average_precision_score(y_test, lgb_tuned_proba)
 
print(f"    LightGBM Tuned AUC   : {lgb_tuned_auc}")
print(f"    LightGBM Tuned Avg-PR: {lgb_tuned_ap}")
print(f"    Training time        : {lgb_tuned_time:.1f}s")
 
# ---------------------------------------------------------------------------
# SECTION 7: PARETO TRADEOFF TABLE
# ---------------------------------------------------------------------------
print("\n[7] PARETO TRADEOFF — AUC vs Training Time:")
print(f"\n    {'Model':<35} {'AUC':>8} {'AP':>8} {'Time(s)':>9} {'AUC/sec':>10}")
print(f"    {'-'*72}")
 
# TODO: Construct and display metrics analyzing execution time against raw classification score gains

results = [
    ("XGBoost Default", xgb_auc, xgb_ap, xgb_time),
    ("LightGBM Default", lgb_default_auc, lgb_default_ap, lgb_default_time),
    ("LightGBM Tuned", lgb_tuned_auc, lgb_tuned_ap, lgb_tuned_time)
]
for name, auc, ap, t in results:
    print(
        f"    {name:<35} "
        f"{auc:>8.4f} "
        f"{ap:>8.4f} "
        f"{t:>9.1f} "
        f"{auc/t:>10.4f}"
    )
# ---------------------------------------------------------------------------
# SECTION 8: Visualize — Pareto Plot + Optuna History
# ---------------------------------------------------------------------------
print("\n[8] Plotting Pareto tradeoff and Optuna history...")
 
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Lab 1.4: LightGBM + Optuna — AUC vs Training Time Pareto Tradeoff", fontsize=12, fontweight='bold')
 
# --- Plot 1: Pareto Scatter Diagram (AUC vs Execution Time) ---
# TODO: Build an overlay trace layout comparing your model runs as discrete scatter points on axes[0]
# Hint: map training times along the X-axis and computed validation AUC scores across the Y-axis
axes[0].scatter(xgb_time, xgb_auc,
                s=120,
                label='XGBoost')

axes[0].scatter(lgb_default_time, lgb_default_auc,
                s=120,
                label='LGBM Default')

axes[0].scatter(lgb_tuned_time, lgb_tuned_auc,
                s=120,
                label='LGBM Tuned')

axes[0].axhline(y=0.80,
                color='red',
                linestyle='--')

axes[0].grid(True, alpha=0.3)

# TODO: Add target reference marks using axes[0].axhline(y=0.80, color='red', linestyle='--')
axes[0].set_xlabel("Training Time (seconds)")
axes[0].set_ylabel("AUC-ROC")
axes[0].set_title("Pareto: AUC vs Training Time\n(top-left = ideal)")
axes[0].legend(fontsize=7, loc='lower right')
 
# --- Plot 2: Optuna Optimization Exploration History ---
# TODO: Extract trial progression numbers and their objective values from the study object
trial_nums = []
trial_aucs = []
best_so_far = []

if study is not None:
    trial_nums = [t.number for t in study.trials]
    trial_aucs = [t.value for t in study.trials]
    best_so_far = pd.Series(trial_aucs).cummax().values
 
# TODO: Create a scatter plot of individual trial paths alongside a line trace showing cumulative improvements over time
# Hint: Plot trial_nums vs trial_aucs on axes[1] as a scatter plot, and trial_nums vs best_so_far as a line chart
axes[1].scatter(
    trial_nums,
    trial_aucs,
    alpha=0.7,
    label='Trial AUC'
)

axes[1].plot(
    trial_nums,
    best_so_far,
    linewidth=2,
    label='Best So Far'
)

axes[1].grid(True, alpha=0.3)

axes[1].set_title("Optuna Optimization History (LightGBM)")
axes[1].set_xlabel("Trial Number")
axes[1].set_ylabel("CV AUC Score")
axes[1].legend(fontsize=9)
 
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "04_pareto_tradeoff.png", dpi=150, bbox_inches='tight')
plt.show()
 
# ---------------------------------------------------------------------------
# SECTION 9: LightGBM Feature Importance
# ---------------------------------------------------------------------------
print("\n[9] LightGBM tuned feature importance...")
 
# TODO: Build a feature importance ranking DataFrame containing columns 'feature' (FEATURE_COLS) and 'importance' (lgb_tuned.feature_importances_)
# Sort it descending by metric scale impact
lgb_imp = pd.DataFrame({
    'feature': FEATURE_COLS,
    'importance': lgb_tuned.feature_importances_
}).sort_values(
    'importance',
    ascending=False
)

print(lgb_imp.to_string(index=False) if lgb_imp is not None else "    Not Implemented")
 
# --- Horizontal Bar Chart Layout ---
fig, ax = plt.subplots(figsize=(9, 6))
# TODO: Render a horizontal layout tracking relative features metrics using ax.barh()
top_feats = lgb_imp.head(15)

ax.barh(
    top_feats['feature'][::-1],
    top_feats['importance'][::-1]
)

ax.set_title("LightGBM Feature Importance (Tuned)", fontweight='bold')
ax.set_xlabel("Importance")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "04_lgb_feature_importance.png", dpi=150, bbox_inches='tight')
plt.show()
 
print("\n" + "=" * 60)
print("  LAB 1.4 COMPLETE — FULL MODULE 1 PROGRESSION")
print("=" * 60)
