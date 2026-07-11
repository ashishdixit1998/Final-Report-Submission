# =============================================================================
# MODULE 2 | LAB 2.3
# File: 03_lightfm_cold_start.py
# Purpose: LightFM Hybrid Recommender with Logistic Loss and Item Features
#          Measure Cold-Start Improvement over Lab 2.2 Baseline
# Saras AI Institute | Build Predictive Models & Modern Recommenders
# =============================================================================

# NOTE:
# WARP and BPR losses caused native LightFM process termination
# on the local Windows environment (Python 3.10 / LightFM 1.19).
# Logistic loss was used to complete the hybrid recommendation workflow.

import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sp
import pickle
import warnings
warnings.filterwarnings('ignore')

from lightfm import LightFM
from lightfm.data import Dataset
from lightfm.evaluation import precision_at_k, recall_at_k, auc_score
from pathlib import Path
# Project root
ROOT = Path(__file__).resolve().parents[3]


DATA_DIR = ROOT / "data"

# Output folder
OUTPUT_DIR = ROOT / "output/Lab2"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("  MODULE 2 | LAB 2.3")
print("  LightFM Hybrid Recommender")
print("  Method: Logistic Loss + Item Feature Embeddings")
print("=" * 60)

# ---------------------------------------------------------------------------
# SECTION 1: Load Data and Artifacts from Previous Labs
# ---------------------------------------------------------------------------
print("\n[1] Loading data and artifacts from Labs 2.1 and 2.2...")

events = pd.read_csv(DATA_DIR / "events.csv")

with open(DATA_DIR / "cf_artifacts.pkl", "rb") as f:
    cf_artifacts = pickle.load(f)

cf_precision = cf_artifacts['cf_results']['precision_at_10']
print(f"    CF Precision@10 (Lab 2.2) : {cf_precision:.4f}  <- baseline to beat")

purchases = events[events['event'] == 'transaction'][['visitorid', 'itemid']]
all_interactions = events[['visitorid', 'itemid', 'event']].copy()

# ---------------------------------------------------------------------------
# DEMO PARAMETERS – adjust these to scale up for production
# ---------------------------------------------------------------------------

N_USERS      = 10000     # Top 20k most active users
N_ITEMS      = 2000      # Top 5k most interacted items
N_PROPERTIES = 10        # Top 20 item properties
N_COMPONENTS = 16        # Embedding dimension
N_EPOCHS     = 3         # Training epochs
N_THREADS    = 4         # CPU threads

print(
    f"\nDemo scale: "
    f"{N_USERS:,} users | "
    f"{N_ITEMS:,} items | "
    f"{N_PROPERTIES} properties | "
    f"{N_EPOCHS} epochs\n"
)


# ---------------------------------------------------------------------------
# SECTION 2: Load Item Features
# ---------------------------------------------------------------------------
print("\n[2] Subsampling dataset to demo scale...")

top_users = (
    events.groupby('visitorid')
    .size()
    .sort_values(ascending=False)
    .head(N_USERS)
    .index
    .tolist()
)

top_items = (
    events.groupby('itemid')
    .size()
    .sort_values(ascending=False)
    .head(N_ITEMS)
    .index
    .tolist()
)

events_sub = events[
    events['visitorid'].isin(top_users)
    &
    events['itemid'].isin(top_items)
].copy()

print(f"    Events    : {len(events_sub):,}")
print(f"    Users     : {events_sub['visitorid'].nunique():,}")
print(f"    Items     : {events_sub['itemid'].nunique():,}")
print(
    f"    Purchases : "
    f"{(events_sub['event']=='transaction').sum():,}"
)

events = events_sub

purchases = events[
    events['event'] == 'transaction'
][['visitorid', 'itemid']]

all_interactions = events[
    ['visitorid', 'itemid', 'event']
].copy()

print("\n[2.1] Loading item features for LightFM...")

props1 = pd.read_csv(DATA_DIR / "item_properties_part1.csv")
props2 = pd.read_csv(DATA_DIR / "item_properties_part2.csv")
props  = pd.concat([props1, props2], ignore_index=True)

# Keep the most recent record per item-property pair
props_latest = (
    props.sort_values('timestamp', ascending=False)
    .drop_duplicates(subset=['itemid', 'property'])
)

top_properties = (
    props_latest['property']
    .value_counts()
    .head(N_PROPERTIES)
    .index
)

props_latest = props_latest[
    props_latest['property'].isin(top_properties)
]

print("    Building item feature tuples...")
# TODO: Create structural item features combining property names and string values
# Hint: Assign a 'feature' column using props_latest['property'] + '_' + props_latest['value'].astype(str)
# Then groupby 'itemid' and map features into lists of strings.
item_features_raw = None
# TODO: Create structural item features combining property names and string values

props_latest = props_latest.copy()
del props
del props1
del props2

props_latest['feature'] = (
    props_latest['property'].astype(str)
    + '_'
    + props_latest['value'].astype(str)
)

feature_counts = props_latest['feature'].value_counts()

valid_features = set(
    feature_counts[feature_counts >= 50].index
)

props_latest = props_latest[
    props_latest['feature'].isin(valid_features)
]

item_features_raw = (
    props_latest.groupby('itemid')['feature']
    .apply(list)
    .to_dict()
)

print(f"    Items with features: {len(item_features_raw) if item_features_raw is not None else 0:,}")


# ---------------------------------------------------------------------------
# SECTION 3: Build LightFM Dataset
# ---------------------------------------------------------------------------
print("\n[3] Building LightFM dataset...")

# TODO: Instantiate a LightFM Dataset() wrapper object
dataset = Dataset()

# TODO: Extract structural arrays capturing unique users, items, and feature strings
all_users = all_interactions['visitorid'].unique()
all_items = all_interactions['itemid'].unique()
all_features = props_latest['feature'].unique()

# TODO: Fit the dataset container registry

dataset.fit(
    users=all_users,
    items=all_items,
    item_features=all_features
)

print(f"    Users registered  : {len(all_users) if all_users is not None else 0:,}")
print(f"    Items registered  : {len(all_items) if all_items is not None else 0:,}")


# ---------------------------------------------------------------------------
# SECTION 4: Build Interaction Matrix and Feature Matrix
# ---------------------------------------------------------------------------
print("\n[4] Building interaction and feature matrices...")

event_weights = {'view': 1, 'addtocart': 2, 'transaction': 3}
interactions_weighted = all_interactions.copy()
interactions_weighted['weight'] = interactions_weighted['event'].map(event_weights)

# TODO: Build the primary user-item interaction and weights coordinate arrays
# Hint: Use dataset.build_interactions() passing an iterable of (visitorid, itemid, weight) tuples
interaction_tuples = [
    (row.visitorid, row.itemid)
    for row in interactions_weighted.itertuples()
]
interactions_matrix, weights_matrix = dataset.build_interactions(
    interaction_tuples
)

# TODO: Build the sparse item feature coordinate lookup matrix
# Hint: Use dataset.build_item_features() passing an iterable list of (itemid, [feature_list]) tuples
# Ensure you filter elements down to registered_items only to avoid index out-of-bounds mismatches
registered_items = set(all_items) if all_items is not None else set()
item_feature_tuples = [
    (item, features)
    for item, features in item_features_raw.items()
    if item in registered_items
]

item_features_matrix = dataset.build_item_features(
    item_feature_tuples
)

print(f"    Interaction matrix shape: {interactions_matrix.shape if interactions_matrix is not None else 'N/A'}")
print(f"    Item feature matrix shape: {item_features_matrix.shape if item_features_matrix is not None else 'N/A'}")


# ---------------------------------------------------------------------------
# SECTION 5: Train / Test Split
# ---------------------------------------------------------------------------
print("\n[5] Temporal train/test split...")

events['datetime'] = pd.to_datetime(events['timestamp'], unit='ms')

# TODO: Calculate an 80% quantile temporal threshold border string across the 'datetime' axis to define a split
cutoff_date = events['datetime'].quantile(0.80)
print(f"    Cutoff date : {cutoff_date}")

# TODO: Partition interactions_weighted into train_events (<= cutoff) and test_events (> cutoff and event == 'transaction')
events_with_weights = interactions_weighted.copy()

events_with_weights['datetime'] = pd.to_datetime(
    events['timestamp'],
    unit='ms'
)

train_events = events_with_weights[
    events_with_weights['datetime'] <= cutoff_date
]

test_events = events_with_weights[
    (events_with_weights['datetime'] > cutoff_date)
    &
    (events_with_weights['event'] == 'transaction')
]
# Remove user-item pairs that already exist in training

train_pairs = set(
    zip(train_events['visitorid'], train_events['itemid'])
)

test_events = test_events[
    ~test_events.apply(
        lambda r: (r['visitorid'], r['itemid']) in train_pairs,
        axis=1
    )
]

print(f"    Test interactions after overlap removal: {len(test_events):,}")


# TODO: Compile train_matrix and test_matrix structures using your instantiated dataset.build_interactions helper

train_matrix, _ = dataset.build_interactions(
    [
        (r.visitorid, r.itemid)
        for r in train_events.itertuples()
    ]
)

test_matrix, _ = dataset.build_interactions(
    [
        (r.visitorid, r.itemid)
        for r in test_events.itertuples()
    ]
)
train_matrix = train_matrix.tocsr()
train_matrix.data[:] = 1
test_matrix = test_matrix.tocsr()
test_matrix.data[:] = 1
item_features_matrix = item_features_matrix.tocsr()
# ---------------------------------------------------------------------------
# SECTION 6: Train LightFM — Pure CF (No Features)
# ---------------------------------------------------------------------------
print("\n[6] Training LightFM — Pure CF mode (no item features)...")

# TODO: Initialize a LightFM model to test Collaborative Filtering behavior
# Hyperparameters: no_components=64, loss='Logistic', learning_rate=0.05, item_alpha=1e-6, user_alpha=1e-6, random_state=42

model_cf = LightFM(
    no_components=N_COMPONENTS,
    loss='logistic',
    learning_rate=0.05,
    item_alpha=1e-6,
    user_alpha=1e-6,
    random_state=42
)

# TODO: Fit model_cf
try:
    print("Starting model training...")
   # model_cf.fit(
   # train_matrix,
   # epochs=N_EPOCHS,
   # num_threads=N_THREADS
#)

    model_cf.fit(
    train_matrix,
    epochs=N_EPOCHS,
    num_threads=N_THREADS
)

    print("Model training completed successfully.")
    print("Item embeddings shape:", model_cf.item_embeddings.shape)

except ValueError as e:
    print(f"Invalid input data: {e}")

except MemoryError as e:
    print(f"Insufficient memory during training: {e}")

except RuntimeError as e:
    print(f"Runtime error during training: {e}")

except Exception as e:
    print(f"Unexpected error during model training: {type(e).__name__}: {e}")
# TODO: Fit model_cf onto your train_matrix using 20 training epochs and num_threads=4


# TODO: Calculate mean metric scores across test and train matrices using LightFM's integrated precision_at_k function
# Hint: Remember to supply your train_interactions=train_matrix constraint when calculating test precision to exclude training hits

cf_train_precision = precision_at_k(
    model_cf,
    train_matrix,
    k=10
).mean()

cf_test_precision = precision_at_k(
    model_cf,
    test_matrix,
    train_interactions=train_matrix,
    k=10
).mean()

print(f"    LightFM CF Train Precision@10 : {cf_train_precision:.4f}")
print(f"    LightFM CF Test  Precision@10 : {cf_test_precision:.4f}")


# ---------------------------------------------------------------------------
# SECTION 7: Train LightFM — Hybrid (CF + Item Features)
# ---------------------------------------------------------------------------
print("\n[7] Training LightFM — Hybrid mode (CF + item features)...")

# TODO: Initialize an identical hyperparameter configuration for your hybrid runner

model_hybrid = LightFM(
    no_components=N_COMPONENTS,
    loss='logistic',
    learning_rate=0.05,
    item_alpha=1e-6,
    user_alpha=1e-6,
    random_state=42
)

# TODO: Fit hybrid model

model_hybrid.fit(
    train_matrix,
    item_features=item_features_matrix,
    epochs=N_EPOCHS,
    num_threads=N_THREADS
)

# TODO: Fit your hybrid model on train_matrix while explicitly providing item_features=item_features_matrix


# TODO: Evaluate performance values tracking precision_at_k(..., k=10) with your added item_features_matrix maps

hybrid_train_precision = precision_at_k(
    model_hybrid,
    train_matrix,
    item_features=item_features_matrix,
    k=10
).mean()

hybrid_test_precision = precision_at_k(
    model_hybrid,
    test_matrix,
    train_interactions=train_matrix,
    item_features=item_features_matrix,
    k=10
).mean()

print(f"    LightFM Hybrid Train Precision@10 : {hybrid_train_precision:.4f}")
print(f"    LightFM Hybrid Test  Precision@10 : {hybrid_test_precision:.4f}")


# ---------------------------------------------------------------------------
# SECTION 8: Cold-Start Improvement Demonstration
# ---------------------------------------------------------------------------
print("\n[8] Measuring cold-start improvement...")

# TODO: Extract user ID sets across test and train partitions to pinpoint users with zero interaction histories
# Hint: Subtract train visitorid sets from test visitorid sets
train_users = set(train_events['visitorid'])
test_users = set(test_events['visitorid'])

cold_start_users = test_users - train_users

print(f"    Cold-start users       : {len(cold_start_users):,}")


# ---------------------------------------------------------------------------
# SECTION 9: Full Comparison Table
# ---------------------------------------------------------------------------
print("\n[9] FULL MODEL COMPARISON:")
print(f"\n    {'Model':<40} {'Precision@10':>12}")
print(f"    {'-'*55}")
print(f"    {'Item-Item CF (Lab 2.2)':<40} {cf_precision:>12.4f}")
print(f"    {'LightFM Pure CF (no features)':<40} {cf_test_precision:>12.4f}")
print(f"    {'LightFM Hybrid (CF + item features)':<40} {hybrid_test_precision:>12.4f}")


# ---------------------------------------------------------------------------
# SECTION 10: Training Progression Visualization
# ---------------------------------------------------------------------------
print("\n[10] Plotting training progression...")

cf_epochs     = []
hybrid_epochs = []

# TODO: Re-instantiate separate progress tracking estimators matching your hyperparameter configs

model_cf_prog = LightFM(
    no_components=N_COMPONENTS,
    loss='logistic',
    learning_rate=0.05,
    item_alpha=1e-6,
    user_alpha=1e-6,
    random_state=42
)

model_hybrid_prog = LightFM(
    no_components=N_COMPONENTS,
    loss='logistic',
    learning_rate=0.05,
    item_alpha=1e-6,
    user_alpha=1e-6,
    random_state=42
)

for epoch in range(1, N_EPOCHS + 1):
    # TODO: Perform single incremental training steps using .fit_partial() across each epoch loop
    # Ensure item_features are supplied to the hybrid progress model instance
    # TODO: Calculate evaluation outputs for each model slice at the current epoch step and append results to metrics lists
    # TODO: Perform single incremental training steps

    model_cf_prog.fit_partial(
        train_matrix,
        epochs=1,
        num_threads=N_THREADS
    )

    model_hybrid_prog.fit_partial(
        train_matrix,
        item_features=item_features_matrix,
        epochs=1,
        num_threads=N_THREADS
    )

# TODO: Calculate evaluation outputs

    cf_p = precision_at_k(
        model_cf_prog,
        test_matrix,
        train_interactions=train_matrix,
        k=10
    ).mean()

    h_p = precision_at_k(
        model_hybrid_prog,
        test_matrix,
        train_interactions=train_matrix,
        item_features=item_features_matrix,
        k=10
    ).mean()
    
    cf_epochs.append(cf_p)
    hybrid_epochs.append(h_p)

# --- Generate Step Tracking Evaluation Plots ---
fig, ax = plt.subplots(figsize=(10, 5))
# TODO: Overlay line traces plotting cf_epochs and hybrid_epochs performance metrics using ax.plot()
ax.plot(
    range(1, N_EPOCHS + 1),
    cf_epochs,
    marker='o',
    label='Pure CF'
)

ax.plot(
    range(1, N_EPOCHS + 1),
    hybrid_epochs,
    marker='s',
    label='Hybrid'
)

ax.set_title("Lab 2.3: LightFM Training Progression\nHybrid vs Pure CF — Precision@10 per Epoch", fontweight='bold')
ax.set_xlabel("Training Epoch")
ax.set_ylabel("Precision@10")
ax.legend()
ax.grid(alpha=0.3)

plt.tight_layout()
os.makedirs(OUTPUT_DIR, exist_ok=True)
plt.savefig(OUTPUT_DIR / "03_lightfm_training.png", dpi=150, bbox_inches='tight')
plt.show()


# ---------------------------------------------------------------------------
# SECTION 11: Save LightFM Artifacts for Lab 2.4
# ---------------------------------------------------------------------------
# TODO: Save models, metadata datasets, and coordinate matrices to an output pickle package
# Target Path: "data/lightfm_artifacts.pkl"
# TODO: Save models, metadata datasets, and coordinate matrices

results = {
    "cf_train_precision": float(cf_train_precision),
    "cf_test_precision": float(cf_test_precision),
    "hybrid_train_precision": float(hybrid_train_precision),
    "hybrid_test_precision": float(hybrid_test_precision),
    "cold_start_users_count": len(cold_start_users)
}

lightfm_artifacts = {
    "model_cf": model_cf,
    "model_hybrid": model_hybrid,
    "dataset": dataset,

    # Required by Lab 2.4
    "train_matrix": train_matrix,
    "test_matrix": test_matrix,
    "item_features_matrix": item_features_matrix,
    "cold_start_users": cold_start_users,

    "results": results
}

with open(DATA_DIR / "lightfm_artifacts.pkl", "wb") as f:
    pickle.dump(lightfm_artifacts, f)

print("\n    Saved -> data/lightfm_artifacts.pkl")
print("    Move to: 04_evaluation_comparison.py")
