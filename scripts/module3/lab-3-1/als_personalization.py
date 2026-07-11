# =============================================================================
# MODULE 3 | LAB 3.1
# File: 01_als_personalization.py
# Purpose: Build a personalized recommendation engine using Alternating
#          Least Squares (ALS) on implicit interaction confidence matrices.
# Saras AI Institute | Build Predictive Models & Modern Recommenders
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sp
import pickle
import time
import warnings
warnings.filterwarnings('ignore')
 
from implicit.als import AlternatingLeastSquares
from implicit.evaluation import ndcg_at_k, precision_at_k
from pathlib import Path
# Project root
ROOT = Path(__file__).resolve().parents[3]


DATA_DIR = ROOT / "data"

# Output folder
OUTPUT_DIR = ROOT / "output/Lab3"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
 
print("=" * 60)
print("  MODULE 3 | LAB 3.1")
print("  ALS Personalization Engine")
print("  Method: Alternating Least Squares on Implicit Feedback")
print("=" * 60)
 
# ---------------------------------------------------------------------------
# DEMO PARAMETERS
# ---------------------------------------------------------------------------
N_FACTORS    = 64     
N_ITERATIONS = 10     
N_USERS      = 50000  
N_ITEMS      = 10000  
 
print(f"\n  Demo scale: {N_USERS:,} users | {N_ITEMS:,} items | {N_FACTORS} factors | {N_ITERATIONS} iterations\n")
 
# ---------------------------------------------------------------------------
# SECTION 1: Load Events and Routing Split
# ---------------------------------------------------------------------------
print("[1] Loading events and Module 2 routing split...")
 
events = pd.read_csv(DATA_DIR / "events.csv")
events['datetime'] = pd.to_datetime(events['timestamp'], unit='ms')
 
with open(DATA_DIR / "routing_split.pkl", "rb") as f:
    routing = pickle.load(f)
 
als_users   = routing['als_users']
lfm_users   = routing['lightfm_users']
cutoff_date = routing['cutoff_date']
 
print(f"    Total events       : {len(events):,}")
print(f"    ALS target users   : {len(als_users):,}")
print(f"    LightFM users      : {len(lfm_users):,}")
print(f"    Train cutoff       : {cutoff_date}")
 
# ---------------------------------------------------------------------------
# SECTION 2: Subsample for Demo Speed
# ---------------------------------------------------------------------------
print("\n[2] Subsampling to demo scale...")
 
# TODO: Isolate top users and items based on total interaction volume counts to speed up processing
top_users = (
    events['visitorid']
    .value_counts()
    .head(N_USERS)
    .index
)
top_items = (
    events['itemid']
    .value_counts()
    .head(N_ITEMS)
    .index
)
 
# TODO: Filter the events dataframe to include only records matching top_users and top_items lists
events_sub = events[
    events['visitorid'].isin(top_users) &
    events['itemid'].isin(top_items)
].copy()
 
print(f"    Subsampled events : {len(events_sub) if events_sub is not None else 0:,}")


# ---------------------------------------------------------------------------
# SECTION 3: Build Confidence-Weighted Interaction Matrix
# ---------------------------------------------------------------------------
print("\n[3] Building confidence-weighted interaction matrix...")
 
# TODO: Define mapping parameters to structure implicit signal strengths ($C_{ui} = 1 + \alpha R_{ui}$)
# Values to set: view -> 1.0, addtocart -> 5.0, transaction -> 40.0
ALPHA_VIEW = 1.0
ALPHA_ADDTOCART = 5.0
ALPHA_TXN = 40.0
 
conf_map = {'view': ALPHA_VIEW, 'addtocart': ALPHA_ADDTOCART, 'transaction': ALPHA_TXN}
events_sub['confidence'] = events_sub['event'].map(conf_map)
 
# TODO: Generate continuous unique user and item arrays from events_sub
user_ids = events_sub['visitorid'].unique()
item_ids = events_sub['itemid'].unique()

# TODO: Build conversion dictionary lookups mapping raw identifiers to continuous coordinate integers
user_to_idx = {
    uid: idx
    for idx, uid in enumerate(user_ids)
}

item_to_idx = {
    iid: idx
    for idx, iid in enumerate(item_ids)
}
 
# TODO: Aggregate calculated confidence tracking weights per user-item intersection pair
# Hint: Group events_sub by ['visitorid', 'itemid'] and sum 'confidence' columns, then reset index
interactions = (
    events_sub
    .groupby(['visitorid', 'itemid'])['confidence']
    .sum()
    .reset_index()
)
 
# TODO: Map raw identifiers within interactions down to coordinate row/col index indices
row = interactions['visitorid'].map(user_to_idx)
col = interactions['itemid'].map(item_to_idx)
dat = interactions['confidence']
 
# TODO: Build the sparse matrix tracking user item shapes
# Hint: Instantiate a sp.csr_matrix using configuration tuples: (dat, (row, col))
user_item = sp.csr_matrix(
    (
        dat,
        (row, col)
    ),
    shape=(len(user_ids), len(item_ids))
)


# TODO: Transpose user_item to generate the item_user matrix explicitly as a CSR matrix
item_user = user_item.T.tocsr()

print(f"    Item-user matrix shape: {item_user.shape if item_user is not None else 'N/A'}")


# ---------------------------------------------------------------------------
# SECTION 4: Temporal Train/Test Split
# ---------------------------------------------------------------------------
print("\n[4] Temporal train/test split...")
 
# TODO: Partition events_sub into training logs (<= cutoff_date) and testing logs (> cutoff_date and transaction events only)
train_events = events_sub[
    events_sub['datetime'] <= cutoff_date
].copy()

test_events = events_sub[
    (events_sub['datetime'] > cutoff_date) &
    (events_sub['event'] == 'transaction')
].copy()
 
# TODO: Aggregate train interactions and map them to row/col indexing arrays to build train_user_item
train_interactions = (
    train_events
    .groupby(['visitorid', 'itemid'])['confidence']
    .sum()
    .reset_index()
)

train_interactions = train_interactions[
    train_interactions['visitorid'].isin(user_to_idx) &
    train_interactions['itemid'].isin(item_to_idx)
]

train_row = train_interactions['visitorid'].map(user_to_idx)
train_col = train_interactions['itemid'].map(item_to_idx)

train_user_item = sp.csr_matrix(
    (
        train_interactions['confidence'],
        (train_row, train_col)
    ),
    shape=(len(user_ids), len(item_ids))
)
# Remember to expose item_user matrix variants for implicit model ingestion

train_item_user = train_user_item.T.tocsr()
 
# TODO: Compile test purchases matrix using np.ones structure to build test labels
# Note: Ensure you check that test item and user IDs map within index dictionary boundaries using .notna()
test_events = test_events.copy()

test_events['user_idx'] = test_events['visitorid'].map(user_to_idx)
test_events['item_idx'] = test_events['itemid'].map(item_to_idx)

test_events = test_events[
    test_events['user_idx'].notna() &
    test_events['item_idx'].notna()
]

test_user_item = sp.csr_matrix(
    (
        np.ones(len(test_events)),
        (
            test_events['user_idx'].astype(int),
            test_events['item_idx'].astype(int)
        )
    ),
    shape=(len(user_ids), len(item_ids))
)

test_item_user = test_user_item.T.tocsr()
 
print(f"    Train matrix shape : {train_user_item.shape if train_user_item is not None else 'N/A'}")
print(f"    Test matrix shape  : {test_user_item.shape if test_user_item is not None else 'N/A'}")
 
# ---------------------------------------------------------------------------
# SECTION 5: Train ALS Model
# ---------------------------------------------------------------------------
print("\n[5] Training ALS model...")
 
# TODO: Instantiate an AlternatingLeastSquares model object.
# Parameters: factors=N_FACTORS, regularization=0.01, iterations=N_ITERATIONS, use_gpu=False, random_state=42
model = AlternatingLeastSquares(
    factors=N_FACTORS,
    regularization=0.01,
    iterations=N_ITERATIONS,
    use_gpu=False,
    random_state=42
)

# TODO: Fit the model configuration using the train_item_user matrix
t0 = time.time()
model.fit(train_user_item)
train_time = time.time() - t0
 
print(f"\n    Training complete in {train_time:.1f}s")


# ---------------------------------------------------------------------------
# SECTION 6: Evaluate — NDCG@10
# ---------------------------------------------------------------------------
print("\n[6] Evaluating ALS model...")
 
# TODO: Calculate NDCG@10 rankings using implicit evaluation tools
# Hint: Call ndcg_at_k() providing model, train_item_user, test_item_user, and K=10
baseline_ndcg = ndcg_at_k(
    model,
    train_user_item,
    test_user_item,
    K=10
)
 
print(f"    ALS NDCG@10    : {baseline_ndcg:.4f}")
 
# ---------------------------------------------------------------------------
# SECTION 7: Confidence Weight Experiment
# ---------------------------------------------------------------------------
print("\n[7] Confidence weight tuning experiment...")
 
configs = [
    ("Low emphasis  (view=1, cart=2, txn=10)",  1.0, 2.0,  10.0),
    ("Balanced      (view=1, cart=5, txn=40)",  1.0, 5.0,  40.0),
    ("Purchase-heavy(view=0.5,cart=3, txn=80)", 0.5, 3.0,  80.0),
]
 
print(f"\n    {'Config':<45} {'NDCG@10':>8} {'Time(s)':>8}")
print(f"    {'-'*63}")
 
best_ndcg   = -1
best_model  = model
best_config = "Balanced (view=1, cart=5, txn=40)"
best_item_user = train_item_user
config_scores = []
for config_name, a_view, a_cart, a_txn in configs:
    # TODO: Loop over the weight configurations, construct temporary confidence maps,
    # aggregate weights, build a temporary sparse matrix, fit a new ALS model, 
    # and track the best performing trial based on NDCG@10 scores.
    temp_events = train_events.copy()

    temp_events['confidence'] = temp_events['event'].map({
        'view': a_view,
        'addtocart': a_cart,
        'transaction': a_txn
    })

    temp_interactions = (
        temp_events
        .groupby(['visitorid', 'itemid'])['confidence']
        .sum()
        .reset_index()
    )

    rows = temp_interactions['visitorid'].map(user_to_idx)
    cols = temp_interactions['itemid'].map(item_to_idx)

    temp_user_item = sp.csr_matrix(
        (
            temp_interactions['confidence'],
            (rows, cols)
        ),
        shape=(len(user_ids), len(item_ids))
    )

    temp_item_user = temp_user_item.T.tocsr()

    als = AlternatingLeastSquares(
        factors=N_FACTORS,
        regularization=0.01,
        iterations=N_ITERATIONS,
        use_gpu=False,
        random_state=42
    )

    start = time.time()
    als.fit(temp_user_item)
    elapsed = time.time() - start

    score = ndcg_at_k(
    als,
    temp_user_item,
    test_user_item,
    K=10
)

    config_scores.append(score)

    print(f"    {config_name:<45} {score:>8.4f} {elapsed:>8.1f}")

    if score > best_ndcg:
        best_ndcg = score
        best_model = als
        best_config = config_name
        best_item_user = temp_item_user
 
# Reset variables to capture optimal structures found
model          = best_model
item_user      = best_item_user
 
# ---------------------------------------------------------------------------
# SECTION 8: Sample Recommendations
# ---------------------------------------------------------------------------
print("\n[8] Sample recommendations for returning users...")

if user_ids is not None and model is not None:

    sample_users = user_ids[:5]

    for user_id in sample_users:

        user_idx = user_to_idx[user_id]

        item_indices, scores = model.recommend(
            userid=user_idx,
            user_items=train_user_item[user_idx],
            N=5,
            filter_already_liked_items=True
        )

        print(f"\nUser {user_id}")

        for idx, score in zip(item_indices, scores):
            idx = int(idx)
            print(f"    Item {item_ids[idx]}   Score={score:.4f}")
# ---------------------------------------------------------------------------
# SECTION 9: Visualizations
# ---------------------------------------------------------------------------
print("\n[9] Generating visualizations...")
 
# --- Latent Embedding Profile Vector Magnitudes ---
# TODO: Calculate L2 norms across both user and item latent matrices (model.user_factors, model.item_factors)
# Hint: Use np.linalg.norm(..., axis=1)
user_norms = np.linalg.norm(
    model.user_factors,
    axis=1
)

item_norms = np.linalg.norm(
    model.item_factors,
    axis=1
)
 
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("Lab 3.1: ALS Personalization Engine\nEmbedding Analysis & Confidence Weight Tuning", fontsize=12, fontweight='bold')
 
# TODO: Plot histograms on axes[0] and axes[1] detailing user and item L2 embedding norms distributions

axes[0].hist(user_norms, bins=30)

axes[0].set_title("User Embedding Norms")
axes[0].set_xlabel("L2 Norm")
axes[0].set_ylabel("Users")

axes[1].hist(item_norms, bins=30)

axes[1].set_title("Item Embedding Norms")
axes[1].set_xlabel("L2 Norm")
axes[1].set_ylabel("Items")
# --- Config Performance Comparisons ---
# TODO: Construct a bar plot layout on axes[2] mapping calculated NDCG scores across configs list
# Hint: Use axes[2].bar()


axes[2].set_title("Confidence Weight Tuning\nNDCG@10 per Config")
axes[2].set_ylabel("NDCG@10")
axes[2].bar(
    [c[0] for c in configs],
    config_scores
)

axes[2].tick_params(axis='x', rotation=20)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "01_als_analysis.png", dpi=150, bbox_inches='tight')
plt.show()
 
# ---------------------------------------------------------------------------
# SECTION 10: Save Artifacts
# ---------------------------------------------------------------------------
print("\n[10] Saving ALS artifacts...")
 
# TODO: Export tracking matrices, mappings lists, indexes, and factors down to a binary pickle format
# Target Path: "data/als_artifacts.pkl"

artifacts = {
    "model": model,

    # ALS embeddings
    "item_factors": model.item_factors,
    "user_factors": model.user_factors,

    # IDs
    "user_ids": user_ids,
    "item_ids": item_ids,

    # Lookup dictionaries
    "user_to_idx": user_to_idx,
    "item_to_idx": item_to_idx,

    # Sparse matrices
    "user_item_matrix": train_user_item,
    "item_user": item_user,

    # Training configuration
    "best_config": best_config
}

with open(DATA_DIR / "als_artifacts.pkl", "wb") as f:
    pickle.dump(artifacts, f)
print("    Saved -> data/als_artifacts.pkl")
print("    Move to: 02_faiss_index.py")
