# =============================================================================
# MODULE 2 | LAB 2.2
# File: 02_collaborative_filtering.py
# Purpose: Item-Item Collaborative Filtering on implicit purchase matrix
#          Evaluate Precision@10 vs Content-Based baseline from Lab 2.1
# Saras AI Institute | Build Predictive Models & Modern Recommenders
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sp
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
import pickle
import warnings
warnings.filterwarnings('ignore')
from pathlib import Path
# Project root
ROOT = Path(__file__).resolve().parents[3]


DATA_DIR = ROOT / "data"

# Output folder
OUTPUT_DIR = ROOT / "output/Lab2"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("  MODULE 2 | LAB 2.2")
print("  Item-Item Collaborative Filtering")
print("  Method: Implicit Purchase Matrix + Cosine Similarity")
print("=" * 60)

# ---------------------------------------------------------------------------
# SECTION 1: Load Events and Build Interaction Matrix
# ---------------------------------------------------------------------------
print("\n[1] Loading events and building implicit interaction matrix...")

events = pd.read_csv(DATA_DIR / "events.csv")

# TODO: Define confidence weights for implicit user actions.
# Assign weights: 'view' -> 1, 'addtocart' -> 2, 'transaction' -> 3
event_weights = {
    'view': 1,
    'addtocart': 2,
    'transaction': 3
}

# TODO: Map your defined event weights dictionary onto the events['event'] column to populate a new column 'weight'
events['weight'] = events['event'].map(event_weights)

print(f"    Total events    : {len(events):,}")


# TODO: Aggregate interaction strengths per unique user-item combination.
# Hint: Group the events dataframe by ['visitorid', 'itemid'], extract the 'weight' column, and call .sum(). Reset the index.
interactions = (
    events
    .groupby(['visitorid', 'itemid'])['weight']
    .sum()
    .reset_index()
)

print(f"    Unique user-item pairs: {len(interactions) if interactions is not None else 0:,}")


# ---------------------------------------------------------------------------
# SECTION 2: Build Sparse Matrix
# ---------------------------------------------------------------------------
print("\n[2] Building sparse user-item matrix...")

# TODO: Extract lists of all unique user IDs ('visitorid') and item IDs ('itemid') from the interactions dataframe
user_ids = interactions['visitorid'].unique()
item_ids = interactions['itemid'].unique()

# TODO: Generate continuous coordinate index maps (dictionaries) for users and items
# Format: {raw_id: coordinate_index_integer}
user_to_idx = {u: i for i, u in enumerate(user_ids)}
item_to_idx = {i: j for j, i in enumerate(item_ids)}

# TODO: Map the raw IDs inside your interactions dataframe to their respective structural coordinates
# Hint: Use .map() with your index dictionaries on 'visitorid' and 'itemid' columns to get underlying indices (.values)
row_idx = interactions['visitorid'].map(user_to_idx).values
col_idx = interactions['itemid'].map(item_to_idx).values
data = interactions['weight'].values

# TODO: Assemble a Compressed Sparse Row (CSR) matrix using your mapped coordinates and weights
# Hint: Pass a data configuration tuple ((data, (row_idx, col_idx))) alongside explicit shape constraints
user_item_matrix = sp.csr_matrix(
    (data, (row_idx, col_idx)),
    shape=(len(user_ids), len(item_ids))
)

# TODO: Transpose the user_item_matrix and format explicitly as a CSR matrix to facilitate item similarity calculations
item_user_matrix = user_item_matrix.T.tocsr()

print(f"    User-item matrix shape : {user_item_matrix.shape if user_item_matrix is not None else 'N/A'}")
print(f"    Non-zero entries       : {user_item_matrix.nnz:,}" if user_item_matrix is not None else "N/A")


# ---------------------------------------------------------------------------
# SECTION 3: Compute Item-Item Similarity
# ---------------------------------------------------------------------------
print("\n[3] Computing item-item similarity (batched for memory efficiency)...")

# TODO: Normalize item vectors using L2 norm criteria across the structural item_user_matrix profile
# Hint: Use sklearn's normalize() function specifying norm='l2'
item_user_norm = normalize(item_user_matrix, norm='l2')

MAX_ITEMS = min(1000, len(item_ids)) if item_ids is not None else 5000
print(f"    Computing similarity for top {MAX_ITEMS:,} items (by interaction count)...")

# TODO: Identify the top most interacted items to isolate an executable benchmark evaluation boundary
# Hint: Compute horizontal sums across your item_user_matrix, flatten the array, use np.argsort()[::-1], and slice to MAX_ITEMS
item_counts = np.asarray(item_user_matrix.sum(axis=1)).flatten()

top_item_indices = np.argsort(item_counts)[::-1][:MAX_ITEMS]
top_item_ids = item_ids[top_item_indices]
top_item_set = set(top_item_ids)

# TODO: Subset your normalized item matrix using top_item_indices and compute the baseline cosine similarity against all item vectors
# Hint: Call cosine_similarity(item_subset, item_user_norm)
item_subset = item_user_norm[top_item_indices]

similarity_matrix = cosine_similarity(
    item_subset,
    item_user_norm,
    dense_output=False
)


# ---------------------------------------------------------------------------
# SECTION 4: Build CF Recommendation Function
# ---------------------------------------------------------------------------
print("\n[4] Building CF recommendation function...")

def get_cf_recommendations(item_id, top_k=10):
    """
    Given an item_id, return top_k most similar items based on item-item collaborative filtering.
    Returns a pandas DataFrame tracking 'item_id' and 'similarity'.
    """
    # TODO: Check if item_id exists in item_to_idx mapping boundary. If not, return an empty DataFrame.
    if item_id not in item_to_idx:
        return pd.DataFrame()

    item_idx_in_catalog = item_to_idx[item_id]

    # TODO: Retrieve the full similarity score array for this item.
    # Logic: If item_id is inside top_item_ids, pull its precomputed vector row from similarity_matrix.
    # Otherwise, fall back to calculating its specific cosine similarity on the fly using item_user_norm.
    if item_id in top_item_set:
        row = np.where(top_item_ids == item_id)[0][0]

        sims = similarity_matrix[row].toarray().flatten()

    else:
        sims = cosine_similarity(
            item_user_norm[item_idx_in_catalog],
            item_user_norm
        ).flatten()

    # TODO: Exclude the query item itself from recommendations by overriding its coordinate index in the sims array to -1
    sims[item_idx_in_catalog] = -1
    # TODO: Discover indices tracking the top_k largest similarity values and build a return DataFrame matching target items
    top_indices = np.argsort(sims)[::-1][:top_k]

    return pd.DataFrame({
        'item_id'    : item_ids[top_indices] if item_ids is not None else [],
        'similarity' : sims[top_indices] if sims is not None else []
    })


# ---------------------------------------------------------------------------
# SECTION 5: Test the CF Recommender
# ---------------------------------------------------------------------------
print("\n[5] Testing CF recommender...")

if top_item_ids is not None:
    popular_item = top_item_ids[0]
    print(f"    Query item ID (most popular): {popular_item}")

    # TODO: Run your get_cf_recommendations routine to isolate matching candidate items for popular_item
    cf_recs = get_cf_recommendations(popular_item)
    print(cf_recs.to_string(index=False) if cf_recs is not None else "    Not Implemented")


# ---------------------------------------------------------------------------
# SECTION 6: Evaluate Precision@10 vs CB Baseline
# ---------------------------------------------------------------------------
print("\n[6] Evaluating CF — Precision@10 vs CB baseline...")

# Load Content-Based results from Lab 2.1 archive artifact
try:
    with open(DATA_DIR / "cb_artifacts.pkl", "rb") as f:
        cb_artifacts = pickle.load(f)
    cb_precision = cb_artifacts['cb_results']['precision_at_10']
    print(f"    CB Precision@10 (Lab 2.1): {cb_precision:.4f}")
except:
    cb_precision = None
    print("    CB artifacts not found — run Lab 2.1 first")

# TODO: Isolate transactional events to verify retrieval scores across users with multiple purchases
# Filter events down to 'transaction' values and subset column structures to ['visitorid', 'itemid']
purchases = events[
    events['event'] == 'transaction'
][['visitorid', 'itemid']]
if purchases is not None:
    purchases.columns = ['user_id', 'item_id']

# TODO: Isolate sequential multi-purchase records by keeping users with 2 or more historical purchases
multi_buyers = purchases.groupby('user_id').filter(
    lambda x: len(x) >= 2
)

cf_hits   = 0
cf_total  = 0

if multi_buyers is not None:
    eval_users = multi_buyers['user_id'].unique()[:500]

    for user in eval_users:
        user_items    = purchases[purchases['user_id'] == user]['item_id'].values
        catalog_items = [i for i in user_items if i in item_to_idx]

        if len(catalog_items) < 2:
            continue

        for i in range(len(catalog_items)-1):

            query_item = catalog_items[i]
            target_item = catalog_items[i+1]

            recs = get_cf_recommendations(query_item)

            if recs is None or len(recs) == 0:
                continue

            if target_item in recs['item_id'].values:
                cf_hits += 1

            cf_total += 1

# TODO: Compute ultimate cf_precision ratio tracking performance across valid boundaries
cf_precision = (
    cf_hits / cf_total
    if cf_total > 0
    else 0.0
)

print(f"\n    Evaluated on {cf_total} users")
print(f"    CF Hits@10     : {cf_hits}")
print(f"    CF Precision@10: {cf_precision:.4f}")


# ---------------------------------------------------------------------------
# SECTION 7: Sparsity Problem Visualization
# ---------------------------------------------------------------------------
print("\n[7] Visualizing the sparsity problem...")

# --- Rendering Long Tail Distributions ---
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Lab 2.2: Collaborative Filtering — The Sparsity Problem", fontsize=13, fontweight='bold')

# TODO: Plot an item interaction frequency distribution histogram using axes[0] on a logarithmic scale
# Hint: Use axes[0].hist() and configure axes[0].set_yscale('log') or set log=True
item_interactions = (
    interactions.groupby('itemid')['weight']
    .sum()
)

axes[0].hist(
    item_interactions,
    bins=50,
    log=True
)

axes[0].set_title("Item Interaction Count Distribution\n(log scale — power law)")
axes[0].set_xlabel("Number of Interactions per Item")
axes[0].set_ylabel("Number of Items (log)")

# TODO: Plot user interaction frequency distribution counts using a log histogram format on axes[1]
user_interactions = (
    interactions.groupby('visitorid')['weight']
    .sum()
)

axes[1].hist(
    user_interactions,
    bins=50,
    log=True
)

axes[1].set_title("User Interaction Count Distribution\n(log scale — power law)")
axes[1].set_xlabel("Number of Interactions per User")
axes[1].set_ylabel("Number of Users (log)")

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "02_cf_sparsity_analysis.png", dpi=150, bbox_inches='tight')
plt.show()


# ---------------------------------------------------------------------------
# SECTION 8: Save CF Artifacts for Lab 2.4
# ---------------------------------------------------------------------------
# TODO: Dump your sparse matrix definitions, catalog arrays, index mappings, and calculated metrics to a pickle file
# Path: "data/cf_artifacts.pkl"
cf_artifacts = {
    'user_item_matrix': user_item_matrix,
    'item_user_matrix': item_user_matrix,

    'item_ids': item_ids,
    'user_ids': user_ids,

    'item_to_idx': item_to_idx,
    'user_to_idx': user_to_idx,

    'top_item_ids': top_item_ids,

    'cf_results': {
        'precision_at_10': cf_precision
    }
}

with open(DATA_DIR / "cf_artifacts.pkl", "wb") as f:
    pickle.dump(cf_artifacts, f)

print("\n    Saved -> data/cf_artifacts.pkl")
print("    Move to: 03_lightfm_cold_start.py")
