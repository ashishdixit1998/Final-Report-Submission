# =============================================================================
# MODULE 2 | LAB 2.4
# File: 04_evaluation_comparison.py
# Purpose: Head-to-head evaluation of all three models
#          Metrics: Precision@K, NDCG@K, Coverage
#          Build comparison table + 20/80 A/B user split
# Saras AI Institute | Build Predictive Models & Modern Recommenders
# =============================================================================
#
# TEACHING NOTE:
# We now have three recommenders built across Labs 2.1, 2.2, and 2.3.
# This lab evaluates them fairly on the SAME test users using THREE metrics:
#
# Precision@K : Of the top K recommendations, how many did the user buy?
# NDCG@K      : Did relevant items appear at the TOP of the list?
#               (Normalized Discounted Cumulative Gain — position matters)
# Coverage    : What fraction of the total catalog can each model recommend?
#               (A model recommending only popular items has low coverage)
#
# Finally: we implement the 20/80 A/B split — the foundation of Module 3.
# =============================================================================
 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sp
import pickle
import warnings
warnings.filterwarnings('ignore')
 
from lightfm.evaluation import precision_at_k, recall_at_k
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
from pathlib import Path
# Project root
ROOT = Path(__file__).resolve().parents[3]


DATA_DIR = ROOT / "data"

# Output folder
OUTPUT_DIR = ROOT / "output/Lab2"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
 
print("=" * 60)
print("  MODULE 2 | LAB 2.4")
print("  Head-to-Head Model Evaluation")
print("  Metrics: Precision@K, NDCG@K, Coverage")
print("=" * 60)
 
# ---------------------------------------------------------------------------
# SECTION 1: Load All Artifacts
# ---------------------------------------------------------------------------
print("\n[1] Loading all model artifacts...")
 
with open(DATA_DIR / "cb_artifacts.pkl", "rb") as f:
    cb_artifacts = pickle.load(f)
 
with open(DATA_DIR / "cf_artifacts.pkl", "rb") as f:
    cf_artifacts = pickle.load(f)
 
with open(DATA_DIR / "lightfm_artifacts.pkl", "rb") as f:
    lfm_artifacts = pickle.load(f)
 
# CB artifacts
cb_tfidf_matrix  = cb_artifacts['tfidf_matrix']
cb_item_ids      = cb_artifacts['item_ids']
cb_vectorizer    = cb_artifacts['vectorizer']
 
# CF artifacts
cf_user_item     = cf_artifacts['user_item_matrix']
cf_item_ids      = cf_artifacts['item_ids']
cf_item_to_idx   = cf_artifacts['item_to_idx']
cf_user_to_idx   = cf_artifacts['user_to_idx']
 
# LightFM artifacts
lfm_model        = lfm_artifacts['model_hybrid']
lfm_dataset      = lfm_artifacts['dataset']
lfm_test_matrix  = lfm_artifacts['test_matrix']
lfm_train_matrix = lfm_artifacts['train_matrix']
lfm_item_features= lfm_artifacts['item_features_matrix']
cold_start_users = lfm_artifacts['cold_start_users']
 
# Load events
events    = pd.read_csv(DATA_DIR / "events.csv")
purchases = events[events['event'] == 'transaction'][['visitorid', 'itemid']].copy()
purchases.columns = ['user_id', 'item_id']
 
print(f"    All artifacts loaded successfully")
print(f"    CB items indexed    : {len(cb_item_ids):,}")
print(f"    CF matrix shape     : {cf_user_item.shape}")
print(f"    LightFM test matrix : {lfm_test_matrix.shape}")
 
# ---------------------------------------------------------------------------
# SECTION 2: Define NDCG@K Function
# ---------------------------------------------------------------------------
# NDCG = Normalized Discounted Cumulative Gain
# It rewards models that put relevant items HIGHER in the ranked list.
# Position 1 counts more than position 10.
# Perfect NDCG = 1.0 (all relevant items at the top)
 
def ndcg_at_k(recommended_items, relevant_items, k=10):
    """
    Compute NDCG@K for one user.
    recommended_items: ordered list of recommended item IDs
    relevant_items   : set of items the user actually interacted with
    """
    recommended_k = recommended_items[:k]
 
    # DCG: sum of relevance / log2(position+1)
    dcg = 0.0
    for i, item in enumerate(recommended_k):
        if item in relevant_items:
            dcg += 1.0 / np.log2(i + 2)  # i+2 because log2(1)=0
 
    # Ideal DCG: all relevant items at the top
    ideal_hits = min(len(relevant_items), k)
    idcg = sum(1.0 / np.log2(i + 2) for i in range(ideal_hits))
 
    return dcg / idcg if idcg > 0 else 0.0
 
# ---------------------------------------------------------------------------
# SECTION 3: Define Coverage Metric
# ---------------------------------------------------------------------------
def compute_coverage(all_recommendations, total_catalog_size):
    """
    Coverage = fraction of catalog items that appear in any recommendation.
    Low coverage = model recommends the same popular items to everyone.
    High coverage = model discovers the long tail of the catalog.
    """
    recommended_items = set()
    for recs in all_recommendations:
        recommended_items.update(recs)
    return len(recommended_items) / total_catalog_size
 
# ---------------------------------------------------------------------------
# SECTION 4: CB Recommendation Function
# ---------------------------------------------------------------------------
def get_cb_recommendations(item_id, top_k=10):
    if item_id not in cb_item_ids:
        return []
    idx  = np.where(cb_item_ids == item_id)[0][0]
    vec  = cb_tfidf_matrix[idx]
    sims = cosine_similarity(vec, cb_tfidf_matrix).flatten()
    sims[idx] = -1
    top_idx = np.argsort(sims)[::-1][:top_k]
    return list(cb_item_ids[top_idx])
 
# ---------------------------------------------------------------------------
# SECTION 5: CF Recommendation Function
# ---------------------------------------------------------------------------
item_user_norm = normalize(cf_user_item.T.tocsr(), norm='l2')
 
def get_cf_recommendations(item_id, top_k=10):
    if item_id not in cf_item_to_idx:
        return []
    idx  = cf_item_to_idx[item_id]
    vec  = item_user_norm[idx]
    sims = cosine_similarity(vec, item_user_norm).flatten()
    sims[idx] = -1
    top_idx = np.argsort(sims)[::-1][:top_k]
    return list(cf_item_ids[top_idx])
 
# ---------------------------------------------------------------------------
# SECTION 6: LightFM Recommendation Function
# ---------------------------------------------------------------------------
lfm_user_map, _, lfm_item_map, _ = lfm_dataset.mapping()
lfm_item_ids_list = list(lfm_item_map.keys())
n_lfm_items = len(lfm_item_ids_list)
 
def get_lfm_recommendations(user_id, top_k=10):
    if user_id not in lfm_user_map:
        return []
    user_idx = lfm_user_map[user_id]
    scores   = lfm_model.predict(
        user_ids=np.full(n_lfm_items, user_idx),
        item_ids=np.arange(n_lfm_items),
        item_features=lfm_item_features
    )
    top_idx = np.argsort(scores)[::-1][:top_k]
    return [lfm_item_ids_list[i] for i in top_idx]
 
# ---------------------------------------------------------------------------
# SECTION 7: Run Evaluation — All Three Models
# ---------------------------------------------------------------------------
print("\n[2] Running head-to-head evaluation (500 users, K=10)...")
 
K = 10
eval_users = purchases['user_id'].unique()[:500]
 
cb_precisions,  cb_ndcgs,  cb_recs_all  = [], [], []
cf_precisions,  cf_ndcgs,  cf_recs_all  = [], [], []
lfm_precisions, lfm_ndcgs, lfm_recs_all = [], [], []
 
for i, user in enumerate(eval_users):
    relevant   = set(purchases[purchases['user_id'] == user]['item_id'].values)
    query_items = list(relevant)
    if not query_items:
        continue
    query_item = query_items[0]
    remaining  = relevant - {query_item}
    if not remaining:
        continue
 
    # Content-Based
    cb_recs = get_cb_recommendations(query_item, top_k=K)
    if cb_recs:
        cb_precisions.append(len(set(cb_recs) & remaining) / K)
        cb_ndcgs.append(ndcg_at_k(cb_recs, remaining, k=K))
        cb_recs_all.append(cb_recs)
 
    # Collaborative Filtering
    cf_recs = get_cf_recommendations(query_item, top_k=K)
    if cf_recs:
        cf_precisions.append(len(set(cf_recs) & remaining) / K)
        cf_ndcgs.append(ndcg_at_k(cf_recs, remaining, k=K))
        cf_recs_all.append(cf_recs)
 
    # LightFM
    lfm_recs = get_lfm_recommendations(user, top_k=K)
    if lfm_recs:
        lfm_precisions.append(len(set(lfm_recs) & remaining) / K)
        lfm_ndcgs.append(ndcg_at_k(lfm_recs, remaining, k=K))
        lfm_recs_all.append(lfm_recs)
 
    if (i + 1) % 100 == 0:
        print(f"    Evaluated {i+1}/500 users...")
 
# Compute aggregated metrics
total_catalog = len(cb_item_ids)
 
results = {
    'Content-Based (TF-IDF)' : {
        'precision_at_k' : np.mean(cb_precisions)  if cb_precisions  else 0,
        'ndcg_at_k'      : np.mean(cb_ndcgs)       if cb_ndcgs       else 0,
        'coverage'       : compute_coverage(cb_recs_all,  total_catalog),
        'evaluated'      : len(cb_precisions),
    },
    'Item-Item CF (Implicit)' : {
        'precision_at_k' : np.mean(cf_precisions)  if cf_precisions  else 0,
        'ndcg_at_k'      : np.mean(cf_ndcgs)       if cf_ndcgs       else 0,
        'coverage'       : compute_coverage(cf_recs_all,  total_catalog),
        'evaluated'      : len(cf_precisions),
    },
    'LightFM Hybrid (WARP)' : {
        'precision_at_k' : np.mean(lfm_precisions) if lfm_precisions else 0,
        'ndcg_at_k'      : np.mean(lfm_ndcgs)      if lfm_ndcgs      else 0,
        'coverage'       : compute_coverage(lfm_recs_all, total_catalog),
        'evaluated'      : len(lfm_precisions),
    },
}
 
# ---------------------------------------------------------------------------
# SECTION 8: Print Comparison Table
# ---------------------------------------------------------------------------
print("\n[3] FULL COMPARISON TABLE (K=10):")
print(f"\n    {'Model':<30} {'P@10':>8} {'NDCG@10':>9} {'Coverage':>10} {'Users':>7}")
print(f"    {'-'*68}")
 
for model_name, metrics in results.items():
    print(
        f"    {model_name:<30} "
        f"{metrics['precision_at_k']:>8.4f} "
        f"{metrics['ndcg_at_k']:>9.4f} "
        f"{metrics['coverage']:>10.4f} "
        f"{metrics['evaluated']:>7}"
    )
 
best_model = max(results, key=lambda m: results[m]['ndcg_at_k'])
print(f"\n    Best model by NDCG@10: {best_model}")
 
# ---------------------------------------------------------------------------
# SECTION 9: 20/80 A/B User Split
# ---------------------------------------------------------------------------
print("\n[4] Building 20/80 A/B user split...")
 
events['datetime'] = pd.to_datetime(events['timestamp'], unit='ms')
cutoff_date = events['datetime'].quantile(0.8)
 
train_users     = set(events[events['datetime'] <= cutoff_date]['visitorid'].unique())
test_users      = set(events[events['datetime'] >  cutoff_date]['visitorid'].unique())
new_users       = test_users - train_users
returning_users = test_users & train_users
 
print(f"    Total test users : {len(test_users):,}")
print(f"    New users        : {len(new_users):,}  ({len(new_users)/len(test_users)*100:.1f}%)")
print(f"    Returning users  : {len(returning_users):,}  ({len(returning_users)/len(test_users)*100:.1f}%)")
 
np.random.seed(42)
new_users_list  = list(new_users)
lightfm_users   = set(np.random.choice(new_users_list, size=int(len(new_users_list)*0.20), replace=False))
remaining_new   = set(new_users_list) - lightfm_users
 
print(f"\n    ROUTING DECISIONS:")
print(f"    -> LightFM (20% of new users) : {len(lightfm_users):,} users")
print(f"    -> ALS Module 3 (returning)   : {len(returning_users):,} users")
print(f"    -> Remaining new (explore)    : {len(remaining_new):,} users")
 
# Use the actual ALS users from the CF artifact
als_users = set(cf_user_to_idx.keys())

routing_split = {
    'lightfm_users': lightfm_users,
    'als_users': als_users,
    'cutoff_date': cutoff_date,
    'train_users': train_users,
}
with open(DATA_DIR / "routing_split.pkl", "wb") as f:
    pickle.dump(routing_split, f)
print("\n    Saved -> data/routing_split.pkl  (Module 3 loads this)")
 
# ---------------------------------------------------------------------------
# SECTION 10: Visualization
# ---------------------------------------------------------------------------
print("\n[5] Plotting model comparison...")
 
model_names   = list(results.keys())
p_at_k_vals   = [results[m]['precision_at_k'] for m in model_names]
ndcg_vals     = [results[m]['ndcg_at_k']      for m in model_names]
coverage_vals = [results[m]['coverage']        for m in model_names]
colors        = ['#4C72B0', '#DD8452', '#1B3A6B']
short_names   = ['CB', 'Item-Item CF', 'LightFM']
 
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle(
    "Lab 2.4: Head-to-Head Model Evaluation (K=10)\n"
    "Content-Based vs Collaborative Filtering vs LightFM Hybrid",
    fontsize=12, fontweight='bold'
)
 
for ax, vals, title in zip(
    axes,
    [p_at_k_vals, ndcg_vals, coverage_vals],
    ["Precision@10", "NDCG@10", "Catalog Coverage"]
):
    bars = ax.bar(range(3), vals, color=colors, edgecolor='white', width=0.6)
    ax.set_xticks(range(3))
    ax.set_xticklabels(short_names, fontsize=10)
    ax.set_title(title)
    ax.set_ylabel("Score")
    for bar, v in zip(bars, vals):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.001,
            f"{v:.4f}", ha='center', va='bottom',
            fontsize=9, fontweight='bold'
        )
 
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "04_model_comparison.png", dpi=150, bbox_inches='tight')
plt.show()
print("    Saved -> output/Lab2/04_model_comparison.png")
 
# ---------------------------------------------------------------------------
# FINAL SUMMARY
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("  LAB 2.4 COMPLETE — WEEK 2 DELIVERABLE SUMMARY")
print("=" * 60)
print(f"""
  MODEL COMPARISON (K=10):
  {'Model':<30} {'P@10':>8} {'NDCG@10':>9} {'Coverage':>10}""")
 
for model_name, metrics in results.items():
    print(
        f"  {model_name:<30} "
        f"{metrics['precision_at_k']:>8.4f} "
        f"{metrics['ndcg_at_k']:>9.4f} "
        f"{metrics['coverage']:>10.4f}"
    )
 
print(f"""
  A/B ROUTING SPLIT:
  LightFM (new users, 20%)  : {len(lightfm_users):,} users
  ALS (returning, Module 3) : {len(returning_users):,} users
 
  Week 2 Deliverable:
  [OK] LightFM trained on item features + interaction matrix
  [OK] Handles new users via content feature embeddings
  [OK] Precision@10 and NDCG@10 vs CB and CF baselines
  [OK] 20/80 routing split saved -> data/routing_split.pkl
 
  Next: Module 3 — ALS Personalization Engine for returning users
""")
print("   Move to: Module 3 -> 01_als_personalization.py")
