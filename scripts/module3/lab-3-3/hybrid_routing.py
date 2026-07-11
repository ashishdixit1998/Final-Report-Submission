# =============================================================================
# MODULE 3 | LAB 3.3
# File: 03_hybrid_routing.py
# Purpose: Build a production-grade Hybrid Routing and Post-Processing layer
#          integrating ALS (returning users), LightFM (cold-start), 
#          Already-Purchased Exclusion, and MMR Diversity re-ranking.
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
from pathlib import Path
# Project root
ROOT = Path(__file__).resolve().parents[3]


DATA_DIR = ROOT / "data"

# Output folder
OUTPUT_DIR = ROOT / "output/Lab3"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False

from lightfm import LightFM

print("=" * 60)
print("  MODULE 3 | LAB 3.3")
print("  Hybrid Routing Layer")
print("  LightFM (new) + ALS (returning) + MMR + Exclusion")
print("=" * 60)

# ---------------------------------------------------------------------------
# SECTION 1: Load All Artifacts
# ---------------------------------------------------------------------------
print("\n[1] Loading all artifacts...")

with open(DATA_DIR / "routing_split.pkl",    "rb") as f: routing   = pickle.load(f)
with open(DATA_DIR / "lightfm_artifacts.pkl","rb") as f: lfm_art   = pickle.load(f)
with open(DATA_DIR / "als_artifacts.pkl",    "rb") as f: als_art   = pickle.load(f)
with open(DATA_DIR / "faiss_artifacts.pkl",  "rb") as f: faiss_art = pickle.load(f)

als_users    = routing['als_users']
lfm_users    = routing['lightfm_users']
cutoff_date  = routing['cutoff_date']

lfm_model         = lfm_art['model_hybrid']
lfm_dataset       = lfm_art['dataset']
lfm_item_features = lfm_art['item_features_matrix']
lfm_train_matrix  = lfm_art['train_matrix']

als_model       = als_art['model']
als_user_to_idx = als_art['user_to_idx']
als_item_to_idx = als_art['item_to_idx']
als_user_ids    = als_art['user_ids']
als_item_ids    = als_art['item_ids']
als_user_item   = als_art['user_item_matrix']

item_factors_norm = faiss_art['item_factors_norm']
user_factors_norm = faiss_art['user_factors_norm']
faiss_item_ids    = faiss_art['item_ids']
faiss_user_to_idx = faiss_art['user_to_idx']

events = pd.read_csv(DATA_DIR / "events.csv")
events['datetime'] = pd.to_datetime(events['timestamp'], unit='ms')

# TODO: Build a historical purchase dictionary tracking already bought items per user
# Hint: Filter events down to 'transaction', group by 'visitorid', pull 'itemid', 
# and transform into a dictionary mapping user keys to sets of item IDs.
purchases = (
    events[events["event"] == "transaction"]
    .groupby("visitorid")["itemid"]
    .apply(set)
    .to_dict()
)

if FAISS_AVAILABLE:
    try:
        faiss_index = faiss.read_index(DATA_DIR / "faiss_index.bin")
        print(f"    FAISS index loaded : {faiss_index.ntotal:,} items")
    except:
        FAISS_AVAILABLE = False
        print("    FAISS index not found — using numpy fallback")

lfm_user_map, _, lfm_item_map, _ = lfm_dataset.mapping()
lfm_item_ids_list = list(lfm_item_map.keys())
n_lfm_items = len(lfm_item_ids_list)

faiss_id_to_idx = {item_id: idx for idx, item_id in enumerate(faiss_item_ids)}

# ---------------------------------------------------------------------------
# SECTION 2: Post-Processing Functions
# ---------------------------------------------------------------------------
print("\n[2] Defining post-processing functions...")

# --- Score Normalization ---
def min_max_normalize(scores):
    """
    Scale absolute engine output values to a uniform [0, 1] interval.
    Formula: $S_{norm} = \frac{S - S_{min}}{S_{max} - S_{min}}$
    """
    # TODO: Implement min-max normalization. Convert scores to an array.
    # Handle the boundary case where s_max == s_min by returning an array of 0.5s.
    scores = np.asarray(scores, dtype=float)

    if len(scores) == 0:
        return scores

    s_min = scores.min()
    s_max = scores.max()

    if s_max == s_min:
        return np.full_like(scores, 0.5, dtype=float)

    return (scores - s_min) / (s_max - s_min)

# --- Already-Purchased Exclusion ---
def exclude_purchased(items, scores, user_id):
    """
    Filter recommendation lists to remove items that a user has already bought.
    """
    # TODO: Verify if user_id exists in the purchases dictionary.
    # Loop over the zipped items and scores, filtering out items present in the purchase history.
    # Return two unzipped collections: (filtered_items, filtered_scores).
    if user_id not in purchases:
        return items, scores

    purchased = purchases[user_id]

    filtered_items = []
    filtered_scores = []

    for item, score in zip(items, scores):
        if item not in purchased:
            filtered_items.append(item)
            filtered_scores.append(score)

    return filtered_items, filtered_scores

# --- MMR Diversity Injection ---
def mmr_rerank(items, scores, embeddings, id_to_idx, top_k=10, lam=0.5):
    """
    Maximal Marginal Relevance greedy re-ranking optimization.
    Objective: $\arg\max_{D_i \in R \setminus S} \left[ \lambda \cdot \text{Rel}(D_i) - (1 - \lambda) \cdot \max_{D_j \in S} \text{Sim}(D_i, D_j) \right]$
    """
    items  = list(items)
    scores = list(scores)

    if len(items) == 0:
        return [], []

    # TODO: Normalize scores using your min_max_normalize tool
    norm_scores = min_max_normalize(scores)

    # TODO: Extract baseline matrices containing (item_id, norm_score, normalized_embedding_vector)
    # Ensure items exist in your id_to_idx dictionary mapping boundaries
    valid = []

    for item, score in zip(items, norm_scores):
        if item in id_to_idx:
            idx = id_to_idx[item]
            valid.append(
                (
                    item,
                    score,
                    embeddings[idx]
                )
            )

    if not valid:
        return items[:top_k], scores[:top_k]

    selected_items  = []
    selected_scores = []
    selected_embs   = []
    remaining       = list(range(len(valid)))

    for _ in range(min(top_k, len(valid))):
        if not remaining:
            break

        if not selected_embs:
            # TODO: Pick the first entry based on maximum relevance score alone
            best = max(
            remaining,
            key=lambda i: valid[i][1]
        )
        else:
            # TODO: Convert selected_embs into a numpy matrix array
            sel_matrix = np.array(selected_embs)
            mmr_vals   = []
            
            # TODO: Iterate over remaining index options, calculate relevance weights, 
            # calculate similarity products against the sel_matrix, and solve the MMR optimization formula
            for i in remaining:
                rel  = lam * valid[i][1] # Hint: lam * normalized relevance score
                sims = valid[i][2] @ sel_matrix.T # Hint: embedding vector matrix dot product transposition
                div  = (1 - lam) * np.max(sims) # Hint: (1 - lam) * maximum similarity value
                mmr_vals.append(rel - div)
                
            best = remaining[np.argmax(mmr_vals)]

        # Append chosen targets to tracking lists
        selected_items.append(valid[best][0])
        selected_scores.append(valid[best][1])
        selected_embs.append(valid[best][2])
        remaining.remove(best)

    return selected_items, selected_scores


# ---------------------------------------------------------------------------
# SECTION 3: Engine Recommendation Functions
# ---------------------------------------------------------------------------
print("\n[3] Defining engine recommendation functions...")

def get_als_recs(user_id, top_k=30):
    """
    Query the personalized ALS embedding space using FAISS index lookups or NumPy fallback.
    """
    # TODO: Check if user_id exists in faiss_user_to_idx index parameters.
    # Isolate user vector embeddings, call .search() or perform inner products using item_factors_norm.
    # Return lists: (retrieved_item_ids, scores)
    if user_id not in faiss_user_to_idx:
        return [], []

    user_idx = faiss_user_to_idx[user_id]
    user_vector = user_factors_norm[user_idx].reshape(1, -1)

    if FAISS_AVAILABLE:
        scores, indices = faiss_index.search(user_vector.astype(np.float32), top_k)

        items = [faiss_item_ids[i] for i in indices[0] if i >= 0]
        scores = scores[0][:len(items)]

    else:
        sims = item_factors_norm @ user_vector.T
        sims = sims.flatten()

        order = np.argsort(-sims)[:top_k]

        items = [faiss_item_ids[i] for i in order]
        scores = sims[order]

    return items, scores.tolist()

def get_lfm_recs(user_id, top_k=30):
    """
    Generate predictions for cold-start or low-engagement profiles using LightFM Hybrid.
    """
    # TODO: Verify user mapping parameters, build prediction range frameworks, 
    # invoke lfm_model.predict() utilizing item_features=lfm_item_features, 
    # sort descending, and isolate top_k entries.
    if user_id not in lfm_user_map:
        return [], []

    user_idx = lfm_user_map[user_id]

    scores = lfm_model.predict(
        user_idx,
        np.arange(n_lfm_items),
        item_features=lfm_item_features
    )

    order = np.argsort(-scores)[:top_k]

    items = [lfm_item_ids_list[i] for i in order]
    values = scores[order]

    return items, values.tolist()


# ---------------------------------------------------------------------------
# SECTION 4: The Routing Function
# ---------------------------------------------------------------------------
print("\n[4] Defining routing function...")

def route_and_recommend(user_id, interaction_threshold=3,
                        top_k=10, lam_mmr=0.5,
                        apply_exclusion=True, apply_mmr=True):
    """
    Executes core architectural engine routing rules and controls the post-processing pipeline.
    """
    result = {
        'user_id'    : user_id,
        'engine'     : None,
        'raw_count'  : 0,
        'final_recs' : [],
        'scores'     : [],
        'excluded'   : 0,
        'latency_ms' : 0,
    }

    t0 = time.perf_counter()

    # --- STEP 1: ROUTING DECISION ---
    # TODO: Retrieve the number of training interactions (nnz) recorded for this user inside als_user_item.
    # Configure a conditional boolean 'use_als' requiring user registration and nnz >= interaction_threshold.
    use_als = False

    if user_id in als_user_to_idx:

        idx = als_user_to_idx[user_id]

        interactions = als_user_item[idx].nnz

        use_als = interactions >= interaction_threshold

    # --- STEP 2: GET RAW RECOMMENDATIONS ---
    # TODO: Route the request. Call get_als_recs if use_als is true, otherwise branch to get_lfm_recs.
    # Request a candidate multiplier size (e.g., top_k * 3) to allow downstream filtering space.
    if use_als:

        result["engine"] = "ALS"

        raw_items, raw_scores = get_als_recs(
            user_id,
            top_k * 3
        )

    else:

        result["engine"] = "LightFM"

        raw_items, raw_scores = get_lfm_recs(
            user_id,
            top_k * 3
        )

    result['raw_count'] = len(raw_items)
    if not raw_items:
        result['latency_ms'] = (time.perf_counter() - t0) * 1000
        return result

    # --- STEP 3: ALREADY-PURCHASED EXCLUSION ---
    if apply_exclusion:
        n_before = len(raw_items)
        raw_items, raw_scores = exclude_purchased(raw_items, raw_scores, user_id)
        raw_items, raw_scores = list(raw_items), list(raw_scores)
        result['excluded'] = n_before - len(raw_items)

    if not raw_items:
        result['latency_ms'] = (time.perf_counter() - t0) * 1000
        return result

    # --- STEP 4: SCORE NORMALIZATION ---
    norm_scores = min_max_normalize(raw_scores).tolist()

    # --- STEP 5: MMR DIVERSITY RERANKING ---
    if apply_mmr and len(raw_items) > top_k:
        # TODO: Call mmr_rerank passing extracted variables, embeddings, maps, top_k targets, and lambda constraints
        final_items, final_scores = mmr_rerank(
        raw_items,
        norm_scores,
        item_factors_norm,
        faiss_id_to_idx,
        top_k=top_k,
        lam=lam_mmr
    )
    else:
        final_items  = raw_items[:top_k]
        final_scores = norm_scores[:top_k]

    result['final_recs'] = final_items
    result['scores']     = final_scores
    result['latency_ms'] = (time.perf_counter() - t0) * 1000

    return result


# ---------------------------------------------------------------------------
# SECTION 5: Test the Routing Layer
# ---------------------------------------------------------------------------
print("\n[5] Testing routing layer...")

als_test = [u for u in list(als_users)[:20] if u in faiss_user_to_idx and faiss_user_to_idx[u] < len(user_factors_norm)][:3]
lfm_test = [u for u in list(lfm_users)[:20] if u in lfm_user_map][:3]

# TODO: Execute route_and_recommend calls on sample test arrays (als_test and lfm_test) to print sample performance metrics
print("\nALS Users")

for user in als_test:

    result = route_and_recommend(user)

    print(
        f"{user} | "
        f"{result['engine']} | "
        f"{len(result['final_recs'])} recs | "
        f"{result['latency_ms']:.2f} ms"
    )

print("\nLightFM Users")

for user in lfm_test:

    result = route_and_recommend(user)

    print(
        f"{user} | "
        f"{result['engine']} | "
        f"{len(result['final_recs'])} recs | "
        f"{result['latency_ms']:.2f} ms"
    )

# ---------------------------------------------------------------------------
# SECTION 6: MMR Diversity Analysis
# ---------------------------------------------------------------------------
print("\n[6] MMR diversity analysis...")

def intra_list_diversity(items, embeddings, id_to_idx):
    """Calculates average distance metrics across item sets: $D(L) = \frac{2}{|L|(|L|-1)}\sum_{i}\sum_{j>i}(1 - \text{Sim}(i,j))$"""
    embs = [embeddings[id_to_idx[it]] for it in items if it in id_to_idx]
    if len(embs) < 2: return 0.0
    embs = np.array(embs)
    sims = embs @ embs.T
    n    = len(embs)
    pairs = [(1 - sims[i,j]) for i in range(n) for j in range(i+1, n)]
    return np.mean(pairs) if pairs else 0.0

if als_test:
    test_user = als_test[0]
    # TODO: Run contrast calls isolating a test request with apply_mmr=False vs apply_mmr=True.
    # Pass outputs through intra_list_diversity to measure performance gains.
    test_user = als_test[0]

    no_mmr = route_and_recommend(
        test_user,
        apply_mmr=False,
        top_k=10
    )

    with_mmr = route_and_recommend(
        test_user,
        apply_mmr=True,
        top_k=10
    )

    div_before = intra_list_diversity(
        no_mmr["final_recs"],
        item_factors_norm,
        faiss_id_to_idx
    )

    div_after = intra_list_diversity(
        with_mmr["final_recs"],
        item_factors_norm,
        faiss_id_to_idx
    )

    print(f"User : {test_user}")
    print(f"Diversity before MMR : {div_before:.3f}")
    print(f"Diversity after MMR  : {div_after:.3f}")


# ---------------------------------------------------------------------------
# SECTION 7: Routing Statistics Across Sample Users
# ---------------------------------------------------------------------------
print("\n[7] Routing statistics across sample users...")

sample_all = ([u for u in list(als_users)[:50] if u in faiss_user_to_idx and faiss_user_to_idx.get(u, 99999) < len(user_factors_norm)] +
              [u for u in list(lfm_users)[:50] if u in lfm_user_map])[:100]

# TODO: Iterate over all elements inside sample_all arrays, gather metrics, 
# track engine distribution shares, and calculate latency statistics.
results = []

for user in sample_all:

    results.append(route_and_recommend(user))

als_count = sum(r["engine"] == "ALS" for r in results)

lfm_count = sum(r["engine"] == "LightFM" for r in results)

latencies = [r["latency_ms"] for r in results]

exclusions = [r["excluded"] for r in results]

lat_mean = np.mean(latencies)

lat_p95 = np.percentile(latencies, 95)

lat_p99 = np.percentile(latencies, 99)

print(f"ALS Requests      : {als_count}")
print(f"LightFM Requests  : {lfm_count}")
print(f"Average Latency   : {lat_mean:.2f} ms")
print(f"P95 Latency       : {lat_p95:.2f} ms")
print(f"P99 Latency       : {lat_p99:.2f} ms")

# ---------------------------------------------------------------------------
# SECTION 8: Visualizations
# ---------------------------------------------------------------------------
print("\n[8] Plotting routing analysis...")

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("Lab 3.3: Hybrid Routing Layer Analysis", fontsize=12, fontweight='bold')

# --- Plot 1: Routing Shares Pie Layout ---
# TODO: Plot a pie chart tracing engine traffic breakdowns between ALS and LightFM partitions
# Hint: Use axes[0].pie([als_count, lfm_count], labels=[...], autopct='%1.1f%%')
axes[0].pie(
    [als_count, lfm_count],
    labels=["ALS", "LightFM"],
    autopct="%1.1f%%",
    startangle=90
)

axes[0].set_title("Routing Share")

# --- Plot 2: End-to-End Latency Profiles ---
# TODO: Draw a latency tracking request frequency histogram on axes[1]
# Mark the calculated p99 vertical latency threshold barrier line using axes[1].axvline()
axes[1].hist(
    latencies,
    bins=20
)

axes[1].axvline(
    lat_p99,
    linestyle="--",
    linewidth=2,
    label=f"P99={lat_p99:.2f} ms"
)

axes[1].legend()

axes[1].set_title("Latency Distribution")

axes[1].set_xlabel("Latency (ms)")

axes[1].set_ylabel("Requests")

# --- Plot 3: Exclusion Volume Trackers ---
# TODO: Render a frequency bar histogram showing counts of items removed per request due to prior purchases
# Hint: Use axes[2].hist(exclusions, bins=...)
axes[2].hist(
    exclusions,
    bins=10
)

axes[2].set_title("Purchased Item Exclusions")

axes[2].set_xlabel("Items Removed")

axes[2].set_ylabel("Requests")

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "03_routing_analysis.png", dpi=150, bbox_inches='tight')
plt.show()


# ---------------------------------------------------------------------------
# SECTION 9: Save Routing Artifacts
# ---------------------------------------------------------------------------
print("\n[9] Saving routing artifacts...")

# TODO: Export calculated summaries, latencies vectors, lists, and metrics down to a serialization storage target
# Target Path: "data/routing_artifacts.pkl"
routing_artifacts = {
    "results": results,
    "als_count": als_count,
    "lfm_count": lfm_count,
    "latencies": latencies,
    "lat_mean": lat_mean,
    "lat_p95": lat_p95,
    "lat_p99": lat_p99,
    "exclusions": exclusions
}

with open(DATA_DIR / "routing_artifacts.pkl", "wb") as f:
    pickle.dump(routing_artifacts, f)

print("    Saved -> data/routing_artifacts.pkl")
print("    Move to: 04_ab_protocol.py")
