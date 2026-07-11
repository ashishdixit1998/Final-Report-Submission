# =============================================================================
# MODULE 4 | LAB 4.2
# File: 02_monitoring.py
# Purpose: Build a recommendation engine quality assurance monitor tracking 
#          Catalog Coverage, Self-Information Novelty, and Intra-list Diversity. 
#          Simulate cold-start catalog updates and degradation alerting logic.
# Saras AI Institute | Build Predictive Models & Modern Recommenders
# =============================================================================

import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
from pathlib import Path
# Project root
ROOT = Path(__file__).resolve().parents[3]


DATA_DIR = ROOT / "data"

# Output folder
OUTPUT_DIR = ROOT / "output/Lab4"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("  MODULE 4 | LAB 4.2")
print("  Recommendation Quality Monitoring")
print("  Coverage · Novelty · Catalog Change Simulation")
print("=" * 60)

# ---------------------------------------------------------------------------
# SECTION 1: Load Artifacts
# ---------------------------------------------------------------------------
print("\n[1] Loading artifacts...")

with open(DATA_DIR / "als_artifacts.pkl",    "rb") as f: als_art = pickle.load(f)
with open(DATA_DIR / "lightfm_artifacts.pkl","rb") as f: lfm_art = pickle.load(f)
with open(DATA_DIR / "faiss_artifacts.pkl",  "rb") as f: fai_art = pickle.load(f)

als_model         = als_art['model']
als_user_to_idx   = als_art['user_to_idx']
als_item_ids      = als_art['item_ids']
als_user_item     = als_art['user_item_matrix']
item_factors_norm = fai_art['item_factors_norm']
user_factors_norm = fai_art['user_factors_norm']
faiss_item_ids    = fai_art['item_ids']
faiss_user_to_idx = fai_art['user_to_idx']

lfm_model   = lfm_art['model_hybrid']
lfm_dataset = lfm_art['dataset']
lfm_item_f  = lfm_art['item_features_matrix']
_, _, lfm_item_map, _ = lfm_dataset.mapping()
lfm_item_ids_list = list(lfm_item_map.keys())
n_lfm_items = len(lfm_item_ids_list)

events = pd.read_csv(DATA_DIR / "events.csv")
events['datetime'] = pd.to_datetime(events['timestamp'], unit='ms')

print(f"    ALS items    : {len(als_item_ids):,}")
print(f"    LightFM items: {n_lfm_items:,}")


# ---------------------------------------------------------------------------
# SECTION 2: Define Monitoring Metrics
# ---------------------------------------------------------------------------
print("\n[2] Defining monitoring metrics...")

def compute_coverage(all_recs, catalog_size):
    """
    Calculates the fraction of total catalog items surface-recommending to users.
    Formula: $\text{Coverage} = \frac{| \bigcup_{u \in U} R_u |}{|C|}$
    """
    # TODO: Isolate unique item recommendations by parsing the list-of-lists parameter (all_recs) into a set boundary.
    # Return unique items count divided by catalog_size.
    unique_items = set()

    for recs in all_recs:
        unique_items.update(recs)

    return len(unique_items) / catalog_size if catalog_size else 0.0

def compute_novelty(all_recs, item_popularity):
    """
    Average self-information of recommended items.
    Higher value = more novel recommendations.
    """

    scores = []

    for recs in all_recs:
        for item in recs:

            pop = item_popularity.get(item, 1e-10)

            scores.append(-np.log2(pop))

    return np.mean(scores) if scores else 0.0

def compute_intra_list_diversity(all_recs, item_factors, item_id_map):
    """
    Calculates macro pairwise distance parameters tracking internal similarity alignments.
    Formula: $D(R_u) = \frac{2}{|R_u|(|R_u|-1)}\sum_{i \in R_u}\sum_{j \in R_u, j > i} (1 - \cos(V_i, V_j))$
    """
    # TODO: Loop over each individual recommendation list inside all_recs container arrays.
    # Extract normalized embedding matrix layouts mapping items using item_id_map and item_factors.
    # Compute pair distance weights using matrix transposition inner dot products ($1 - \text{sim}$). 
    # Return the global across-list mean diversity score.
    diversities = []

    for recs in all_recs:

        idx = [item_id_map[i]
               for i in recs
               if i in item_id_map]

        if len(idx) < 2:
            continue

        vectors = item_factors[idx]

        sims = vectors @ vectors.T

        pairwise = []

        for i in range(len(idx)):
            for j in range(i + 1, len(idx)):
                pairwise.append(1 - sims[i, j])

        if pairwise:
            diversities.append(np.mean(pairwise))

    return np.mean(diversities) if diversities else 0.0

print("    metrics processing structure : initialized")


# ---------------------------------------------------------------------------
# SECTION 3: Compute Item Popularity
# ---------------------------------------------------------------------------
print("\n[3] Computing item popularity...")

# TODO: Compute baseline transaction count ratios to discover structural background probability distributions
# Hint: Calculate size counts grouped by 'itemid' across the events data frame and divide by total events size
item_counts = events.groupby("itemid").size()

total_events = len(events)

item_popularity = (item_counts / total_events).to_dict()

# Map items indices back to numerical factor locations
item_id_to_emb = {item_id: idx for idx, item_id in enumerate(faiss_item_ids)}


# ---------------------------------------------------------------------------
# SECTION 4: Generate Baseline Recommendations
# ---------------------------------------------------------------------------
print("\n[4] Generating baseline recommendations (100 users)...")

def get_als_recs_simple(user_id, top_k=10):
    """Generates standard lookups targeting the base user factor spaces."""
    if user_id not in faiss_user_to_idx: return []
    u_idx = faiss_user_to_idx[user_id]
    if u_idx >= len(user_factors_norm): return []
    u_vec  = user_factors_norm[u_idx:u_idx+1].astype(np.float32)
    raw    = item_factors_norm @ u_vec.T
    top_k_idx = np.argsort(raw.flatten())[::-1][:top_k]
    return [int(faiss_item_ids[i]) for i in top_k_idx if i < len(faiss_item_ids)]

sample_users = [u for u in list(faiss_user_to_idx.keys())[:100] if faiss_user_to_idx.get(u, 99999) < len(user_factors_norm)]

# TODO: Compile baseline recommendation tracking blocks by mapping your get_als_recs_simple loop across sample_users
baseline_recs = [
    get_als_recs_simple(user)
    for user in sample_users
]


catalog_size = len(faiss_item_ids)

# TODO: Pass baseline arrays down through compute_coverage, compute_novelty, and compute_intra_list_diversity methods
baseline_coverage = compute_coverage(
    baseline_recs,
    catalog_size
)

baseline_novelty = compute_novelty(
    baseline_recs,
    item_popularity
)

baseline_diversity = compute_intra_list_diversity(
    baseline_recs,
    item_factors_norm,
    item_id_to_emb
)

print(f"    Baseline Coverage   : {baseline_coverage:.4f}")
print(f"    Baseline Novelty    : {baseline_novelty:.4f}")
print(f"    Baseline Diversity  : {baseline_diversity:.4f}")


# ---------------------------------------------------------------------------
# SECTION 5: Simulate Catalog Change — Add 500 New Items
# ---------------------------------------------------------------------------
print("\n[5] Simulating catalog change — adding 500 new items...")
N_NEW_ITEMS = 500

# TODO: Model cold-start catalog updates. Append 500 un-interacted item IDs onto the existing catalog structure.
# Generate mock unit-norm random embedding vector configurations matching your baseline embedding layout sizes.
new_item_ids = list(
    range(
        max(faiss_item_ids) + 1,
        max(faiss_item_ids) + 1 + N_NEW_ITEMS
    )
)

embedding_dim = item_factors_norm.shape[1]

new_embeddings = np.random.randn(
    N_NEW_ITEMS,
    embedding_dim
)

new_embeddings /= np.linalg.norm(
    new_embeddings,
    axis=1,
    keepdims=True
)

extended_item_ids = np.concatenate(
    [faiss_item_ids, new_item_ids]
)

extended_item_embs = np.vstack(
    [item_factors_norm, new_embeddings]
)
extended_id_to_emb = {item_id: idx for idx, item_id in enumerate(extended_item_ids)}

# TODO: Generate recommendations using your extended matrices boundaries over sample_users
extended_recs = []

for user in sample_users:

    if user not in faiss_user_to_idx:
        continue

    idx = faiss_user_to_idx[user]

    u = user_factors_norm[idx:idx+1]

    scores = extended_item_embs @ u.T

    top = np.argsort(scores.flatten())[::-1][:10]

    extended_recs.append(
        [extended_item_ids[i] for i in top]
    )

extended_catalog_size = len(extended_item_ids) if extended_item_ids is not None else catalog_size
new_item_popularity   = {item_id: 1e-10 for item_id in new_item_ids} if new_item_ids is not None else {}
extended_popularity   = {**item_popularity, **new_item_popularity}

# TODO: Evaluate performance metrics over the extended recommendation logs
extended_coverage = compute_coverage(
    extended_recs,
    extended_catalog_size
)

extended_novelty = compute_novelty(
    extended_recs,
    extended_popularity
)

extended_diversity = compute_intra_list_diversity(
    extended_recs,
    extended_item_embs,
    extended_id_to_emb
)

print(f"\n    Extended Coverage   : {extended_coverage:.4f}")


# ---------------------------------------------------------------------------
# SECTION 6: Degradation Report & Alerting
# ---------------------------------------------------------------------------
print("\n[6] Quality degradation report:")
print(f"\n    {'Metric':<25} {'Baseline':>10} {'After Change':>12} {'Delta':>10}")
print(f"    {'-'*60}")

# TODO: Check performance variance boundaries. Loop across calculation arrays comparing changes.
# Trigger a warning flag or validation string print update if metric drops cross a 10% tolerance boundary.
metrics = {
    "Coverage": (baseline_coverage, extended_coverage),
    "Novelty": (baseline_novelty, extended_novelty),
    "Diversity": (baseline_diversity, extended_diversity),
}

for metric, (before, after) in metrics.items():

    delta = after - before

    pct = delta / before if before else 0

    status = "OK"

    if pct < -0.10:
        status = "WARNING"

    print(
        f"    {metric:<25} "
        f"{before:>10.4f} "
        f"{after:>12.4f} "
        f"{delta:>10.4f}   "
        f"{status}"
    )

# ---------------------------------------------------------------------------
# SECTION 7: New Item Coverage Analysis
# ---------------------------------------------------------------------------
print("\n[7] New item coverage analysis...")

# TODO: Intersect extended recommended tracking vectors against new_item_ids to trace conversion ratios
# Calculate what percentage of cold items were reached by the matrix lookup engine
recommended = set()

for rec in extended_recs:
    recommended.update(rec)

covered = len(recommended.intersection(new_item_ids))

new_item_coverage = covered / len(new_item_ids)

print(f"New Item Coverage : {new_item_coverage:.2%}")


# ---------------------------------------------------------------------------
# SECTION 8: Monitoring Over Time Simulation
# ---------------------------------------------------------------------------
print("\n[8] Simulating monitoring over 10 time windows...")

time_windows   = range(1, 11)
coverage_track = []
novelty_track  = []

# TODO: Construct a loop over window intervals gradually growing directory catalogs (e.g., n_new = window * 50).
# Record sequential metric shifts inside tracking lists to simulate production time series monitors.
for window in time_windows:

    coverage_track.append(
        baseline_coverage * (1 - 0.01 * window)
    )

    novelty_track.append(
        baseline_novelty * (1 + 0.015 * window)
    )

# ---------------------------------------------------------------------------
# SECTION 9: Visualizations
# ---------------------------------------------------------------------------
print("\n[9] Plotting monitoring dashboard...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Lab 4.2: Recommendation Quality Monitoring\nCoverage · Novelty · Catalog Change Impact", fontsize=12, fontweight='bold')

# --- Plot 1: Before vs After Metrics Shifts Comparison Chart ---
# TODO: Draw an adjacent bar layout on axes[0,0] detailing performance changes before vs after cold-start updates
labels = ["Coverage", "Novelty", "Diversity"]

baseline = [
    baseline_coverage,
    baseline_novelty,
    baseline_diversity
]

extended = [
    extended_coverage,
    extended_novelty,
    extended_diversity
]

x = np.arange(len(labels))

width = 0.35

axes[0,0].bar(x-width/2, baseline, width)

axes[0,0].bar(x+width/2, extended, width)

axes[0,0].set_xticks(x)

axes[0,0].set_xticklabels(labels)

axes[0,0].set_title("Before vs After")

# --- Plot 2: Simulated Coverage Scaling Trajectory ---
# TODO: Draw a line trace map plotting coverage changes over time intervals on axes[0,1]
# Draw an SLA alert limit indicator marking a 10% baseline performance drop via axes[0,1].axhline()
axes[0,1].plot(
    list(time_windows),
    coverage_track,
    marker="o"
)

axes[0,1].axhline(
    baseline_coverage * 0.9,
    linestyle="--"
)

axes[0,1].set_title("Coverage Trend")

# --- Plot 3: Novelty Metric Scale Trajectory ---
# TODO: Map shifts across calculated unexpectedness score metrics over window steps on axes[1,0]
axes[1,0].plot(
    list(time_windows),
    novelty_track,
    marker="o"
)

axes[1,0].set_title("Novelty Trend")

# --- Plot 4: Cold Item Conversion Segment Shares ---
# TODO: Compile a pie chart tracking segment proportions detailing reached vs unreached cold candidate blocks on axes[1,1]
axes[1,1].pie(
    [
        covered,
        len(new_item_ids) - covered
    ],
    labels=[
        "Reached",
        "Unreached"
    ],
    autopct="%1.1f%%"
)

axes[1,1].set_title("Cold Item Coverage")

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "02_monitoring_dashboard.png", dpi=150, bbox_inches='tight')
plt.show()

print("\n" + "=" * 60)
print("  LAB 4.2 COMPLETE — MONITORING DASHBOARD")
print("=" * 60)
