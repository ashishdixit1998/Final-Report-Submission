# =============================================================================
# MODULE 3 | LAB 3.4
# File: 04_ab_protocol.py
# Purpose: Establish an offline A/B testing simulation protocol, compute
#          statistical power boundaries, sample sizing, and run hypothesis tests.
# Saras AI Institute | Build Predictive Models & Modern Recommenders
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import pickle
import warnings
import hashlib
warnings.filterwarnings('ignore')

print("=" * 60)
print("  MODULE 3 | LAB 3.4")
print("  Offline A/B Evaluation Protocol")
print("  Hypothesis · Sample Size · Metrics · Guardrails")
print("=" * 60)

# ---------------------------------------------------------------------------
# SECTION 1: Load Artifacts
# ---------------------------------------------------------------------------
print("\n[1] Loading artifacts...")

with open("../../../data/routing_split.pkl",    "rb") as f: routing   = pickle.load(f)
with open("../../../data/als_artifacts.pkl",    "rb") as f: als_art   = pickle.load(f)
with open("../../../data/lightfm_artifacts.pkl","rb") as f: lfm_art   = pickle.load(f)
with open("../../../data/routing_artifacts.pkl","rb") as f: rt_art    = pickle.load(f)

events    = pd.read_csv("../../../data/events.csv")
events['datetime'] = pd.to_datetime(events['timestamp'], unit='ms')
purchases = events[events['event'] == 'transaction'].copy()

print(f"    Total events          : {len(events):,}")
print(f"    Purchase events       : {len(purchases):,}")


# ---------------------------------------------------------------------------
# SECTION 2: Randomization Strategy
# ---------------------------------------------------------------------------
print("\n[2] Randomization Strategy")
print("=" * 55)

def assign_group(user_id, salt="ab_v1"):
    """
    Deterministically maps a user ID into Control or Treatment buckets.
    Ensures group stickiness across runtime iterations without database storage.
    """
    # TODO: Concatenate salt and user_id into a single string token, calculate its hash string integer representation,
    # and check if hash % 100 is less than 50 to return "control", otherwise "treatment".
    token = f"{salt}_{user_id}"
    h = int(hashlib.md5(token.encode()).hexdigest(), 16)

    if h % 100 < 50:
        return "control"
    else:
        return "treatment"

all_users = events['visitorid'].unique()

# TODO: Build an assignment mapping registry dictionary containing {user_id: assign_group(user_id)} pairs
groups = {u: assign_group(u) for u in all_users}

control_users   = [u for u, g in groups.items() if g == "control"]
treatment_users = [u for u, g in groups.items() if g == "treatment"]

print(f"  Total users     : {len(all_users):,}")
print(f"  Control  (50%)  : {len(control_users):,}")
print(f"  Treatment (50%) : {len(treatment_users):,}")


# ---------------------------------------------------------------------------
# SECTION 3: Sample Size Calculation
# ---------------------------------------------------------------------------
print("\n[3] Sample Size Calculation")
print("=" * 55)

baseline_ndcg = float(als_art.get('best_ndcg', 0.017))
baseline_ndcg = max(baseline_ndcg, 0.01)

MDE_RELATIVE  = 0.05
# TODO: Compute absolute minimum detectable effect parameter size matching your relative boundaries
MDE_ABSOLUTE = baseline_ndcg * MDE_RELATIVE
alpha         = 0.05
power         = 0.80

# TODO: Calculate standard statistical deviation z-score intervals for the specified alpha and power boundaries
# Hint: Use stats.norm.ppf() for upper tails -> (1 - alpha / 2) and (power)
z_alpha = stats.norm.ppf(1 - alpha / 2)
z_beta = stats.norm.ppf(power)

print(f"  Baseline NDCG@10      : {baseline_ndcg:.4f}")
print(f"  MDE (5% relative)     : {MDE_ABSOLUTE:.4f} absolute")

# TODO: Formulate variance parameters and map the sample sizing equation to isolate requirements per group
# Mathematical Formula Hint: $n = \frac{2 \cdot (Z_\alpha + Z_\beta)^2 \cdot \sigma^2}{\delta^2}$
# Where variance ($\sigma^2$) can be approximated via Bernoulli properties: baseline * (1 - baseline)
variance = baseline_ndcg * (1 - baseline_ndcg)

n_per_group = int(
    np.ceil(
        2 * ((z_alpha + z_beta) ** 2) * variance /
        (MDE_ABSOLUTE ** 2)
    )
)
total_required = n_per_group * 2 if n_per_group is not None else 0

print(f"  Required per group  : {n_per_group:,}" if n_per_group is not None else "N/A")


# ---------------------------------------------------------------------------
# SECTION 4: NDCG@K Metric Implementation
# ---------------------------------------------------------------------------
def ndcg_at_k(recommended, relevant, k=10):
    """
    Calculates Normalized Discounted Cumulative Gain at rank position k.
    Formula: $NDCG@K = \frac{DCG@K}{IDCG@K}$
    """
    rec_k = recommended[:k]
    # TODO: Implement the Discounted Cumulative Gain calculation loop over top k item values
    # Hint: Use 1.0 / np.log2(i + 2) if item belongs to relevant set
    
    # TODO: Implement Ideal Discounted Cumulative Gain loop based on maximum possible ideal matches
    dcg = 0.0

    for i, item in enumerate(rec_k):
        if item in relevant:
            dcg += 1.0 / np.log2(i + 2)

    ideal_hits = min(len(relevant), k)

    idcg = 0.0

    for i in range(ideal_hits):
        idcg += 1.0 / np.log2(i + 2)

    return dcg / idcg if idcg > 0 else 0.0


# ---------------------------------------------------------------------------
# SECTION 5: Offline Metric Simulation Loop
# ---------------------------------------------------------------------------
print("\n[4] Offline A/B metric simulation...")

cutoff     = routing['cutoff_date']
train_ev   = events[events['datetime'] <= cutoff]
test_purch = purchases[purchases['datetime'] > cutoff]

user_ix_counts = train_ev.groupby('visitorid').size().to_dict()

als_user_to_idx = als_art['user_to_idx']
als_item_ids    = als_art['item_ids']
als_model       = als_art['model']
als_user_item   = als_art['user_item_matrix']

lfm_user_map, _, lfm_item_map, _ = lfm_art['dataset'].mapping()
lfm_item_ids_list = list(lfm_item_map.keys())
n_lfm_items = len(lfm_item_ids_list)
lfm_model   = lfm_art['model_hybrid']
lfm_item_f  = lfm_art['item_features_matrix']

test_users_with_p = test_purch['visitorid'].unique()[:200]

control_ndcgs   = []
treatment_ndcgs = []

print(f"    Simulating on {len(test_users_with_p)} test users...")

for user_id in test_users_with_p:
    group    = groups.get(user_id, 'control')
    relevant = set(test_purch[test_purch['visitorid'] == user_id]['itemid'].values)

    if not relevant:
        continue

    n_ix  = user_ix_counts.get(user_id, 0)
    
    # TODO: Determine if conditions permit querying the returning-user engine (ALS) for treatment records
    # Constraints: group must equal treatment, user must exist in conversion map, interactions >= 3
    use_als = (
    group == "treatment"
    and user_id in als_user_to_idx
    and n_ix >= 3
    )

    recs = []
    if use_als:
        # TODO: Execute an internal model.recommend query lookup to fetch candidate IDs for the current user index
        user_idx = als_user_to_idx[user_id]

        ids, _ = als_model.recommend(
            user_idx,
            als_user_item[user_idx],
            N=10,
            filter_already_liked_items=True
        )

        recs = [als_item_ids[i] for i in ids]
    elif user_id in lfm_user_map:
        # TODO: Fall back to generating raw predictions via lfm_model.predict across item matrices
        u_idx = lfm_user_map[user_id]
        scores = lfm_model.predict(
            u_idx,
            np.arange(n_lfm_items),
            item_features=lfm_item_f
        )

        top = np.argsort(scores)[::-1][:10]

        recs = [lfm_item_ids_list[i] for i in top]

    if not recs:
        continue

    # TODO: Calculate NDCG@10 scores using your implemented ndcg_at_k function,
    # and append outputs to the matching group metrics tracking array (control_ndcgs or treatment_ndcgs)
    score = ndcg_at_k(recs, relevant, k=10)
    if group == "control":
        control_ndcgs.append(score)
    else:
        treatment_ndcgs.append(score)


# ---------------------------------------------------------------------------
# SECTION 6: Statistical Test
# ---------------------------------------------------------------------------
print("\n[5] Statistical Test Results")
print("=" * 55)

if control_ndcgs and treatment_ndcgs:
    # TODO: Extract aggregate distribution statistics across simulation parameters
    ctrl_mean  = np.mean(control_ndcgs)
    treat_mean = np.mean(treatment_ndcgs)
    diff = treat_mean - ctrl_mean
    # TODO: Compute relative lift metric ratios between treatment and control means
    rel_lift   = (
    (diff / ctrl_mean) * 100
    if ctrl_mean > 0 else 0
    )
    
    # TODO: Execute an independent two-sample t-test across metrics containers using stats.ttest_ind()
    # Pull out structural stats indicators and calculate the one-tailed p-value
    t_stat, p_two = stats.ttest_ind(
    treatment_ndcgs,
    control_ndcgs,
    equal_var=False
)
    p_one = p_two / 2 if t_stat > 0 else 1 - (p_two / 2)

    print(f"  Control  (LightFM-only)  : NDCG@10 = {ctrl_mean:.4f}")
    print(f"  Treatment (Hybrid Engine): NDCG@10 = {treat_mean:.4f}")
    print(f"  Relative lift            : {rel_lift:+.2f}%")
    print(f"  p-value                  : {p_one:.4f}")
else:
    ctrl_mean, treat_mean, rel_lift, p_one = 0.017, 0.021, 23.5, 0.031


# ---------------------------------------------------------------------------
# SECTION 7: Visualizations
# ---------------------------------------------------------------------------
print("\n[6] Plotting A/B protocol results...")

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("Lab 3.4: Offline A/B Evaluation Protocol", fontsize=12, fontweight='bold')

# --- Plot 1: NDCG Distribution Contrast ---
# TODO: Build an overlaid histogram layout charting control vs treatment NDCG frequencies on axes[0]
axes[0].hist(
    control_ndcgs,
    bins=20,
    alpha=0.6,
    label="Control"
)

axes[0].hist(
    treatment_ndcgs,
    bins=20,
    alpha=0.6,
    label="Treatment"
)

axes[0].set_title("NDCG Distribution")
axes[0].set_xlabel("NDCG@10")
axes[0].set_ylabel("Frequency")
axes[0].legend()

# --- Plot 2: Sample Sizing Function Curvatures ---
# TODO: Plot the generated sample size requirements progression line across the linear mde_range on axes[1]
# Draw tracking indicator baseline markers tracking configured MDE boundaries using axes[1].axvline()
mde_range = np.linspace(1, 15, 40)

sample_sizes = []

for m in mde_range:
    delta = baseline_ndcg * (m / 100)

    n = (
        2 * ((z_alpha + z_beta) ** 2)
        * variance
        / (delta ** 2)
    )

    sample_sizes.append(n)

axes[1].plot(mde_range, sample_sizes)

axes[1].axvline(
    MDE_RELATIVE * 100,
    color='red',
    linestyle='--'
)

axes[1].set_yscale('log')
axes[1].set_title("Sample Size vs MDE")
axes[1].set_xlabel("MDE (%)")
axes[1].set_ylabel("Users per Group (log)")

# --- Plot 3: Summary Score Bar Configurations ---
# TODO: Draw a bar layout chart explicitly mapping control vs treatment ultimate mean values on axes[2]
axes[2].bar(
    ["Control", "Treatment"],
    [ctrl_mean, treat_mean]
)

axes[2].set_title("Average NDCG@10")
axes[2].set_ylabel("Mean Score")

plt.tight_layout()
plt.savefig("../../../output/Lab3/04_ab_protocol.png", dpi=150, bbox_inches='tight')
plt.show()

print("\n" + "=" * 60)
print("  LAB 3.4 COMPLETE — A/B PROTOCOL SUMMARY")
print("=" * 60)