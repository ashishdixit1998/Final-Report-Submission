# =============================================================================
# MODULE 4 | LAB 4.1
# File: 01_model_serialization.py
# Purpose: Implement MLflow MLOps registry tracking, spin up background
#          FastAPI servers via uvicorn, and benchmark cache-hit vs cache-miss
#          request latencies against production p99 SLA targets (< 50ms).
# Saras AI Institute | Build Predictive Models & Modern Recommenders
# =============================================================================

import os
import time
import json
import pickle
import subprocess
import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("  MODULE 4 | LAB 4.1")
print("  MLflow Registry + FastAPI Endpoint + Latency Benchmark")
print("=" * 60)

# ---------------------------------------------------------------------------
# SECTION 1: Install and Import MLflow
# ---------------------------------------------------------------------------
print("\n[1] Setting up MLflow...")

try:
    import mlflow
    import mlflow.sklearn
    import mlflow.pyfunc
    print(f"    MLflow version : {mlflow.__version__}")
except ImportError:
    print("    Installing MLflow...")
    os.system("pip install mlflow -q")
    import mlflow

# TODO: Configure MLflow back-end database storage tracking URI to "sqlite:///data/mlflow.db"
# Hint: Call mlflow.set_tracking_uri()
mlflow.set_tracking_uri("sqlite:///../../../data/mlflow.db")

# TODO: Initialize or switch to an active MLflow experiment workspace named "hybrid-recommender"
# Hint: Call mlflow.set_experiment()
mlflow.set_experiment("hybrid-recommender")

print(f"    Experiment     : hybrid-recommender")

# ---------------------------------------------------------------------------
# SECTION 2: Load Artifacts
# ---------------------------------------------------------------------------
print("\n[2] Loading model artifacts...")

with open("../../../data/als_artifacts.pkl",    "rb") as f: als_art = pickle.load(f)
with open("../../../data/lightfm_artifacts.pkl","rb") as f: lfm_art = pickle.load(f)
with open("../../../data/faiss_artifacts.pkl",  "rb") as f: fai_art = pickle.load(f)

als_model   = als_art['model']
lfm_model   = lfm_art['model_hybrid']
best_ndcg   = als_art.get('best_ndcg', 0.017)
hybrid_p10  = lfm_art.get('hybrid_test_precision', 0.01)

print(f"    ALS model loaded   : factors={als_model.factors}")
print(f"    LightFM loaded     : components={lfm_model.no_components}")


# ---------------------------------------------------------------------------
# SECTION 3: Register ALS in MLflow
# ---------------------------------------------------------------------------
print("\n[3] Registering ALS model in MLflow...")

# TODO: Open an active MLflow run context assigning a run name="als_personalization_engine"
# Hint: Use python's context manager "with mlflow.start_run(run_name=...):"
with mlflow.start_run(run_name="als_personalization_engine"):
    mlflow.log_param("model_type", "ALS")
    mlflow.log_param("factors", als_model.factors)
    mlflow.log_param("iterations", als_model.iterations)
    mlflow.log_param("regularization", als_model.regularization)
    mlflow.log_param("n_users", len(als_art["user_ids"]))
    mlflow.log_param("n_items", len(als_art["item_ids"]))

    mlflow.log_metric("ndcg_at_10", best_ndcg)

    mlflow.log_artifact("../../../data/als_artifacts.pkl", artifact_path="model")
    mlflow.log_artifact("../../../data/faiss_artifacts.pkl", artifact_path="model")
    # TODO: Log hyperparameter tokens to the metadata registry using mlflow.log_param()
    # Log: "model_type" -> "ALS", "factors" -> als_model.factors, "iterations" -> als_model.iterations, 
    # "regularization" -> als_model.regularization, "n_users" -> len(als_art['user_ids']), "n_items" -> len(als_art['item_ids'])

    
    # TODO: Record validation metrics to the run metadata registry using mlflow.log_metric()
    # Metric: "ndcg_at_10" -> best_ndcg

    
    # TODO: Bind tracking artifact binary paths to the current tracking run using mlflow.log_artifact()
    # Track: "data/als_artifacts.pkl" and "data/faiss_artifacts.pkl" under the directory boundary "model"

    
    # Isolate your dynamic workspace run identifier
    als_run_id = mlflow.active_run().info.run_id
    print(f"    ALS run ID    : {als_run_id}")

# ---------------------------------------------------------------------------
# SECTION 4: Register LightFM in MLflow
# ---------------------------------------------------------------------------
print("\n[4] Registering LightFM model in MLflow...")

# TODO: Open an alternative run logging context tracking the coldstart engine under name="lightfm_coldstart_engine"
with mlflow.start_run(run_name="lightfm_coldstart_engine"):
    mlflow.log_param("model_type", "LightFM")
    mlflow.log_param("loss", "warp")
    mlflow.log_param("no_components", lfm_model.no_components)
    mlflow.log_param("n_items", len(fai_art["item_ids"]))

    mlflow.log_metric("precision_at_10", hybrid_p10)

    mlflow.log_artifact(
        "../../../data/lightfm_artifacts.pkl",
        artifact_path="model"
    )
    # TODO: Log system tracking properties using mlflow.log_param()
    # Params: "model_type" -> "LightFM", "loss" -> "warp", "no_components" -> lfm_model.no_components, "n_items" -> fai_art['n_items']

    
    # TODO: Log validation metric constraints via mlflow.log_metric()
    # Metric: "precision_at_10" -> hybrid_p10

    
    # TODO: Log target binary storage payloads via mlflow.log_artifact()
    # Payload: "data/lightfm_artifacts.pkl" pointing to artifact path "model"

    
    lfm_run_id = mlflow.active_run().info.run_id
    print(f"    LightFM run ID  : {lfm_run_id}")


# ---------------------------------------------------------------------------
# SECTION 5: Show MLflow Registry
# ---------------------------------------------------------------------------
print("\n[5] MLflow experiment runs:")

# TODO: Instantiate an mlflow.tracking.MlflowClient() object, retrieve the "hybrid-recommender" experiment,
# and use client.search_runs() to gather experiment entries to print out metadata updates
client = mlflow.tracking.MlflowClient()

experiment = client.get_experiment_by_name(
    "hybrid-recommender"
)

runs = client.search_runs(
    experiment.experiment_id
)

for run in runs:
    print("=" * 40)
    print("Run ID:", run.info.run_id)
    print("Params :", run.data.params)
    print("Metrics:", run.data.metrics)


# ---------------------------------------------------------------------------
# SECTION 6: Start FastAPI Server
# ---------------------------------------------------------------------------
print("\n[6] Starting FastAPI server...")
print("    Starting uvicorn on http://127.0.0.1:8000 ...")

# TODO: Automate hosting by spawning an background asynchronous process pointing to app.py
# Hint: Use subprocess.Popen() to call -> ["python", "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--log-level", "warning"]
# Map stdout and stderr to subprocess.DEVNULL to silence server logs within the main execution loop
import sys
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]

# Check if a server is already running
server_running = False

try:
    resp = requests.get("http://127.0.0.1:8000/health", timeout=2)
    if resp.status_code == 200:
        server_running = True
        print("    FastAPI server already running.")
        health = resp.json()
        print(f"    Redis Connected : {health['redis_connected']}")
except Exception:
    pass

server_process = None

if not server_running:
    server_process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "app:app",
            "--host",
            "127.0.0.1",
            "--port",
            "8000",
            "--log-level",
            "warning",
        ],
        cwd=ROOT,
    )

    print("    Waiting for server to be ready...")

    for attempt in range(30):
        try:
            resp = requests.get(
                "http://127.0.0.1:8000/health",
                timeout=5
            )

            if resp.status_code == 200:
                print(f"    Server ready after {attempt + 1} attempts")
                health = resp.json()
                print(f"    Redis Connected : {health['redis_connected']}")
                break

        except Exception:
            time.sleep(1)
    else:
        print("    WARNING: Server did not start in 30s")

# ---------------------------------------------------------------------------
# SECTION 7: Test the Endpoint
# ---------------------------------------------------------------------------
print("\n[7] Testing /recommend endpoint...")

if als_art is not None:
    sample_users = [int(u) for u in list(als_art['user_ids'])[:3]]

    for user_id in sample_users:
        # TODO: Construct a requests.get loop fetching recommendations from "http://127.0.0.1:8000/recommend/{user_id}"
        # Parameters to append: {"top_k": 5, "use_cache": True}
        # Print engine outputs and processing latencies returned by the payload body json
        response = requests.get(
        f"http://127.0.0.1:8000/recommend/{user_id}",
        params={
            "top_k": 5,
            "use_cache": True
        }
    )

        if response.status_code == 200:
            data = response.json()

            print(f"\nUser {user_id}")
            print("Cached :", data.get("cached"))
            print("Engine :", data.get("engine"))
            print("Latency:", data.get("latency_ms"), "ms")
            print("Recommendations:", data.get("recommendations"))


# ---------------------------------------------------------------------------
# SECTION 8: Latency Benchmark
# ---------------------------------------------------------------------------
print("\n[8] Latency benchmark (200 requests)...")

N_BENCHMARK = 200
if als_art is not None:
    all_users   = [int(u) for u in list(als_art['user_ids'])]
    bench_users = np.random.choice(all_users, N_BENCHMARK, replace=True)
    print("\nBenchmark users (first 20):")
    print(bench_users[:20])

cold_latencies = []   # Tracks Cache Miss (pipeline generation + serialization costs)
warm_latencies = []   # Tracks Cache Hit  (Redis memory access lookup latencies)

print("    Running cold requests (cache miss)...")
for user_id in bench_users[:100]:
    try:
        requests.delete(f"http://127.0.0.1:8000/cache/{user_id}")

        start = time.perf_counter()

        response = requests.get(
            f"http://127.0.0.1:8000/recommend/{user_id}",
            params={
                "top_k": 10,
                "use_cache": True
            }
        )

        end = time.perf_counter()

        latency = (end - start) * 1000

        if response.status_code == 200:
            cold_latencies.append(latency)
            #cold_latencies.append((end - start) * 1000)

    except Exception:
        pass

print("    Running warm requests (cache hit)...")
for user_id in bench_users[:100]:
    try:
        start = time.perf_counter()

        response = requests.get(
            f"http://127.0.0.1:8000/recommend/{user_id}",
            params={
                "top_k": 10,
                "use_cache": True
            }
        )

        end = time.perf_counter()
        
        if response.status_code == 200:
            data = response.json()

            latency = (end - start) * 1000
            if data["cached"]:
                warm_latencies.append(latency)
                #warm_latencies.append((end - start) * 1000)

    except Exception:
        pass

# --- Statistical Performance Reporting ---
# TODO: Map percentile limits (p50, p95, p99) over cold_latencies and warm_latencies using np.percentile()
# Validate that processing times fit within strict target performance windows (< 50ms)
percentiles = [50, 75, 90, 95, 99]
print("\nCold samples collected:", len(cold_latencies))
print("Warm samples collected:", len(warm_latencies))

print("First 10 cold:", cold_latencies[:10])
print("First 10 warm:", warm_latencies[:10])
cold_p = np.percentile(cold_latencies, percentiles)

warm_p = np.percentile(warm_latencies, percentiles)

print("\nPerformance Summary")
print("=" * 60)

print(f"Cold Requests")
print(f"  P50 : {cold_p[0]:.2f} ms")
print(f"  P75 : {cold_p[1]:.2f} ms")
print(f"  P90 : {cold_p[2]:.2f} ms")
print(f"  P95 : {cold_p[3]:.2f} ms")
print(f"  P99 : {cold_p[4]:.2f} ms")

print()

print(f"Warm Requests")
print(f"  P50 : {warm_p[0]:.2f} ms")
print(f"  P75 : {warm_p[1]:.2f} ms")
print(f"  P90 : {warm_p[2]:.2f} ms")
print(f"  P95 : {warm_p[3]:.2f} ms")
print(f"  P99 : {warm_p[4]:.2f} ms")

print()

if warm_p[4] < 50:
    print("✅ SLA PASSED (P99 < 50 ms)")
else:
    print("❌ SLA FAILED (P99 >= 50 ms)")

# ---------------------------------------------------------------------------
# SECTION 9: Latency Visualization
# ---------------------------------------------------------------------------
print("\n[9] Plotting latency results...")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Lab 4.1: FastAPI Endpoint Latency Benchmark\nCold (cache miss) vs Warm (cache hit)", fontsize=12, fontweight='bold')

# --- Plot 1: Latency Density Distributions Histograms ---
# TODO: Render overlaid tracking histograms tracking cold_latencies against warm_latencies
# Identify SLA constraints by overlaying a reference marker threshold via axes[0].axvline(x=50, color='red')

axes[0].hist(
    cold_latencies,
    bins=20,
    alpha=0.6,
    label="Cold"
)

axes[0].hist(
    warm_latencies,
    bins=20,
    alpha=0.6,
    label="Warm"
)

axes[0].axvline(
    x=50,
    color="red",
    linestyle="--",
    label="SLA"
)

axes[0].legend()

axes[0].set_title("Latency Distribution")
axes[0].set_xlabel("Latency (ms)")
axes[0].set_ylabel("Requests")

# --- Plot 2: Latency Percentile Profiles Bar Graph ---
# TODO: Assemble comparative adjacent tracking bars checking scores across percentiles list: [50, 75, 90, 95, 99]
percentiles = [50, 75, 90, 95, 99]

x = np.arange(len(percentiles))
width = 0.35

axes[1].bar(
    x - width / 2,
    cold_p,
    width,
    label="Cold"
)

axes[1].bar(
    x + width / 2,
    warm_p,
    width,
    label="Warm"
)

axes[1].set_xticks(x)
axes[1].set_xticklabels(percentiles)

axes[1].legend()

axes[1].set_title("Percentile Comparison")
axes[1].set_xlabel("Percentile")
axes[1].set_ylabel("Latency (ms)")

plt.tight_layout()
plt.savefig("../../../output/Lab4/01_fastapi_latency.png", dpi=150, bbox_inches='tight')
plt.show()


# ---------------------------------------------------------------------------
# SHUTDOWN SERVER BACKGROUND TIMERS
# ---------------------------------------------------------------------------
 # TODO: Gracefully shut down background system wrappers to free local listening ports
    # Hint: Call server_process.terminate()
if server_process is not None:
    server_process.terminate()
    server_process.wait()
    print("\n    Server process terminated")
else:
    print("\n    Using existing FastAPI server - not shutting it down.")
