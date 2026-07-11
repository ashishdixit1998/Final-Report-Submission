import os
import json
import time
import pickle
import numpy as np
import pytest
import requests
import warnings
warnings.filterwarnings('ignore')

# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------
BASE_URL   = os.getenv("API_URL", "http://localhost:8000")
DATA_DIR   = os.getenv(
    "ARTIFACTS_DIR",
    r"C:\Users\mvsuser\OneDrive - Multiverse\OneDrive - Multiverse Solutions Pvt Ltd\Documents\Saras Lab\Labs Saras Predictive\Lab 4\Dataset"
)
SERVER_UP  = False  

# ---------------------------------------------------------------------------
# FIXTURES
# ---------------------------------------------------------------------------
@pytest.fixture(scope="session")
def artifacts():
    """
    Load all artifacts once for the entire test session.
    """
    # TODO: Load 'als_artifacts.pkl', 'lightfm_artifacts.pkl', 'faiss_artifacts.pkl', 
    # and 'routing_split.pkl' from DATA_DIR. Return them as a dictionary.
    # Hint: Use pytest.skip() inside an except FileNotFoundError block if files are missing.
    try:
        return {
            "als": pickle.load(open(os.path.join(DATA_DIR, "als_artifacts.pkl"), "rb")),
            "lightfm": pickle.load(open(os.path.join(DATA_DIR, "lightfm_artifacts.pkl"), "rb")),
            "faiss": pickle.load(open(os.path.join(DATA_DIR, "faiss_artifacts.pkl"), "rb")),
            "routing": pickle.load(open(os.path.join(DATA_DIR, "routing_split.pkl"), "rb"))
        }
    except FileNotFoundError:
        pytest.skip("Artifacts not found")

@pytest.fixture(scope="session")
def sample_users(artifacts):
    """
    Return a small set of known user IDs for testing.
    """
    faiss = artifacts["faiss"]

    if "user_to_idx" in faiss:
        return list(faiss["user_to_idx"].keys())[:20]

    if "user_mapping" in faiss:
        return list(faiss["user_mapping"].keys())[:20]

    if "user_to_index" in faiss:
        return list(faiss["user_to_index"].keys())[:20]

    if "user_map" in faiss:
        return list(faiss["user_map"].keys())[:20]

    pytest.skip("No user mapping found in FAISS artifacts")

@pytest.fixture(scope="session")
def api_available():
    """
    Check if the API server is running by hitting the /health endpoint.
    """
    # TODO: Construct a requests.get query pointing to f"{BASE_URL}/health".
    # Return True if status_code == 200, else return False.
    try:
        r = requests.get(f"{BASE_URL}/health", timeout=3)
        return r.status_code == 200
    except:
        return False

# ===========================================================================
# UNIT TESTS — Pure Python, no API needed
# ===========================================================================

class TestScoreNormalization:
    """Unit tests for score normalization function."""

    def _normalize(self, scores):
        """Inline normalization tracker under test."""
        arr = np.array(scores, dtype=float)
        lo, hi = arr.min(), arr.max()
        if hi == lo:
            return [0.5] * len(scores)
        return ((arr - lo) / (hi - lo)).tolist()

    def test_all_scores_in_zero_one_range(self):
        # TODO: Define a dummy list of raw floating scores (e.g., negative, positive, or outside [0, 1])
        # Assert that after passing scores through self._normalize, ALL elements fall within [0.0, 1.0]
        scores = [-5, 0, 100]
        out = self._normalize(scores)
        assert all(0 <= x <= 1 for x in out)

    def test_max_score_is_one(self):
        # TODO: Define a random list of asymmetric scores.
        # Assert that the maximum value of the normalized output list is exactly 1.0 (use pytest.approx)
        out = self._normalize([5, 10, 20])
        assert max(out) == pytest.approx(1.0)

    def test_min_score_is_zero(self):
        # TODO: Define a random list of scores.
        # Assert that the minimum value of the normalized output list is exactly 0.0 (use pytest.approx)
        out = self._normalize([5, 10, 20])
        assert min(out) == pytest.approx(0.0)

    def test_identical_scores_return_half(self):
        # TODO: Pass a list of identical values (e.g., [0.5, 0.5, 0.5]) into self._normalize
        # Assert that all resulting values equal 0.5
        out = self._normalize([5,5,5])
        assert out == [0.5,0.5,0.5]

    def test_single_score_returns_half(self):
        # TODO: Pass a single element score list into self._normalize
        # Assert that the single output scalar equals 0.5
        out = self._normalize([5])
        assert out == [0.5]

    def test_output_length_matches_input(self):
        # TODO: Assert that the size of the output list matches the input length exactly
        scores = [1,2,3,4]
        assert len(self._normalize(scores)) == len(scores)

    def test_negative_scores_handled(self):
        # TODO: Pass a list containing negative numbers (e.g., [-1.0, 0.0, 2.0]) into self._normalize
        # Assert that elements are scaled into a valid [0, 1] range correctly
        out = self._normalize([-1,0,2])
        assert min(out) >= 0
        assert max(out) <= 1


class TestRoutingLogic:
    """Unit tests for the routing decision logic."""

    def _route(self, user_id, als_users, n_interactions, threshold=3):
        """Inline routing logic tracker under test."""
        use_als = (
            user_id in als_users and
            n_interactions >= threshold
        )
        return "ALS" if use_als else "LightFM"

    def test_returning_user_routes_to_als(self, artifacts):
        # TODO: Extract 'als_users' from the routing split artifact. Grab a sample user_id.
        # Call self._route providing an interaction count above threshold (e.g., n_interactions=10).
        # Assert that the returned routing string equals "ALS".
        als_users = artifacts["routing"]["als_users"]
        uid = next(iter(als_users))
        assert self._route(uid, als_users, 10) == "ALS"

    def test_new_user_routes_to_lightfm(self, artifacts):
        # TODO: Define a fake user_id not present in the dataset (e.g., -99999).
        # Assert that calling self._route returns "LightFM".
        als_users = artifacts["routing"]["als_users"]
        assert self._route(-99999, als_users, 10) == "LightFM"

    def test_returning_user_below_threshold_routes_to_lightfm(self, artifacts):
        # TODO: Grab a valid returning user_id, but pass an interaction count below threshold (e.g., n_interactions=1).
        # Assert that the resulting routing engine equals "LightFM".
        als_users = artifacts["routing"]["als_users"]
        uid = next(iter(als_users))
        assert self._route(uid, als_users, 1) == "LightFM"

    def test_threshold_boundary_exact(self, artifacts):
        # TODO: Test exact threshold constraints. Pass n_interactions=3 with threshold=3.
        # Assert that the user is successfully routed to "ALS".
        als_users = artifacts["routing"]["als_users"]
        uid = next(iter(als_users))
        assert self._route(uid, als_users, 3) == "ALS"


class TestPurchaseExclusion:
    """Unit tests for already-purchased item exclusion."""

    def _exclude(self, items, scores, user_purchases):
        filtered = [(it, sc) for it, sc in zip(items, scores) if it not in user_purchases]
        if not filtered:
            return items, scores
        its, scs = zip(*filtered)
        return list(its), list(scs)

    def test_purchased_items_removed(self):
        # TODO: Define list of items, matching score float keys, and a 'purchased' set containing elements to exclude.
        # Assert that excluded items do NOT exist inside the final cleaned items array.
        items=[1,2,3]
        scores=[0.1,0.2,0.3]
        out,_=self._exclude(items,scores,{2})
        assert 2 not in out

    def test_non_purchased_items_kept(self):
        # TODO: Define item arrays, scores, and a purchase exclusion list.
        # Assert that items not marked as purchased are retained in the filtered output list.
        items=[1,2,3]
        scores=[0.1,0.2,0.3]
        out,_=self._exclude(items,scores,{2})
        assert 1 in out
        assert 3 in out

    def test_no_purchases_returns_original(self):
        # TODO: Provide an empty set() for user purchases.
        # Assert that filtered outputs are identical to the raw inputs.
        items=[1,2]
        scores=[0.1,0.2]
        out_i,out_s=self._exclude(items,scores,set())
        assert out_i==items
        assert out_s==scores

    def test_all_purchased_returns_original(self):
        # TODO: Test boundary conditions. Provide a purchase history containing all items in the candidate list.
        # Assert that the function falls back gracefully by returning the original lists.
        items=[1,2]
        scores=[0.1,0.2]
        out_i,out_s=self._exclude(items,scores,{1,2})
        assert out_i==items
        assert out_s==scores

    def test_scores_aligned_after_exclusion(self):
        # TODO: Define items, scores, and a partial exclusion set.
        # Loop through the outputs and assert that every retained item's score remains perfectly aligned with its original input score.
        items=[1,2,3]
        scores=[10,20,30]
        out_i,out_s=self._exclude(items,scores,{2})
        assert out_s[out_i.index(3)]==30


class TestNDCG:
    """Unit tests for NDCG@K metric."""

    def _ndcg(self, recommended, relevant, k=10):
        rec_k = recommended[:k]
        dcg   = sum(1.0/np.log2(i+2) for i, it in enumerate(rec_k) if it in relevant)
        idcg  = sum(1.0/np.log2(i+2) for i in range(min(len(relevant), k)))
        return dcg / idcg if idcg > 0 else 0.0

    def test_perfect_ranking_returns_one(self):
        # TODO: Define a relevant set and recommended list where the top items perfectly match the relevant set.
        # Assert that self._ndcg returns 1.0 (use pytest.approx).
        relevant = {1, 2, 3}
        recommended = [1, 2, 3]
        assert self._ndcg(recommended, relevant) == pytest.approx(1.0)

    def test_no_relevant_returns_zero(self):
        # TODO: Define completely disjoint relevant sets and recommendations.
        # Assert that self._ndcg returns 0.0.
        relevant = {10, 20}
        recommended = [1, 2, 3]
        assert self._ndcg(recommended, relevant) == 0.0

    def test_relevant_at_top_beats_bottom(self):
        # TODO: Contrast two lists: one with a relevant item at index 0, and another with the same item at the bottom.
        # Assert that the score for the top-ranked item is higher than the bottom-ranked item.
        relevant = {5}

        top = self._ndcg([5, 1, 2, 3], relevant)
        bottom = self._ndcg([1, 2, 3, 5], relevant)

        assert top > bottom


class TestMMRDiversity:
    """Unit tests for MMR reranking."""

    def _mmr(self, items, scores, embeddings, id_to_idx, top_k=5, lam=0.5):
        norm_s = np.array(scores, dtype=float)
        lo, hi = norm_s.min(), norm_s.max()
        if hi > lo: norm_s = (norm_s - lo) / (hi - lo)
        valid = [(it, sc, embeddings[id_to_idx[it]]) for it, sc in zip(items, norm_s) if it in id_to_idx]
        if not valid: return items[:top_k]
        selected, sel_embs, remaining = [], [], list(range(len(valid)))
        for _ in range(min(top_k, len(valid))):
            if not remaining: break
            if not sel_embs:
                best = max(remaining, key=lambda i: valid[i][1])
            else:
                sel_m = np.array(sel_embs)
                mmr_v = []
                for i in remaining:
                    rel = lam * valid[i][1]
                    sim = (valid[i][2] @ sel_m.T).max()
                    mmr_v.append(rel - (1-lam)*sim)
                best = remaining[np.argmax(mmr_v)]
            selected.append(valid[best][0])
            sel_embs.append(valid[best][2])
            remaining.remove(best)
        return selected

    def test_output_length_equals_top_k(self):
        # TODO: Mock a list of candidate items, relevance scores, an id mapping dictionary, and unit-norm embeddings.
        # Assert that the output length of self._mmr matches top_k exactly.
        items = [0,1,2,3,4]
        scores = [5,4,3,2,1]

        embeddings = np.eye(5)
        id_to_idx = {i:i for i in range(5)}

        out = self._mmr(
            items,
            scores,
            embeddings,
            id_to_idx,
            top_k=3
        )

        assert len(out) == 3

    def test_no_duplicate_items(self):
        # TODO: Pass a list of candidates into self._mmr.
        # Assert that the returned recommendations contain no duplicate IDs.
        items = [0,1,2,3,4]
        scores = [5,4,3,2,1]

        embeddings = np.eye(5)
        id_to_idx = {i:i for i in range(5)}

        out = self._mmr(
            items,
            scores,
            embeddings,
            id_to_idx
        )

        assert len(out) == len(set(out))

    def test_lambda_one_returns_highest_relevance_first(self):
        # TODO: Configure pure relevance by setting lam=1.0. Pass mismatched items/scores.
        # Assert that the first returned item is the one with the highest absolute score.
        items = [0,1,2]
        scores = [1,9,2]

        embeddings = np.eye(3)
        id_to_idx = {0:0,1:1,2:2}

        out = self._mmr(
            items,
            scores,
            embeddings,
            id_to_idx,
            lam=1.0
        )

        assert out[0] == 1


# ===========================================================================
# INTEGRATION TESTS — Require running API server
# ===========================================================================

class TestAPIHealth:
    """Integration tests for the /health endpoint."""

    def test_health_returns_200(self, api_available):
        if not api_available: pytest.skip("API server not running")
        # TODO: Issue a requests.get method call to f"{BASE_URL}/health". Assert status code is 200.
        r = requests.get(f"{BASE_URL}/health")
        assert r.status_code == 200

    def test_health_response_schema(self, api_available):
        if not api_available: pytest.skip("API server not running")
        # TODO: Extract the JSON payload from the health endpoint.
        # Assert that keys like "status", "als_loaded", "lfm_loaded", and "redis_connected" exist in the response.
        if not api_available:
            pytest.skip("API server not running")

        data = requests.get(f"{BASE_URL}/health").json()

        assert "status" in data
        assert "als_loaded" in data
        assert "lfm_loaded" in data
        assert "redis_connected" in data

    def test_health_status_is_ok(self, api_available):
        if not api_available: pytest.skip("API server not running")
        # TODO: Assert that response JSON data["status"] == "ok".
        data = requests.get(f"{BASE_URL}/health").json()
        assert data["status"] == "ok"


class TestRecommendEndpoint:
    """Integration tests for the /recommend endpoint."""

    def test_valid_user_returns_200(self, api_available, sample_users):
        if not api_available: pytest.skip("API server not running")
        # TODO: Query the recommendation endpoint using a valid user ID from the sample_users fixture.
        # Assert that the server handles the call with an HTTP status code of 200.
        r = requests.get(
        f"{BASE_URL}/recommend/{sample_users[0]}"
        )

        assert r.status_code == 200

    def test_response_has_required_fields(self, api_available, sample_users):
        if not api_available: pytest.skip("API server not running")
        # TODO: Assert that the endpoint response body contains keys like "engine", "recommendations", "scores", and "latency_ms".
        data = requests.get(
        f"{BASE_URL}/recommend/{sample_users[0]}"
        ).json()

        assert "engine" in data
        assert "recommendations" in data
        assert "scores" in data
        assert "latency_ms" in data

    def test_recommendations_count_matches_top_k(self, api_available, sample_users):
        if not api_available: pytest.skip("API server not running")
        # TODO: Pass params={"top_k": 5} to the endpoint call.
        # Assert that the length of returned recommendations list is less than or equal to 5.
        data = requests.get(
        f"{BASE_URL}/recommend/{sample_users[0]}",
        params={"top_k": 5}
        ).json()

        assert len(data["recommendations"]) <= 5

    def test_scores_in_zero_one_range(self, api_available, sample_users):
        if not api_available:
            pytest.skip("API server not running")

        data = requests.get(
            f"{BASE_URL}/recommend/{sample_users[0]}"
        ).json()

        assert all(0.0 <= score <= 1.0 for score in data["scores"])

    def test_latency_under_threshold(self, api_available, sample_users):
        if not api_available: pytest.skip("API server not running")
        # TODO: Query the recommendation endpoint for a sample user.
        # Assert that the returned latency field "latency_ms" is well under your production SLA limits (e.g., < 500ms).
        data = requests.get(
        f"{BASE_URL}/recommend/{sample_users[0]}"
        ).json()

        assert data["latency_ms"] < 500


# ===========================================================================
# END-TO-END TESTS — Seeded dataset verification
# ===========================================================================

class TestEndToEnd:
    """End-to-end integration and system behavior assertions."""

    def test_e2e_recommendations_are_integers(self, api_available, sample_users):
        if not api_available: pytest.skip("API server not running")
        # TODO: Call the recommendation endpoint. Assert that all item identifiers in the recommendations list are raw python integers.
        data = requests.get(
        f"{BASE_URL}/recommend/{sample_users[0]}"
        ).json()
        assert all(isinstance(item, int) for item in data["recommendations"])

    def test_e2e_multiple_users_get_different_recs(self, api_available, sample_users):
        if not api_available:
            pytest.skip("API server not running")

        # First user
        data1 = requests.get(
            f"{BASE_URL}/recommend/{sample_users[0]}",
            params={"top_k": 10}
        ).json()

        assert "recommendations" in data1, f"First user response: {data1}"

        # Find a second user that actually has recommendations
        data2 = None

        for uid in sample_users[1:]:
            resp = requests.get(
                f"{BASE_URL}/recommend/{uid}",
                params={"top_k": 10}
            ).json()

            if "recommendations" in resp:
                data2 = resp
                break

        if data2 is None:
            pytest.skip("No second user with recommendations found")

        recs1 = set(data1["recommendations"])
        recs2 = set(data2["recommendations"])

        overlap = len(recs1.intersection(recs2))

        assert overlap < 10
    
    def test_e2e_stats_endpoint(self, api_available):
        if not api_available: pytest.skip("API server not running")
        # TODO: Issue a requests.get method target against f"{BASE_URL}/stats".
        # Assert response is 200, and fields like "als_users" or "lfm_items" exist.
        r = requests.get(f"{BASE_URL}/stats")

        assert r.status_code == 200

        data = r.json()

        assert "als_users" in data
        assert "lfm_items" in data


# ===========================================================================
# STANDALONE RUNNER — run without pytest for demo
# ===========================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("  MODULE 4 | LAB 4.3")
    print("  Running test suite in standalone mode")
    print("=" * 60)
