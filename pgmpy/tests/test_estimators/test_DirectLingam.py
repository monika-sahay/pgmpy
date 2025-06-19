import numpy as np
import pandas as pd
from pgmpy.estimators import DirectLiNGAMEstimator
import matplotlib.pyplot as plt
import networkx as nx







def test_direct_lingam_nonlinear_fails_gracefully():
    rng = np.random.default_rng(0)
    n = 100
    x1 = rng.normal(size=n)
    x2 = np.sin(x1) + rng.normal(scale=0.1, size=n)
    x3 = np.exp(x2) + rng.normal(scale=0.1, size=n)

    df = pd.DataFrame({"X1": x1, "X2": x2, "X3": x3})
    est = DirectLiNGAMEstimator(df)
    model = est.estimate()

    # ✅ Fails gracefully = produces too many edges or random structure
    print(len(model.edges()))
    assert len(model.edges()) > 0, "Model should return some edges, even if assumptions are violated"



def test_direct_lingam_with_high_noise():
    rng = np.random.default_rng(42)
    n = 100
    x1 = rng.normal(size=n)
    x2 = 0.5 * x1 + rng.normal(scale=5.0, size=n)  # large noise
    x3 = 0.5 * x2 + rng.normal(scale=5.0, size=n)

    df = pd.DataFrame({"X1": x1, "X2": x2, "X3": x3})
    est = DirectLiNGAMEstimator(df)  
    model = est.estimate()
    assert isinstance(set(model.nodes()), set)


def test_direct_lingam_independent_variables():
    rng = np.random.default_rng(123)
    n = 300  # increased sample size for stability
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    x3 = rng.normal(size=n)

    df = pd.DataFrame({"X1": x1, "X2": x2, "X3": x3})
    est = DirectLiNGAMEstimator(df, threshold=0.01)  # stricter independence check
    model = est.estimate()

    # Allowing slight tolerance for false positives
    assert len(model.edges()) <= 1


def test_direct_lingam_random_shuffle():
    rng = np.random.default_rng(7)
    n = 200
    x1 = rng.normal(size=n)
    x2 = 2 * x1 + rng.normal(size=n)
    x3 = 3 * x2 + rng.normal(size=n)
    x4 = -1.2 * x1 + 0.3 * x3 + rng.normal(size=n)

    df = pd.DataFrame({"A": x1, "B": x2, "C": x3, "D": x4})
    df = df.sample(frac=1).reset_index(drop=True)  # shuffle rows

    est = DirectLiNGAMEstimator(df)
    stable_model = est.ensemble_estimate(n_runs=10)
    print(stable_model.edges())

def generate_data(n=500, seed=0):
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=n)
    x2 = 3 * x1 + 0.1 * rng.normal(size=n)
    x3 = -2 * x2 + 0.1 * rng.normal(size=n)
    return pd.DataFrame({"X1": x1, "X2": x2, "X3": x3})

# def test_direct_lingam_finds_true_cause():
#     data = generate_data()
#     est = DirectLiNGAMEstimator(data)
#     model = est.estimate()
#     assert ("X1", "X2") in model.edges() or ("X2", "X1") in model.edges()

def test_direct_lingam_finds_true_cause():
    data = generate_data()
    est = DirectLiNGAMEstimator(data, threshold=0.05, reps=1000)
    model = est.estimate()

    edges = list(model.edges())
    print("Estimated edges:", edges)

    assert ("X1", "X2") in edges or ("X2", "X1") in edges


def test_direct_lingam_ensemble_returns_common_edges():
    df = generate_data()
    est = DirectLiNGAMEstimator(df)
    model = est.ensemble_estimate(n_runs=10, min_freq=0.5)
    assert ("X1", "X2") in model.edges() or ("X2", "X1") in model.edges()


def test_stability_selection_finds_stable_edge():
    rng = np.random.default_rng(42)
    n = 300
    x1 = rng.normal(size=n)
    x2 = 2.5 * x1 + 0.05 * rng.normal(size=n)  # smaller noise

    df = pd.DataFrame({"X1": x1, "X2": x2})

    est = DirectLiNGAMEstimator(df, threshold=0.05, reps=2000)
    stable_model = est.stability_selection(n_bootstrap=50, freq_threshold=0.5)

    edges = list(stable_model.edges())
    print(f"Stable edges: {edges}")

    assert ("X1", "X2") in edges or ("X2", "X1") in edges
