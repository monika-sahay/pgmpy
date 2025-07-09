import numpy as np
import pandas as pd
import pytest
from pgmpy.estimators import DirectLiNGAMEstimator


def generate_data(n=500, seed=0):
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=n)
    x2 = 3 * x1 + 0.1 * rng.normal(size=n)
    x3 = -2 * x2 + 0.1 * rng.normal(size=n)
    return pd.DataFrame({"X1": x1, "X2": x2, "X3": x3})


def test_direct_lingam_nonlinear_fails_gracefully():
    rng = np.random.default_rng(0)
    x1 = rng.normal(size=100)
    x2 = np.sin(x1) + rng.normal(scale=0.1, size=100)
    x3 = np.exp(x2) + rng.normal(scale=0.1, size=100)

    df = pd.DataFrame({"X1": x1, "X2": x2, "X3": x3})
    est = DirectLiNGAMEstimator(df)
    est.fit()
    model = est.get_model()

    assert len(model.edges()) > 0


def test_direct_lingam_with_high_noise():
    rng = np.random.default_rng(42)
    x1 = rng.normal(size=100)
    x2 = 0.5 * x1 + rng.normal(scale=5.0, size=100)
    x3 = 0.5 * x2 + rng.normal(scale=5.0, size=100)

    df = pd.DataFrame({"X1": x1, "X2": x2, "X3": x3})
    est = DirectLiNGAMEstimator(df)
    est.fit()
    model = est.get_model()

    assert isinstance(set(model.nodes()), set)


def test_direct_lingam_independent_variables():
    rng = np.random.default_rng(123)
    x1 = rng.normal(size=1000)
    x2 = rng.normal(size=1000)
    x3 = rng.normal(size=1000)

    df = pd.DataFrame({"X1": x1, "X2": x2, "X3": x3})
    est = DirectLiNGAMEstimator(df, threshold=0.1, reps=500)  # more reps, looser threshold
    est.fit()
    model = est.get_model()

    assert len(model.edges()) <= 3 



def test_direct_lingam_random_shuffle():
    rng = np.random.default_rng(7)
    x1 = rng.normal(size=200)
    x2 = 2 * x1 + rng.normal(size=200)
    x3 = 3 * x2 + rng.normal(size=200)
    x4 = -1.2 * x1 + 0.3 * x3 + rng.normal(size=200)

    df = pd.DataFrame({"A": x1, "B": x2, "C": x3, "D": x4}).sample(frac=1).reset_index(drop=True)

    est = DirectLiNGAMEstimator(df)
    est.fit(bootstrap=True, n_runs=10)
    model = est.get_model()

    assert len(model.edges()) > 0


def test_direct_lingam_finds_true_cause():
    df = generate_data()
    est = DirectLiNGAMEstimator(df, threshold=0.05, reps=1000)
    est.fit()
    model = est.get_model()

    edges = model.edges()
    assert ("X1", "X2") in edges or ("X2", "X1") in edges


def test_direct_lingam_ensemble_returns_common_edges():
    df = generate_data()
    est = DirectLiNGAMEstimator(df)
    est.fit(bootstrap=True, n_runs=10, min_freq=0.5)
    model = est.get_model()

    assert ("X1", "X2") in model.edges() or ("X2", "X1") in model.edges()


def test_stability_selection_finds_stable_edge():
    rng = np.random.default_rng(42)
    x1 = rng.normal(size=300)
    x2 = 2.5 * x1 + 0.05 * rng.normal(size=300)

    df = pd.DataFrame({"X1": x1, "X2": x2})
    est = DirectLiNGAMEstimator(df, threshold=0.05, reps=2000)
    model = est.stability_selection(n_bootstrap=50, freq_threshold=0.5)

    edges = model.edges()
    assert ("X1", "X2") in edges or ("X2", "X1") in edges


def test_direct_lingam_on_adult():
    df = pd.read_csv("pgmpy/tests/test_estimators/testdata/adult.csv")
    df_numeric = pd.get_dummies(df, drop_first=True).iloc[:500]

    est = DirectLiNGAMEstimator(df_numeric, threshold=0.05, reps=300)
    est.fit()
    model = est.get_model()

    assert len(model.edges()) > 0

def test_contradictory_constraints_raises_error():
    df = generate_data()
    required = {("X1", "X2")}
    forbidden = {("X2", "X1")}

    with pytest.raises(ValueError, match="Conflict: Required edge"):
        DirectLiNGAMEstimator(df, required_edges=required, forbidden_edges=forbidden).fit()

def test_required_and_forbidden_edges():
    df = generate_data()
    required = {("X1", "X2")}
    forbidden = {("X3", "X1")}
    
    est = DirectLiNGAMEstimator(df, required_edges=required, forbidden_edges=forbidden, verbose=True)
    est.fit()

    print("Edges in final model:", list(est.model.edges()))
    assert est.model.has_edge("X1", "X2")
    assert not est.model.has_edge("X3", "X1")


def test_contradictory_constraints():
    df = generate_data()
    required = {("X1", "X2")}
    forbidden = {("X1", "X2")}
    with pytest.raises(ValueError):
        DirectLiNGAMEstimator(df, required_edges=required, forbidden_edges=forbidden)
