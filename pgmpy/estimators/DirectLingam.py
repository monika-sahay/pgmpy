import numpy as np
import pandas as pd
from pgmpy.base import DAG
from sklearn.linear_model import LinearRegression
from hyppo.independence import Hsic
from sklearn.preprocessing import StandardScaler
from collections import Counter


def hsic_test(x, y, threshold=0.05, reps=1000, verbose=False):
    hsic = Hsic()
    stat, pval = hsic.test(x, y, reps=reps)
    if verbose:
        print(f"HSIC p-value = {pval:.4f}, threshold = {threshold}")
    return pval

class DirectLiNGAMEstimator:
    def __init__(self, data: pd.DataFrame, threshold=0.05, reps=1000, verbose=False,
                 forbidden_edges=None, required_edges=None):
        self.data = pd.DataFrame(StandardScaler().fit_transform(data), columns=data.columns)
        self.variables = list(data.columns)
        self.threshold = threshold
        self.reps = reps
        self.verbose = verbose
        self.model = DAG()
        self.forbidden_edges = set(forbidden_edges or [])
        self.required_edges = set(required_edges or [])

    def _is_exogenous(self, x, rest):
        if not rest:
            return True
        x_data = self.data[x].values
        rest_data = self.data[rest].values
        reg = LinearRegression().fit(rest_data, x_data)
        residuals = x_data - reg.predict(rest_data)
        pvals = []
        for i in range(rest_data.shape[1]):
            p = hsic_test(residuals.reshape(-1, 1), rest_data[:, i].reshape(-1, 1),
                        threshold=self.threshold, reps=self.reps, verbose=self.verbose)
            pvals.append(p)

        # If at least 80% of residuals pass independence test, we accept
        pass_rate = sum(p > self.threshold for p in pvals) / len(pvals)
        return pass_rate >= 0.8

    def estimate(self, direction_delta=0.2, fail_gracefully=True):
        ordered_vars = []
        remaining = set(self.variables)

        while remaining:
            found = False
            for var in list(remaining):
                others = list(remaining - {var})
                if self._is_exogenous(var, others):
                    ordered_vars.append(var)
                    remaining.remove(var)
                    found = True
                    break
            if not found:
                if fail_gracefully:
                    if self.verbose:
                        print("⚠️ Falling back due to nonlinear or cyclic structure.")
                    self.model = DAG()
                    self.model.add_nodes_from(self.variables)
                    # Fully connect variables as noisy fallback
                    print(len(self.model.edges()))
                    for i in range(len(self.variables)):
                        for j in range(len(self.variables)):
                            if i != j:
                                self.model.add_edge(self.variables[i], self.variables[j])
                                print(len(self.model.edges()))
                    return self.model
                else:
                    if self.verbose:
                        print("No exogenous variable found — likely nonlinear or cyclic structure.")
                    raise RuntimeError("Model assumption violated: No exogenous variable found.", print(self.model.edges()))

        for i in range(1, len(ordered_vars)):
            for j in range(i):
                src = ordered_vars[j]
                tgt = ordered_vars[i]
                x = self.data[src].values.reshape(-1, 1)
                y = self.data[tgt].values.reshape(-1, 1)
                reg1 = LinearRegression().fit(x, y)
                resid1 = y - reg1.predict(x)
                pval1 = hsic_test(resid1, x, threshold=self.threshold, reps=self.reps, verbose=self.verbose)

                reg2 = LinearRegression().fit(y, x)
                resid2 = x - reg2.predict(y)
                pval2 = hsic_test(resid2, y, threshold=self.threshold, reps=self.reps, verbose=self.verbose)

                if pval1 > pval2 and pval1 > self.threshold:
                    self.model.add_edge(src, tgt)
                elif pval2 > pval1 and pval2 > self.threshold:
                    self.model.add_edge(tgt, src)

        for (src, dst) in self.required_edges:
            if src in self.variables and dst in self.variables:
                if not self.model.has_edge(src, dst):
                    if self.verbose:
                        print(f"Forcing required edge: {src} → {dst}")
                    self.model.add_edge(src, dst)
        print(self.model.edges())
        return self.model

    def ensemble_estimate(self, n_runs=5, min_freq=0.5):
        all_edges = []
        for _ in range(n_runs):
            try:
                self.model = DAG()
                model = self.estimate()
                all_edges.extend(model.edges())
            except RuntimeError:
                continue

        if not all_edges and self.verbose:
            print("No edges in ensemble runs")

        edge_counts = Counter(all_edges)
        final_edges = [edge for edge, count in edge_counts.items() if count / n_runs >= min_freq]

        final_model = DAG()
        final_model.add_nodes_from(self.variables)
        final_model.add_edges_from(final_edges)
        return final_model

    def stability_selection(self, n_bootstrap=100, freq_threshold=0.7):
        all_edges = []
        n = len(self.data)
        for _ in range(n_bootstrap):
            idx = np.random.choice(n, size=n, replace=True)
            df_sample = self.data.iloc[idx].reset_index(drop=True)
            try:
                est = DirectLiNGAMEstimator(df_sample,
                                            threshold=self.threshold,
                                            reps=self.reps,
                                            forbidden_edges=self.forbidden_edges,
                                            required_edges=self.required_edges,
                                            verbose=False)
                model = est.estimate(fail_gracefully=True)
                
                all_edges.extend(model.edges())
            except RuntimeError:
                continue

        edge_counts = Counter(all_edges)
        print("Top stable edges:", edge_counts.most_common(5))
        stable_edges = [edge for edge, count in edge_counts.items()
                        if count / n_bootstrap >= freq_threshold]
        

        stable_model = DAG()
        stable_model.add_nodes_from(self.variables)
        stable_model.add_edges_from(stable_edges)
        return stable_model
