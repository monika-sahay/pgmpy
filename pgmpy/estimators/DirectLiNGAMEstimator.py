# pgmpy/estimaors/DirectLinGameEstimator.py
import numpy as np
import pandas as pd
from pgmpy.base import DAG
from sklearn.linear_model import Lasso
from hyppo.independence import Hsic
from sklearn.preprocessing import StandardScaler
from collections import Counter
from scipy.stats import kurtosis
import networkx as nx


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
        self.latent_confounds_ = []

        conflict = self.required_edges & self.forbidden_edges
        if conflict:
            raise ValueError(f"Contradictory constraints found: {conflict}")

    def _is_exogenous(self, x, rest):
        if not rest:
            return True
        x_data = self.data[x].values
        if kurtosis(x_data) < 3:
            if self.verbose:
                print(f"⚠️ {x} is too Gaussian-like; skipping.")
            return False
        rest_data = self.data[rest].values
        reg = Lasso(alpha=0.01).fit(rest_data, x_data)
        residuals = x_data - reg.predict(rest_data)
        pvals = []
        for i in range(rest_data.shape[1]):
            p = hsic_test(residuals.reshape(-1, 1), rest_data[:, i].reshape(-1, 1),
                          threshold=self.threshold, reps=self.reps, verbose=self.verbose)
            pvals.append(p)

        pass_rate = sum(p > self.threshold for p in pvals) / len(pvals)
        return pass_rate >= 0.8

    def estimate(self, direction_delta=0.2, fail_gracefully=True):
        ordered_vars = []
        if self.required_edges:
            forced_vars = set(src for src, _ in self.required_edges)
            ordered_vars.extend(forced_vars)
            remaining = set(self.variables) - forced_vars
        else:
            remaining = set(self.variables)

        # Gaussian check fallback: skip all edges if all variables ~Gaussian
        if all(np.abs(kurtosis(self.data.values, axis=0) - 3) < 0.2):
            if self.verbose:
                print("All variables are Gaussian-like. Returning empty DAG.")
            return DAG()
        confounder_pairs = []
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
                    for i in range(len(self.variables)):
                        for j in range(len(self.variables)):
                            if i != j:
                                self.model.add_edge(self.variables[i], self.variables[j])

                    corr_matrix = np.corrcoef(self.data.T)
                    prune_threshold = 0.05
                    to_remove = []
                    for i, src in enumerate(self.variables):
                        for j, tgt in enumerate(self.variables):
                            if i != j and abs(corr_matrix[i][j]) < prune_threshold:
                                to_remove.append((src, tgt))
                    for edge in to_remove:
                        if self.model.has_edge(*edge):
                            self.model.remove_edge(*edge)

                    # Remove forbidden edges explicitly
                    for edge in list(self.model.edges()):
                        if edge in self.forbidden_edges:
                            self.model.remove_edge(*edge)
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
                reg1 = Lasso(alpha=0.01).fit(x, y)
                resid1 = y - reg1.predict(x)
                pval1 = hsic_test(resid1, x, threshold=self.threshold, reps=self.reps, verbose=self.verbose)

                reg2 = Lasso(alpha=0.01).fit(y, x)
                resid2 = x - reg2.predict(y)
                pval2 = hsic_test(resid2, y, threshold=self.threshold, reps=self.reps, verbose=self.verbose)

                if pval1 > pval2 and (pval1 - pval2) > direction_delta and pval1 > self.threshold:
                    self.model.add_edge(src, tgt)
                elif pval2 > pval1 and (pval2 - pval1) > direction_delta and pval2 > self.threshold:
                    self.model.add_edge(tgt, src)
           
            for u, v in list(self.model.edges()):
                if self.model.has_edge(v, u):
                    self.model.remove_edge(u, v)
                    self.model.remove_edge(v, u)
                    confounder_pairs.append((u, v))

            if confounder_pairs:
                print("⚠️ Suspected latent confounders (removed bidirectional edges):", confounder_pairs)
            
        self.latent_confounds_ = confounder_pairs

        for (src, dst) in self.required_edges:
            if src in self.variables and dst in self.variables:
                if not self.model.has_edge(src, dst):
                    if self.verbose:
                        print(f"Forcing required edge: {src} → {dst}")
                    self.model.add_edge(src, dst)

        for edge in list(self.model.edges()):
            if edge in self.forbidden_edges:
                if self.verbose:
                    print(f"Removing forbidden edge: {edge}")
                self.model.remove_edge(*edge)

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
    
    def get_probabilistic_adjacency(self, all_edges, n_bootstrap):
        matrix = pd.DataFrame(0, index=self.variables, columns=self.variables)
        for (src, tgt) in all_edges:
            matrix.loc[src, tgt] += 1
        matrix = matrix / n_bootstrap
        return matrix

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
        print("Top stable edges:", edge_counts.most_common(10))
        stable_edges = [edge for edge, count in edge_counts.items()
                        if count / n_bootstrap >= freq_threshold]
        weak_edges = [edge for edge, count in edge_counts.items() if 0.3 <= count / n_bootstrap < freq_threshold]
        print("\n🔍 Weak signal edges (below threshold):")
        for edge in weak_edges:
            print(edge, f"→ {edge_counts[edge]}/{n_bootstrap}")
        self.p_adj_matrix_ = self.get_probabilistic_adjacency(all_edges, n_bootstrap)

        stable_model = DAG()
        stable_model.add_nodes_from(self.variables)
        stable_model.add_edges_from(stable_edges)
        
        return stable_model
    

    def pairwise_scores(self):
        """
        Compute pairwise causal scores using independence of residuals.

        Returns:
            dict: Keys are tuples (X, Y), values are p-values for independence.
        """
        scores = {}
        for i, X in enumerate(self.variables):
            for j, Y in enumerate(self.variables):
                if i == j:
                    continue

                x = self.data[X].values.reshape(-1, 1)
                y = self.data[Y].values.reshape(-1, 1)

                # Fit X → Y
                reg_xy = Lasso(alpha=0.01).fit(x, y)
                resid_xy = y - reg_xy.predict(x).reshape(-1, 1)
                pval_xy = hsic_test(resid_xy, x, threshold=self.threshold, reps=self.reps)

                # Fit Y → X
                reg_yx = Lasso(alpha=0.01).fit(y, x)
                resid_yx = x - reg_yx.predict(y).reshape(-1, 1)
                pval_yx = hsic_test(resid_yx, y, threshold=self.threshold, reps=self.reps)

                # Higher p-value ⇒ more independent ⇒ better direction
                direction = (X, Y) if pval_xy > pval_yx else (Y, X)
                scores[(X, Y)] = {
                    "X->Y_pval": round(pval_xy, 4),
                    "Y->X_pval": round(pval_yx, 4),
                    "preferred": direction,
                }

        return scores
    

    def pairwise_graph(self, min_diff=0.0):  # You can still use min_diff if you like
        G = nx.DiGraph()
        G.add_nodes_from(self.variables)

        scores = self.pairwise_scores()
        for (X, Y), s in scores.items():
            p_xy = s["X->Y_pval"]
            p_yx = s["Y->X_pval"]

            if p_xy < 0.05 and p_xy < p_yx:
                G.add_edge(X, Y)
            elif p_yx < 0.05 and p_yx < p_xy:
                G.add_edge(Y, X)

        return G



