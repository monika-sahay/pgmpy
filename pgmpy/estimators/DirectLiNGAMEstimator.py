import numpy as np
import pandas as pd
from collections import Counter
from scipy.stats import kurtosis
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
from pgmpy.base import DAG
from pgmpy.estimators import BaseEstimator
from hyppo.independence import Hsic


def hsic_test(x, y, threshold=0.1, reps=500):
    hsic = Hsic(threshold=threshold)
    try:
        stat, pval = hsic.test(x, y, reps=reps)
    except Exception:
        return 0.0, 1.0
    return stat, pval


class DirectLiNGAMEstimator(BaseEstimator):
    def __init__(self, data, threshold=0.05, reps=1000,
                 forbidden_edges=None, required_edges=None, verbose=False):
        super().__init__(data)
        self.data = pd.DataFrame(StandardScaler().fit_transform(data), columns=self.variables)
        self.threshold = threshold
        self.reps = reps
        self.verbose = verbose
        self.required_edges = set(required_edges or [])
        self.forbidden_edges = set(forbidden_edges or [])
        self.model = None
        self.latent_confounds_ = []

        if self.required_edges & self.forbidden_edges:
            raise ValueError("Conflicting constraints: same edge in required and forbidden sets.")

    def _is_exogenous(self, x, rest):
        if not rest:
            return True
        x_data = self.data[x].values
        if abs(kurtosis(x_data)) < 0.2:
            return True
        rest_data = self.data[list(rest)].values
        # regress x on rest
        reg = Lasso(alpha=0.01).fit(rest_data, x_data)
        residuals = x_data - reg.predict(rest_data)
        # test residuals for independence with each rest var
        pvals = []
        for i in range(rest_data.shape[1]):
            p = hsic_test(residuals.reshape(-1,1), rest_data[:,i].reshape(-1,1),
                          threshold=self.threshold, reps=self.reps)
            pvals.append(p)
        # exogenous if most residual-vs-var tests show independence
        return np.mean(np.array(pvals) > self.threshold) >= 0.8
    
    def fit(self, direction_delta=0.2, bootstrap=False, n_runs=5, min_freq=0.5, fail_gracefully=True):
        # Validate conflicting directed constraints
        for u, v in self.required_edges:
            if (v, u) in self.forbidden_edges:
                raise ValueError(f"Conflict: Required edge {u}→{v} contradicts forbidden edge {v}→{u}")

        self.latent_confounds_ = []

        if bootstrap:
            return self._fit_ensemble(n_runs=n_runs, min_freq=min_freq, direction_delta=direction_delta)
        else:
            return self._fit_once(direction_delta=direction_delta, fail_gracefully=fail_gracefully)

    def _fit_once(self, direction_delta=0.2, fail_gracefully=True):
        remaining = set(self.variables)
        ordered = []

        # if all variables are near-Gaussian, skip to empty DAG
        if np.all(np.abs(kurtosis(self.data.values, axis=0)) < 0.2):
            if self.verbose:
                print("All variables Gaussian-like; skipping.")
            dag = DAG()
            dag.add_nodes_from(self.variables)
            self.model = dag
            return self

        # find a causal ordering by peeling off 'exogenous' vars
        while remaining:
            for var in list(remaining):
                if self._is_exogenous(var, remaining - {var}):
                    ordered.append(var)
                    remaining.remove(var)
                    break
            else:
                # fallback to full graph, but enforce constraints
                if fail_gracefully:
                    dag = self._fallback_dag()
                    # enforce forbidden/required on fallback
                    for u,v in self.forbidden_edges:
                        if dag.has_edge(u,v):
                            dag.remove_edge(u,v)
                    for u,v in self.required_edges:
                        if not dag.has_edge(u,v):
                            dag.add_edge(u,v)
                    self.model = dag
                    return self
                raise RuntimeError("Could not find an exogenous variable; model may not be identifiable.")

        # build adjacency by pairwise HSIC-based orientation
        n = len(self.variables)
        idx = {v:i for i,v in enumerate(self.variables)}
        adj = np.zeros((n,n))
        for i in range(1,len(ordered)):
            for j in range(i):
                X, Y = ordered[j], ordered[i]
                if (X,Y) in self.forbidden_edges or (Y,X) in self.forbidden_edges:
                    continue
                x = self.data[X].values.reshape(-1,1)
                y = self.data[Y].values.reshape(-1,1)
                r1 = y - Lasso(alpha=0.01).fit(x,y).predict(x)
                r2 = x - Lasso(alpha=0.01).fit(y,x).predict(y)
                p1_stat, p1_pval = hsic_test(r1.reshape(-1,1), x)
                p2_stat, p2_pval = hsic_test(r2.reshape(-1,1), y)
                # strict test
                if p1_pval > p2_pval and (p1_pval - p2_pval) > direction_delta and p1_pval > self.threshold:
                    adj[idx[X],idx[Y]] = 1
                elif p2_pval > p1_pval and (p2_pval - p1_pval) > direction_delta and p2_pval > self.threshold:
                    adj[idx[Y],idx[X]] = 1
                elif p1_pval > self.threshold and p2_pval > self.threshold:
                    if p1_pval > p2_pval:
                        adj[idx[X],idx[Y]] = 1
                    elif p2_pval > p1_pval:
                        adj[idx[Y],idx[X]] = 1

        # enforce constraints on adjacency
        for u,v in self.forbidden_edges:
            adj[idx[u],idx[v]] = 0
        for u,v in self.required_edges:
            adj[idx[u],idx[v]] = 1

        # build final DAG
        dag = DAG()
        dag.add_nodes_from(self.variables)
        for u in self.variables:
            for v in self.variables:
                if adj[idx[u],idx[v]] == 1:
                    dag.add_edge(u,v)
        # remove bidirs as latent confounds
        for u,v in list(dag.edges()):
            if dag.has_edge(v,u):
                dag.remove_edge(u,v)
                dag.remove_edge(v,u)
                self.latent_confounds_.append((u,v))
        # final forbidden cleanup
        for u,v in self.forbidden_edges:
            if dag.has_edge(u,v):
                dag.remove_edge(u,v)
        self.model = dag
        return self

    def _fit_ensemble(self, n_runs=5, min_freq=0.5, direction_delta=0.2):
        all_edges = []
        for _ in range(n_runs):
            idx = np.random.choice(len(self.data), size=len(self.data), replace=True)
            sample = self.data.iloc[idx].reset_index(drop=True)
            try:
                est = DirectLiNGAMEstimator(sample, threshold=self.threshold,
                                            reps=self.reps,
                                            required_edges=self.required_edges,
                                            forbidden_edges=self.forbidden_edges)
                est.fit()
                all_edges.extend(est.model.edges())
            except RuntimeError:
                continue
        counts = Counter(all_edges)
        dag = DAG()
        dag.add_nodes_from(self.variables)
        for edge,c in counts.items():
            if c / n_runs >= min_freq:
                dag.add_edge(*edge)
        for u,v in self.forbidden_edges:
            if dag.has_edge(u,v):
                dag.remove_edge(u,v)
        self.model = dag
        return self

    def stability_selection(self, n_bootstrap=20, freq_threshold=0.5):
        all_edges = []
        for _ in range(n_bootstrap):
            sample = self.data.sample(frac=0.9, replace=True)
            est = DirectLiNGAMEstimator(sample,
                                        threshold=self.threshold,
                                        reps=self.reps,
                                        required_edges=self.required_edges,
                                        forbidden_edges=self.forbidden_edges)
            est.fit()
            all_edges.extend(est.model.edges())
        freqs = pd.Series(all_edges).value_counts() / n_bootstrap
        sel = [tuple(e) for e in freqs[freqs >= freq_threshold].index]
        dag = DAG()
        dag.add_nodes_from(self.variables)
        dag.add_edges_from(sel)
        for u,v in self.forbidden_edges:
            if dag.has_edge(u,v):
                dag.remove_edge(u,v)
        return dag

    def _fallback_dag(self):
        dag = DAG()
        dag.add_nodes_from(self.variables)
        for i in self.variables:
            for j in self.variables:
                if i != j:
                    dag.add_edge(i,j)
        return dag

    def get_model(self):
        return self.model

    def get_adjacency_matrix(self):
        idx = {v:i for i,v in enumerate(self.variables)}
        mat = np.zeros((len(idx),len(idx)))
        for u,v in self.model.edges():
            mat[idx[u],idx[v]] = 1
        return mat






# import numpy as np
# import pandas as pd
# from collections import Counter
# from scipy.stats import kurtosis
# from sklearn.linear_model import Lasso
# from sklearn.preprocessing import StandardScaler
# from pgmpy.base import DAG
# from pgmpy.estimators import BaseEstimator
# from hyppo.independence import Hsic


# def hsic_test(x, y, threshold=0.05, reps=1000):
#     hsic = Hsic()
#     stat, pval = hsic.test(x, y, reps=reps)
#     return pval


# class DirectLiNGAMEstimator(BaseEstimator):
#     def __init__(self, data, threshold=0.05, reps=1000,
#                  forbidden_edges=None, required_edges=None, verbose=False):
#         super().__init__(data)
#         self.data = pd.DataFrame(StandardScaler().fit_transform(data), columns=self.variables)
#         self.threshold = threshold
#         self.reps = reps
#         self.verbose = verbose
#         self.required_edges = set(required_edges or [])
#         self.forbidden_edges = set(forbidden_edges or [])
#         self.model = None
#         self.latent_confounds_ = []

#         if self.required_edges & self.forbidden_edges:
#             raise ValueError("Conflicting constraints: same edge in required and forbidden sets.")

#     def _is_exogenous(self, x, rest):
#         if not rest:
#             return True
#         x_data = self.data[x].values
#         if abs(kurtosis(x_data) - 3) < 0.2:
#             return True
#         rest_data = self.data[rest].values
#         reg = Lasso(alpha=0.01).fit(rest_data, x_data)
#         residuals = x_data - reg.predict(rest_data)
#         pvals = [hsic_test(residuals.reshape(-1, 1), rest_data[:, i].reshape(-1, 1),
#                            threshold=self.threshold, reps=self.reps)
#                  for i in range(rest_data.shape[1])]
#         return np.mean(np.array(pvals) > self.threshold) >= 0.8

#     def fit(self, direction_delta=0.2, bootstrap=False, n_runs=5, min_freq=0.5, fail_gracefully=True):
#         if bootstrap:
#             return self._fit_ensemble(n_runs=n_runs, min_freq=min_freq)
#         else:
#             return self._fit_once(direction_delta, fail_gracefully)

#     def _fit_once(self, direction_delta=0.2, fail_gracefully=True):
#         remaining = set(self.variables)
#         ordered = []

#         if all(np.abs(kurtosis(self.data.values, axis=0) - 3) < 0.2):
#             if self.verbose:
#                 print("All variables Gaussian-like; skipping.")
#             self.model = DAG()
#             self.model.add_nodes_from(self.variables)
#             return self

#         while remaining:
#             for var in list(remaining):
#                 others = list(remaining - {var})
#                 if self._is_exogenous(var, others):
#                     ordered.append(var)
#                     remaining.remove(var)
#                     break
#             else:
#                 if fail_gracefully:
#                     self.model = self._fallback_dag()
#                     return self
#                 raise RuntimeError("Could not find exogenous variable")

#         dag = DAG()
#         dag.add_nodes_from(self.variables)

#         for i in range(1, len(ordered)):
#             for j in range(i):
#                 X, Y = ordered[j], ordered[i]
#                 x, y = self.data[X].values.reshape(-1, 1), self.data[Y].values.reshape(-1, 1)
#                 resid1 = y - Lasso(alpha=0.01).fit(x, y).predict(x).reshape(-1, 1)
#                 resid2 = x - Lasso(alpha=0.01).fit(y, x).predict(y).reshape(-1, 1)
#                 pval1, pval2 = hsic_test(resid1, x), hsic_test(resid2, y)

#                 if pval1 > pval2 and (pval1 - pval2) > direction_delta and pval1 > self.threshold:
#                     dag.add_edge(X, Y)
#                 elif pval2 > pval1 and (pval2 - pval1) > direction_delta and pval2 > self.threshold:
#                     dag.add_edge(Y, X)

#         for u, v in list(dag.edges()):
#             if dag.has_edge(v, u):
#                 dag.remove_edge(u, v)
#                 dag.remove_edge(v, u)
#                 self.latent_confounds_.append((u, v))

#         for (u, v) in self.required_edges:
#             dag.add_edge(u, v)

#         # Final cleanup: ensure forbidden edges are removed
#         for (u, v) in self.forbidden_edges:
#             if dag.has_edge(u, v):
#                 dag.remove_edge(u, v)

#         self.model = dag
#         return self

#     def _fit_ensemble(self, n_runs=5, min_freq=0.5):
#         all_edges = []
#         for _ in range(n_runs):
#             idx = np.random.choice(len(self.data), size=len(self.data), replace=True)
#             sample = self.data.iloc[idx].reset_index(drop=True)
#             try:
#                 est = DirectLiNGAMEstimator(sample,
#                                             threshold=self.threshold,
#                                             reps=self.reps,
#                                             required_edges=self.required_edges,
#                                             forbidden_edges=self.forbidden_edges)
#                 est = est.fit()
#                 all_edges.extend(est.model.edges())
#             except RuntimeError:
#                 continue

#         edge_counts = Counter(all_edges)
#         final_edges = [e for e, c in edge_counts.items() if c / n_runs >= min_freq]
#         dag = DAG()
#         dag.add_nodes_from(self.variables)
#         dag.add_edges_from(final_edges)

#         for (u, v) in self.forbidden_edges:
#             if dag.has_edge(u, v):
#                 dag.remove_edge(u, v)

#         self.model = dag
#         return self
    
#     def stability_selection(self, n_bootstrap=20, freq_threshold=0.5):
#         all_edges = []

#         for _ in range(n_bootstrap):
#             sample_df = self.data.sample(frac=0.9, replace=True, random_state=np.random.randint(0, 1e6))
#             est = DirectLiNGAMEstimator(
#                 sample_df,
#                 threshold=self.threshold,
#                 reps=self.reps,
#                 required_edges=self.required_edges,
#                 forbidden_edges=self.forbidden_edges,
#             )
#             est.fit()
#             model = est.get_model()
#             all_edges.extend(model.edges())

#         edge_counts = pd.Series(all_edges).value_counts()
#         edge_freq = edge_counts / n_bootstrap
#         selected_edges = edge_freq[edge_freq >= freq_threshold].index.tolist()

#         dag = DAG()
#         dag.add_nodes_from(self.variables)
#         dag.add_edges_from(selected_edges)

#         # Remove forbidden edges
#         for (u, v) in self.forbidden_edges:
#             if dag.has_edge(u, v):
#                 dag.remove_edge(u, v)

#         return dag

#     def _fallback_dag(self):
#         dag = DAG()
#         dag.add_nodes_from(self.variables)
#         for i in self.variables:
#             for j in self.variables:
#                 if i != j:
#                     dag.add_edge(i, j)
#         return dag

#     def get_model(self):

#         return self.model

#     def get_adjacency_matrix(self):
#         idx = {v: i for i, v in enumerate(self.variables)}
#         mat = np.zeros((len(idx), len(idx)))
#         for u, v in self.model.edges():
#             mat[idx[u], idx[v]] = 1
#         return mat
