import numpy as np
import pandas as pd
from pgmpy.base import DAG
from sklearn.linear_model import LinearRegression
from scipy.stats import pearsonr

class DirectLiNGAMEstimator:
    def __init__(self, data: pd.DataFrame):
        self.data = data
        self.variables = list(data.columns)
        self.n_vars = len(self.variables)
        self.model = DAG()

    def _is_exogenous(self, x, rest):
        """
        Test if variable x is exogenous using independence of residuals.
        """
        x_data = self.data[x].values.reshape(-1, 1)
        for y in rest:
            y_data = self.data[y].values
            reg = LinearRegression().fit(x_data, y_data)
            residuals = y_data - reg.predict(x_data)
            corr, _ = pearsonr(residuals, x_data.ravel())
            if abs(corr) > 0.1:  # Tune this threshold
                return False
        return True

    def estimate(self):
        ordered_vars = []
        remaining = set(self.variables)

        while remaining:
            for var in remaining:
                others = list(remaining - {var})
                if self._is_exogenous(var, others):
                    ordered_vars.append(var)
                    remaining.remove(var)
                    break

        # Add edges based on ordering
        for i in range(1, len(ordered_vars)):
            for j in range(i):
                X = self.data[ordered_vars[j]].values.reshape(-1, 1)
                y = self.data[ordered_vars[i]].values
                reg = LinearRegression().fit(X, y)
                coef = reg.coef_[0]
                if abs(coef) > 1e-3:  # small threshold
                    self.model.add_edge(ordered_vars[j], ordered_vars[i])

        return self.model
