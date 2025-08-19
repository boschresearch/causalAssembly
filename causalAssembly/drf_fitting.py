"""Utility classes and functions related to causalAssembly.

Copyright (c) 2023 Robert Bosch GmbH

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as published
by the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.
This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Affero General Public License for more details.
You should have received a copy of the GNU Affero General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import rpy2.robjects as ro
import rpy2.robjects.packages as rpackages
from rpy2.robjects import numpy2ri, pandas2ri
from rpy2.robjects.conversion import localconverter
from rpy2.robjects.packages import importr
from scipy.stats import gaussian_kde

from causalAssembly.dag import DAG
from causalAssembly.models_dag import ProcessCell, ProductionLineGraph

# Converter for numpy + pandas instead of using deprecated activate()
R_CONVERTER = ro.default_converter + numpy2ri.converter + pandas2ri.converter

base_r_package = importr("base")
utils = importr("utils")

if not rpackages.isinstalled("drf"):
    # select a mirror for R packages
    utils.chooseCRANmirror(ind=1)
    utils.install_packages("drf", repos="https://cloud.r-project.org/")

drf_r_package = importr("drf")


class DRF:
    """Wrapper around the corresponding R package.

    Distributional Random Forests (Cevid et al., 2020).
    Closely adopted from their python wrapper.
    """

    def __init__(self, **fit_params):
        """Initialize the DRF object with fit parameters."""
        self.fit_params = fit_params

    def fit(self, X: pd.DataFrame, Y: pd.DataFrame | pd.Series):
        """Fit DRF in order to estimate conditional distribution P(Y|X=x).

        Args:
            X (pd.DataFrame): Predictor variables.
            Y (pd.DataFrame | pd.Series): Response variable(s).
        """
        self.X_train = X
        self.Y_train = Y

        # Use localconverter
        with localconverter(R_CONVERTER):
            X_r = ro.conversion.py2rpy(X)
            Y_r = ro.conversion.py2rpy(Y)
            self.r_fit_object = drf_r_package.drf(X_r, Y_r, **self.fit_params)

    def produce_sample(
        self,
        newdata: pd.DataFrame,
        random_state: np.random.Generator,
        n: int = 1,
    ) -> np.ndarray:
        """Sample data from fitted drf.

        Args:
            newdata (pd.DataFrame): Data samples to predict from.
            random_state (np.random.Generator): control random state.
            n (int, optional): Number of n-samples to draw. Defaults to 1.

        Returns:
            np.ndarray: New predicted sample of Y.
        """
        with localconverter(R_CONVERTER):
            newdata_r = ro.conversion.py2rpy(newdata)
            r_output = drf_r_package.predict_drf(self.r_fit_object, newdata_r)

            # Convert back to Python
            weights = ro.conversion.rpy2py(base_r_package.as_matrix(r_output[0]))
            Y = ro.conversion.rpy2py(base_r_package.as_matrix(r_output[1]))

        if not isinstance(Y, pd.DataFrame):
            Y = pd.DataFrame(Y)

        sample = np.zeros((newdata.shape[0], Y.shape[1], n))
        for i in range(newdata.shape[0]):
            for j in range(n):
                ids = random_state.choice(range(Y.shape[0]), 1, p=weights[i, :])[0]
                sample[i, :, j] = Y.iloc[ids, :]

        return sample[:, 0, 0]


def fit_drf(graph: ProductionLineGraph | ProcessCell | DAG, data: pd.DataFrame):
    """Fit distributional random forests to the factorization implied by the current graph.

    Args:
        graph (ProductionLineGraph | ProcessCell | DAG): Graph to fit the DRF to.
        data (pd.DataFrame): Columns of dataframe need to match name and order of the graph

    Raises: ValueError: Raises error if columns don't meet this requirement

    Returns: (dict): dict of fitted DRFs.
    """
    tempdata = data.copy()

    if set(graph.nodes).issubset(tempdata.columns):
        tempdata = tempdata[graph.nodes]
    else:
        raise ValueError("Data columns don't match node names.")

    drf_dict = {}
    for node in graph.nodes:
        parents = graph.parents(of_node=node)
        if not parents:
            drf_dict[node] = gaussian_kde(tempdata[node].to_numpy())
        elif parents:
            # default setting as suggested in the paper
            drf_object = DRF(min_node_size=15, num_trees=2000, splitting_rule="FourierMMD")
            X = tempdata[parents]
            Y = tempdata[node]
            drf_object.fit(X, Y)
            drf_dict[node] = drf_object
        else:
            raise ValueError("Unexpected behavior in DRF. Check whether data and DAG match?")
    return drf_dict
