""" Utility classes and functions related to causalAssembly.
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
import copy

import networkx as nx
import numpy as np
import pandas as pd


class DAGmetrics:
    """Class to calculate performance metrics for DAGs.
    Make sure that the ground truth and the estimated DAG have the same order of
    rows/columns. If these objects are nx.DiGraphs, make sure that graph.nodes()
    have the same oder or pass a new nodelist to the class when initiating. The
    same can be done for pd.DataFrames. In case `truth` and `est` are np.ndarray
    objects it is the users responsibility to make sure that the objects are
    indeed comparable.
    """

    def __init__(
        self,
        truth: nx.DiGraph | pd.DataFrame | np.ndarray,
        est: nx.DiGraph | pd.DataFrame | np.ndarray,
        nodelist: list = None,
    ):
        if not isinstance(truth, nx.DiGraph | pd.DataFrame | np.ndarray):
            raise TypeError(
                "Ground truth graph has to be one of the permitted classes."
            )

        if not isinstance(est, nx.DiGraph | pd.DataFrame | np.ndarray):
            raise TypeError("Estimated graph has to be one of the permitted classes")

        self.truth = DAGmetrics._convert_to_numpy(truth, nodelist=nodelist)
        self.est = DAGmetrics._convert_to_numpy(est, nodelist=nodelist)

        self.metrics = None

    def _calculate_scores(self):
        """Calculate Precision, Recall and F1 and g score

        Return:
        precision: float
            TP/(TP + FP)
        recall: float
            TP/(TP + FN)
        f1: float
            2*(recall*precision)/(recall+precision)
        gscore: float
            max(0, (TP-FP))/(TP+FN)
        """
        assert (
            self.est.shape == self.truth.shape
            and self.est.shape[0] == self.est.shape[1]
        )
        TP = np.where((self.est + self.truth) == 2, 1, 0).sum(axis=1).sum()
        TP_FP = self.est.sum(axis=1).sum()
        FP = TP_FP - TP
        TP_FN = self.truth.sum(axis=1).sum()

        precision = TP / max(TP_FP, 1)
        recall = TP / max(TP_FN, 1)
        F1 = 2 * (recall * precision) / max((recall + precision), 1)
        gscore = max(0, (TP - FP)) / max(TP_FN, 1)

        return {"precision": precision, "recall": recall, "f1": F1, "gscore": gscore}

    def _shd(self, count_anticausal_twice: bool = True):
        """Calculate Structural Hamming Distance (SHD).

        Args:
            count_anticausal_twice (bool, optional): If edge is pointing in the wrong direction
                it's also missing in the right direction and is counted twice. Defaults to True.
        """
        dist = np.abs(self.truth - self.est)
        if count_anticausal_twice:
            return np.sum(dist)
        else:
            dist = dist + dist.transpose()
            dist[dist > 1] = 1
            return np.sum(dist) / 2

    def collect_metrics(self) -> dict[str, float | int]:
        """Collects all metrics defined in this class in a dict.

        Returns:
            dict[str, float|int]: Metrics calculated
        """
        metrics = self._calculate_scores()
        metrics["shd"] = self._shd()
        self.metrics = metrics

    @classmethod
    def _convert_to_numpy(
        cls,
        graph: nx.DiGraph | pd.DataFrame | np.ndarray,
        nodelist: list = None,
    ):
        if isinstance(graph, np.ndarray):
            return copy.deepcopy(graph)
        elif isinstance(graph, pd.DataFrame):
            if nodelist:
                return copy.deepcopy(graph.reindex(nodelist)[nodelist].to_numpy())
            else:
                return copy.deepcopy(graph.to_numpy())
        elif isinstance(graph, nx.DiGraph):
            return nx.to_numpy_array(G=graph, nodelist=nodelist)
