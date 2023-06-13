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
import random
import string

import networkx as nx
import numpy as np
import pandas as pd
import pytest

from causalAssembly.metrics import DAGmetrics


class TestDAGmetrics:
    @pytest.fixture(scope="class")
    def gt(self):
        names = list(string.ascii_lowercase)[0:5]
        temp_np = np.array(
            [
                [0, 1, 0, 0, 0],
                [0, 0, 1, 0, 0],
                [0, 0, 0, 1, 0],
                [0, 0, 0, 0, 1],
                [0, 0, 0, 0, 0],
            ]
        )
        return pd.DataFrame(temp_np, columns=names, index=names)

    @pytest.fixture(scope="class")
    def est(self):
        names = list(string.ascii_lowercase)[0:5]
        temp_np = np.array(
            [
                [0, 1, 0, 0, 0],
                [0, 0, 1, 0, 0],
                [0, 0, 0, 0, 1],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
            ]
        )
        return pd.DataFrame(temp_np, columns=names, index=names)

    def test_pd_input_works(self, gt, est):
        met = DAGmetrics(truth=gt, est=est)
        assert np.array_equal(met.truth, gt.to_numpy())
        assert np.array_equal(met.est, est.to_numpy())

    def test_nx_input_works(self, gt, est):
        gt_nx = nx.from_pandas_adjacency(gt, create_using=nx.DiGraph)
        est_nx = nx.from_pandas_adjacency(est, create_using=nx.DiGraph)
        met = DAGmetrics(truth=gt_nx, est=est_nx)
        assert np.array_equal(met.truth, gt.to_numpy())
        assert np.array_equal(met.est, est.to_numpy())

    def test_pd_change_order(self, gt, est):
        nodelist = list(string.ascii_lowercase)[0:5]
        random.shuffle(nodelist)
        met = DAGmetrics(truth=gt, est=est, nodelist=nodelist)
        assert np.array_equal(met.truth, gt.reindex(nodelist)[nodelist].to_numpy())
        assert np.array_equal(met.est, est.reindex(nodelist)[nodelist].to_numpy())

    def test_nx_change_order(self, gt, est):
        nodelist = list(string.ascii_lowercase)[0:5]
        random.shuffle(nodelist)

        gt_nx = nx.from_pandas_adjacency(gt, create_using=nx.DiGraph)
        est_nx = nx.from_pandas_adjacency(est, create_using=nx.DiGraph)

        met = DAGmetrics(truth=gt_nx, est=est_nx, nodelist=nodelist)
        assert np.array_equal(met.truth, gt.reindex(nodelist)[nodelist].to_numpy())
        assert np.array_equal(met.est, est.reindex(nodelist)[nodelist].to_numpy())

    def test_metrics_values(self, gt, est):
        met = DAGmetrics(truth=gt, est=est)
        met.collect_metrics()

        assert met.metrics["shd"] == 3
        assert met.metrics["gscore"] >= 0 and met.metrics["gscore"] <= 1
        assert met.metrics["f1"] >= 0 and met.metrics["f1"] <= 1
        assert met.metrics["recall"] >= 0 and met.metrics["recall"] <= 1
        assert met.metrics["precision"] >= 0 and met.metrics["precision"] <= 1
