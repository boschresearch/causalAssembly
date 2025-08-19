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

import networkx as nx
import numpy as np
import pandas as pd
import pytest

from causalAssembly.dag import DAG
from causalAssembly.pdag import PDAG


class TestDAG:
    """Test class for the DAG class.

    Returns:
        _type_: _description_
    """

    @pytest.fixture(scope="class")
    def example_dag(self) -> DAG:
        """Example dag.

        Returns:
            DAG: _description_
        """
        return DAG(nodes=["A", "B", "C"], edges=[("A", "B"), ("A", "C")])

    def test_instance_is_created(self):
        """Check whether an instance of DAG can be created with nodes only."""
        dag = DAG(nodes=["A", "B", "C"])
        assert isinstance(dag, DAG)

    def test_edges(self, example_dag: DAG):
        """Test edges of the DAG.

        Args:
            example_dag (DAG): _description_
        """
        TWO = 2
        assert example_dag.num_edges == TWO
        assert set(example_dag.edges) == {("A", "B"), ("A", "C")}

    def test_children(self, example_dag: DAG):
        """Test children of the DAG.

        Args:
            example_dag (DAG): _description_
        """
        assert set(example_dag.children(of_node="A")) == {"B", "C"}
        assert example_dag.children(of_node="B") == []
        assert example_dag.children(of_node="C") == []

    def test_parents(self, example_dag: DAG):
        """TEst parents of the DAG.

        Args:
            example_dag (DAG): _description_
        """
        assert example_dag.parents(of_node="A") == []
        assert set(example_dag.parents(of_node="B")) == {"A"}
        assert set(example_dag.parents(of_node="C")) == {"A"}

    def test_from_pandas_adjacency(self, example_dag: DAG):
        """Test import from pandas adjacency matrix.

        Args:
            example_dag (DAG): _description_
        """
        amat = pd.DataFrame(
            [[0, 1, 1], [0, 0, 0], [0, 0, 0]],
            columns=["A", "B", "C"],
            index=["A", "B", "C"],
        )
        from_pandas_pdag = DAG.from_pandas_adjacency(pd_amat=amat)
        assert np.allclose(from_pandas_pdag.adjacency_matrix, amat)
        assert set(from_pandas_pdag.edges) == set(example_dag.edges)

    def test_remove_edge(self, example_dag: DAG):
        """Test removing edges.

        Args:
            example_dag (DAG): _description_
        """
        assert ("A", "C") in example_dag.edges
        example_dag.remove_edge("A", "C")
        assert ("A", "C") not in example_dag.edges
        with pytest.raises(AssertionError, match=r"Edge does not exist in current DAG"):
            example_dag.remove_edge("B", "A")

    def test_remove_node(self, example_dag: DAG):
        """Test removing nodes.

        Args:
            example_dag (DAG): _description_
        """
        assert "C" in example_dag.nodes
        example_dag.remove_node("C")
        assert "C" not in example_dag.nodes

    def test_to_cpdag(self):
        """Test conversion to CPDAG."""
        TWO = 2
        dag = DAG()
        dag.add_edges_from([("A", "B"), ("A", "C")])
        cpdag = dag.to_cpdag()
        assert isinstance(cpdag, PDAG)
        assert cpdag.num_undir_edges == TWO
        assert cpdag.num_dir_edges == 0

    def test_adjacency_matrix(self, example_dag: DAG):
        """Test return of adjacency matrix.

        Args:
            example_dag (DAG): _description_
        """
        amat = example_dag.adjacency_matrix
        assert amat.shape[0] == amat.shape[1] == example_dag.num_nodes
        assert amat.sum().sum() == example_dag.num_edges

    def test_to_networkx(self, example_dag: DAG):
        """Test conversion to NetworkX graph."""
        nxg = example_dag.to_networkx()
        assert isinstance(nxg, nx.DiGraph)
        assert set(nxg.edges) == set(example_dag.edges)

    def test_from_networkx(self, example_dag: DAG):
        """Test conversion from NetworkX graph to DAG."""
        nxg = example_dag.to_networkx()
        from_networkx_dag = DAG.from_nx(nxg)
        assert set(from_networkx_dag.edges) == set(example_dag.edges)

    def test_error_when_cyclic(self):
        """Test that an error is raised when trying to create a cyclic DAG."""
        dag = DAG()
        with pytest.raises(ValueError):
            dag.add_edges_from([("A", "C"), ("C", "D"), ("D", "A")])
