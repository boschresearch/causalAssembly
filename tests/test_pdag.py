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

from causalAssembly.pdag import PDAG, dag2cpdag

TWO = 2
THREE = 3
FOUR = 4


class TestPDAG:
    """Test PDAG class."""

    @pytest.fixture(scope="class")
    def mixed_pdag(self) -> PDAG:
        """Set up pdag."""
        pdag = PDAG(
            nodes=["A", "B", "C"],
            dir_edges=[("A", "B"), ("A", "C")],
            undir_edges=[("B", "C")],
        )
        return pdag

    def test_instance_is_created(self):
        """Test instance."""
        pdag = PDAG(nodes=["A", "B", "C"])
        assert isinstance(pdag, PDAG)

    def test_dir_edges(self):
        """Test dir edges."""
        pdag = PDAG(nodes=["A", "B", "C"], dir_edges=[("A", "B"), ("A", "C")])
        assert pdag.num_dir_edges == TWO
        assert pdag.num_undir_edges == 0
        assert set(pdag.dir_edges) == {("A", "B"), ("A", "C")}

    def test_undir_edges(self):
        """Test undir edges."""
        pdag = PDAG(nodes=["A", "B", "C"], undir_edges=[("A", "B"), ("A", "C")])
        assert pdag.num_dir_edges == 0
        assert pdag.num_undir_edges == TWO
        assert set(pdag.undir_edges) == {("A", "B"), ("A", "C")}

    def test_mixed_edges(self, mixed_pdag: PDAG):
        """Test mixed edges."""
        assert mixed_pdag.num_dir_edges == TWO
        assert mixed_pdag.num_undir_edges == 1
        assert set(mixed_pdag.dir_edges) == {("A", "B"), ("A", "C")}
        assert set(mixed_pdag.undir_edges) == {("B", "C")}

    def test_children(self, mixed_pdag: PDAG):
        """Test child edges."""
        assert mixed_pdag.children(node="A") == {"B", "C"}
        assert mixed_pdag.children(node="B") == set()
        assert mixed_pdag.children(node="C") == set()

    def test_parents(self, mixed_pdag: PDAG):
        """Test parent edges."""
        assert mixed_pdag.parents(node="A") == set()
        assert mixed_pdag.parents(node="B") == {"A"}
        assert mixed_pdag.parents(node="C") == {"A"}

    def test_neighbors(self, mixed_pdag: PDAG):
        """Test neighbors."""
        assert mixed_pdag.neighbors(node="C") == {"B", "A"}
        assert mixed_pdag.undir_neighbors(node="C") == {"B"}
        assert mixed_pdag.is_adjacent(i="B", j="C")

    def test_from_pandas_adjacency(self, mixed_pdag: PDAG):
        """Test import from pandas.

        Args:
            mixed_pdag (PDAG): _description_
        """
        amat = pd.DataFrame(
            [[0, 1, 1], [0, 0, 1], [0, 1, 0]],
            columns=["A", "B", "C"],
            index=["A", "B", "C"],
        )
        from_pandas_pdag = PDAG.from_pandas_adjacency(pd_amat=amat)
        assert np.allclose(from_pandas_pdag.adjacency_matrix, amat)
        assert set(from_pandas_pdag.dir_edges) == set(mixed_pdag.dir_edges)
        assert from_pandas_pdag.num_undir_edges == mixed_pdag.num_undir_edges

    def test_remove_edge(self, mixed_pdag: PDAG):
        """Test remove edges.

        Args:
            mixed_pdag (PDAG): _description_
        """
        assert ("A", "C") in mixed_pdag.dir_edges
        mixed_pdag.remove_edge("A", "C")
        assert ("A", "C") not in mixed_pdag.dir_edges
        with pytest.raises(AssertionError, match=r"Edge does not exist in current PDAG"):
            mixed_pdag.remove_edge("B", "A")

    def test_change_undir_edge_to_dir_edge(self, mixed_pdag: PDAG):
        """Test change.

        Args:
            mixed_pdag (PDAG): _description_
        """
        assert ("B", "C") in mixed_pdag.undir_edges or (
            "C",
            "B",
        ) in mixed_pdag.undir_edges
        mixed_pdag.undir_to_dir_edge(tail="C", head="B")
        assert ("C", "B") in mixed_pdag.dir_edges
        assert ("B", "C") not in mixed_pdag.dir_edges
        assert ("B", "C") not in mixed_pdag.undir_edges
        assert ("C", "B") not in mixed_pdag.undir_edges

    def test_remove_node(self, mixed_pdag: PDAG):
        """Test remove nodes.

        Args:
            mixed_pdag (PDAG): _description_
        """
        assert "C" in mixed_pdag.nodes
        mixed_pdag.remove_node("C")
        assert "C" not in mixed_pdag.nodes

    def test_to_dag(self, mixed_pdag: PDAG):
        """Test conversion to DAG.

        Args:
            mixed_pdag (PDAG): _description_
        """
        dag = mixed_pdag.to_dag()
        assert nx.is_directed_acyclic_graph(dag)
        assert set(mixed_pdag.dir_edges).issubset(set(dag.edges))

    def test_adjacency_matrix(self, mixed_pdag: PDAG):
        """Test return of adjacency matrix.

        Args:
            mixed_pdag (PDAG): _description_
        """
        amat = mixed_pdag.adjacency_matrix
        assert amat.shape[0] == amat.shape[1] == mixed_pdag.nnodes
        assert amat.sum().sum() == mixed_pdag.num_dir_edges + 2 * mixed_pdag.num_undir_edges

    def test_dag2cpdag(self):
        """Test conversion from DAG to CPDAG."""
        dag1 = nx.DiGraph([("1", "2"), ("2", "3"), ("3", "4")])
        cpdag1 = dag2cpdag(dag=dag1)
        assert cpdag1.num_dir_edges == 0
        assert cpdag1.num_undir_edges == THREE

        dag2 = nx.DiGraph([("1", "3"), ("2", "3")])
        cpdag2 = dag2cpdag(dag=dag2)
        assert set(cpdag2.dir_edges) == set(dag2.edges)
        assert cpdag2.num_undir_edges == 0

        dag3 = nx.DiGraph([("1", "3"), ("2", "3"), ("1", "4")])
        cpdag3 = dag2cpdag(dag=dag3)
        assert cpdag3.num_dir_edges == TWO
        assert cpdag3.num_undir_edges == 1

    def test_example_a_to_allDAGs(self):
        """Test example PDAG to allDAGs."""
        # Set up CPDAG: a - c - b -> MEC has 3 Members
        pdag = nx.Graph([("a", "c"), ("b", "c")])
        amat = nx.to_pandas_adjacency(pdag)

        # Inititiate
        example_pdag = PDAG.from_pandas_adjacency(pd_amat=amat)

        # Act
        all_dags = example_pdag.to_allDAGs()

        assert len(all_dags) == THREE
        assert all([isinstance(dag, nx.DiGraph) for dag in all_dags])

    def test_example_b_to_allDAGs(self):
        """Test example PDAG to allDAGs."""
        # Set up CPDAG: a - (b,c,d) -> MEC has 4 Members
        pdag = nx.Graph([("a", "b"), ("a", "c"), ("a", "d")])
        amat = nx.to_pandas_adjacency(pdag)

        # Inititiate
        example_pdag = PDAG.from_pandas_adjacency(pd_amat=amat)

        # Act
        all_dags = example_pdag.to_allDAGs()

        assert len(all_dags) == FOUR
        assert all([isinstance(dag, nx.DiGraph) for dag in all_dags])

    def test_empty_graph_to_allDAGs(self):
        """Test empty graph to allDAGs."""
        # Set up empty PDAG, has exaclty one DAG that is the same as the PDAG.
        pdag = nx.Graph()
        pdag.add_nodes_from(["a", "b", "c", "d"])
        amat = nx.to_pandas_adjacency(pdag)

        # Inititiate
        empty_pdag = PDAG.from_pandas_adjacency(pd_amat=amat)

        # Act
        all_dags = empty_pdag.to_allDAGs()

        assert len(all_dags) == 1
        assert all_dags[0].edges == pdag.edges
        assert set(all_dags[0].nodes) == set(pdag.nodes)

    def test_to_random_dag(self):
        """Test to random DAG."""
        # Set up CPDAG: a - (b,c,d) -> MEC has 4 Members
        pdag = nx.Graph([("a", "b"), ("a", "c"), ("a", "d")])
        amat = nx.to_pandas_adjacency(pdag)

        # Inititiate
        example_pdag = PDAG.from_pandas_adjacency(pd_amat=amat)

        # Act
        random_dag = example_pdag.to_random_dag()

        assert isinstance(random_dag, nx.DiGraph)
        assert nx.is_directed_acyclic_graph(random_dag)
