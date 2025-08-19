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

import math
import os
import pickle

import networkx as nx
import numpy as np
import pandas as pd
import pytest

from causalAssembly.models_dag import NodeAttributes, ProcessCell, ProductionLineGraph


class TestProcessCell:
    """Test process Cell.

    Returns:
        _type_: _description_
    """

    @pytest.fixture(scope="class")
    def cell(self):
        """Set up a ProcessCell instance.

        Returns:
            _type_: _description_
        """
        c = ProcessCell(name="PYTEST")
        return c

    @pytest.fixture(scope="class")
    def module(self):
        """Set up a module for testing.

        Returns:
            _type_: _description_
        """
        m = nx.DiGraph()
        m.add_nodes_from(["A", "B", "C"])
        m.add_edges_from([("A", "B"), ("B", "C")])
        return m

    def test_instance_is_created(self):
        """Test whether an instance of ProcessCell can be created with a name."""
        cell = ProcessCell(name="PYTEST")
        assert isinstance(cell, ProcessCell)

    def test_next_module_prefix_works(self):
        """Test whether the next module prefix is generated correctly."""
        # Arrange
        cell = ProcessCell(name="PYTEST")

        # Act
        new_module_prefix = "ABC"
        cell.module_prefix = new_module_prefix

        # Assert
        assert cell.module_prefix == new_module_prefix
        assert cell.next_module_prefix() == "ABC1"

    def test_module_prefix_setter_works(self, cell):
        """TEst that the module prefix can be set correctly.

        Args:
            cell (_type_): _description_
        """
        with pytest.raises(ValueError):
            cell.module_prefix = 1

    def test_add_module_works(self, module):
        """Test that a module can be added to the ProcessCell."""
        TWO = 2
        SIX = 6
        # Arrange
        cell = ProcessCell(name="C01")
        cell.module_prefix = "M"

        # Act
        cell.add_module(module)
        cell.add_module(module)

        # Assert
        assert len(cell.modules) == TWO
        assert len(cell.graph.nodes()) == SIX
        assert cell.next_module_prefix() == "M3"

    def test_connect_by_module_works(self, module):
        """Test that modules can be connected by edges.

        Args:
            module (_type_): _description_
        """
        # Arrange
        cell = ProcessCell(name="PyTestCell")
        m1 = cell.add_module(graph=module)
        m2 = cell.add_module(graph=module)

        # Act
        cell.connect_by_module(m1=m1, m2=m2, edges=[("A", "B")])

        # Assert
        # sloppy but we expect to have one edge more
        expected_no_of_edges = 2 * len(module.edges) + 1
        assert len(cell.graph.edges) == expected_no_of_edges

    @pytest.mark.parametrize(
        "edges",
        [[("Axxx", "B")], [("A", "Bxxx")]],
        ids=["source node invalid", "target node invalid"],
    )
    def test_connect_by_module_fails_with_wrong_node_name(self, module, edges):
        """Test that module connections fails when node names are invalid.

        Args:
            module (_type_): _description_
            edges (_type_): _description_
        """
        # Arrange
        cell = ProcessCell(name="PyTestCell")

        # Act
        m1 = cell.add_module(graph=module)
        m2 = cell.add_module(graph=module)

        # Assert
        with pytest.raises(ValueError):
            cell.connect_by_module(m1=m1, m2=m2, edges=edges)

    def test_node_property(self, module):
        """Test properties of the nodes.

        Args:
            module (_type_): _description_
        """
        # Arrange
        cell = ProcessCell(name="PyTest")

        # Act
        cell.add_module(graph=module)
        cell.add_module(graph=module)

        # Assert
        expected_no_of_nodes = 2 * len(module.nodes)
        assert len(cell.nodes) == expected_no_of_nodes

    def test_repr_is_working(self, module):
        """Test repr is working.

        Args:
            module (_type_): _description_
        """
        # Arrange
        cell = ProcessCell(name="PyTest")

        # Act
        cell.add_module(graph=module)

        # Assert
        assert isinstance(f"{cell}", str)

    @pytest.mark.parametrize(
        "sparsity",
        [0.0, 0.1, 1.0],
        ids=["sparsity=0.0", "sparsity=0.1", "sparsity=1.0"],
    )
    def test_connect_by_random_edges(self, sparsity):
        """Test whether connecting with random edges works.

        Args:
            sparsity (_type_): _description_
        """
        pline = ProductionLineGraph()
        pline.new_cell(name="C1")
        # Arrange
        randomdag = nx.gnp_random_graph(n=100, p=0.0, seed=1, directed=True)
        c = pline.C1

        c.add_module(graph=randomdag, allow_in_edges=True)
        c.add_module(graph=randomdag, allow_in_edges=False)

        # Act
        c.connect_by_random_edges(sparsity=sparsity)

        # Assert
        # expected edges = square of available nodes (nodes in graph) times the sparsity
        expected_edges = int(pow(len(randomdag.nodes), 2) * sparsity)
        assert len(c.graph.edges) == expected_edges

    def test_connect_by_random_edges_fails_with_cyclic_graph(self):
        """Test failure with cyclic graphs."""
        pline = ProductionLineGraph()
        pline.new_cell(name="C1")
        # Arrange
        c = pline.C1
        cyclic_graph = nx.DiGraph()
        cyclic_graph.add_nodes_from(["A", "B", "C", "D", "E"])
        cyclic_graph.add_edges_from([("A", "B"), ("B", "C"), ("C", "A")])

        randomdag = nx.gnp_random_graph(n=100, p=0.0, seed=1, directed=True)
        c.add_module(graph=cyclic_graph, allow_in_edges=True)
        c.add_module(graph=randomdag, allow_in_edges=False)

        # Act and Assert
        with pytest.raises(TypeError):
            c.connect_by_random_edges()

    def test_get_nodes_by_attribute(self, module):
        """Test get nodes by attribute.

        Args:
            module (_type_): _description_
        """
        c = ProcessCell(name="C1")
        c.add_module(graph=module)

        available_attributes = c.get_available_attributes()
        assert isinstance(available_attributes, list)
        assert NodeAttributes.ALLOW_IN_EDGES in available_attributes

    def test_input_cellgraph_directly_works(self):
        """Test whether cellgraph is inputted correctly."""
        THREE = 3
        toygraph = nx.DiGraph()
        toygraph.add_edges_from([("a", "b"), ("a", "c"), ("b", "c")])

        c = ProcessCell(name="toycell")
        c.input_cellgraph_directly(toygraph)
        assert len(c.nodes) == THREE
        assert c.nodes == ["toycell_a", "toycell_b", "toycell_c"]

    def test_ground_truth_cell(self):
        """Test ground truth."""
        pline = ProductionLineGraph()
        pline.new_cell(name="test")
        pline.test.add_random_module()
        pline.test.add_random_module()

        assert isinstance(pline.test.ground_truth, pd.DataFrame)
        assert pline.test.ground_truth.shape[0] == pline.test.ground_truth.shape[1]
        assert pline.test.ground_truth.shape[0] == pline.test.num_nodes
        assert pline.test.ground_truth.sum(axis=1).sum() == pline.test.num_edges


class TestProductionLineGraph:
    """Test ProductionLineGraph."""

    def test_instance_is_created(self):
        """Test whether instance is created."""
        p = ProductionLineGraph()
        assert isinstance(p, ProductionLineGraph)

    def test_getattr_works(self):
        """Test getattr."""
        # Arrange
        station_name = "Station1"
        p = ProductionLineGraph()
        p.new_cell(name=station_name)

        #   # Assert
        assert isinstance(getattr(p, station_name), ProcessCell)
        with pytest.raises(AttributeError):
            p.XXX

    def test_str_representation(self):
        """Test str."""
        p = ProductionLineGraph()
        p.new_cell(name="C1")
        p.C1.add_random_module(n_nodes=10)

        assert isinstance(str(p), str)

    def test_create_cell_works(self):
        """Test cell creation."""
        p = ProductionLineGraph()
        c1 = p.new_cell()

        assert isinstance(c1, ProcessCell)
        assert c1.name == "C0"

    def test_create_cell_with_name_works(self):
        """Test cell with name."""
        p = ProductionLineGraph()
        c1 = p.new_cell(name="PyTest")

        assert c1.name == "PyTest"

    def test_append_same_cell_twice_fails(self):
        """Test failure."""
        p = ProductionLineGraph()
        p.new_cell(name="PyTest")

        with pytest.raises(ValueError):
            p.new_cell(name="PyTest")

    def test_instance_via_cell_number_works(self):
        """Test instance via cell number."""
        n_cells = 10
        p = ProductionLineGraph.via_cell_number(n_cells=n_cells)

        assert len(p.cells) == n_cells

    def test_if_graph_exists(self):
        """Test existence."""
        n_cells = 10
        p = ProductionLineGraph.via_cell_number(n_cells=n_cells)

        assert isinstance(p.graph, nx.DiGraph)

    def test_add_eol_cell(self):
        """Test eol cell."""
        p = ProductionLineGraph()
        p.new_cell()
        p.new_cell(is_eol=True)

        assert isinstance(p.eol_cell, ProcessCell)

    def test_add_eol_cell_twice_fails(self):
        """Test eol twice fails."""
        p = ProductionLineGraph()
        p.new_cell(is_eol=True)

        with pytest.raises(AssertionError):
            p.new_cell(is_eol=True)

    @pytest.mark.parametrize(
        "n_nodes,forward_prob",
        [(10, 0.15), (10, 0.0), (10, 1.0)],
        ids=["some edges", "zero edges", "all edges"],
    )
    def test_connect_cells_works_with_single_cell(self, n_nodes, forward_prob):
        """Test connect."""
        # Arrange
        p = ProductionLineGraph()
        p.new_cell(name="C1")
        p.new_cell(name="C2")
        p.C1.add_random_module(n_nodes=n_nodes)
        p.C2.add_random_module(n_nodes=n_nodes)

        no_of_edges = len(p.graph.edges)
        no_of_expected_edges = no_of_edges + int((n_nodes * n_nodes) * forward_prob)

        # Act
        p.connect_cells(forward_probs=[forward_prob])

        # Assert
        assert len(p.graph.edges) == no_of_expected_edges

    @pytest.mark.parametrize(
        "n_nodes,forward_probs",
        [
            (10, [0.1, 0.1]),
            (10, [0.0, 0.1]),
            (10, [0.1, 0.0]),
            (10, [1, 0.1]),
            (10, [0.1, 1.0]),
            (10, [1, 1.0]),
        ],
        ids=[
            "connect some cells",
            "first cell zero",
            "second cell zero",
            "first cell 1",
            "second cell 1",
            "both cells 1",
        ],
    )
    def test_connect_cells_works_with_multiple_cells(self, n_nodes, forward_probs):
        """Test connect with multiple cells.

        Args:
            n_nodes (_type_): _description_
            forward_probs (_type_): _description_
        """
        # Arrange
        p = ProductionLineGraph()

        p.new_cell(name="C1")
        p.new_cell(name="C2")
        p.new_cell(name="C3")

        p.C1.add_random_module(n_nodes=n_nodes)
        p.C2.add_random_module(n_nodes=n_nodes)
        p.C3.add_random_module(n_nodes=n_nodes)

        no_of_edges = len(p.graph.edges)

        n_nodes_squared = n_nodes * n_nodes
        no_of_expected_edges = no_of_edges + n_nodes_squared * (
            2 * forward_probs[0] + forward_probs[1]
        )

        # Act
        p.connect_cells(forward_probs=forward_probs)

        # Assert
        assert len(p.graph.edges) == int(no_of_expected_edges)

    @pytest.mark.parametrize(
        "n_nodes,forward_prob",
        [(10, 0.15), (10, 0.0), (10, 1.0)],
        ids=["some edges", "zero edges", "all edges"],
    )
    def test_connect_cells_works_with_eol_cell(self, n_nodes, forward_prob):
        """Test connect with eol.

        Args:
            n_nodes (_type_): _description_
            forward_prob (_type_): _description_
        """
        # Arrange
        p = ProductionLineGraph()

        p.new_cell(name="C1")
        p.new_cell(name="C2")
        p.new_cell(name="C3", is_eol=True)

        p.C1.add_random_module(n_nodes=n_nodes)
        p.C2.add_random_module(n_nodes=n_nodes)
        p.C3.add_random_module(n_nodes=n_nodes)

        no_of_edges = len(p.graph.edges)

        no_of_expected_edges = no_of_edges + 3 * int((n_nodes * n_nodes) * forward_prob)

        # Act
        p.connect_cells(forward_probs=[forward_prob])

        # Assert
        assert len(p.graph.edges) == no_of_expected_edges

    def test_connect_across_cells_manually(self):
        """TEst manual connection."""
        n_nodes = 10
        prob = 0.1
        p = ProductionLineGraph()
        p.new_cell(name="C1")
        p.new_cell(name="C2")
        p.C1.add_random_module(n_nodes=n_nodes, p=prob)
        p.C2.add_random_module(n_nodes=n_nodes, p=prob)

        edgelist = [("C1_M1_1", "C2_M1_1"), ("C1_M1_1", "C2_M1_5")]
        p.connect_across_cells_manually(edges=edgelist)

        edges_in_C1 = len(p.C1.modules["M1"].edges)
        edges_in_C2 = len(p.C2.modules["M1"].edges)

        assert p.graph.has_edge(*("C1_M1_1", "C2_M1_1"))
        assert not p.graph.has_edge(*("C2_M1_1", "C1_M1_1"))
        assert len(p.graph.edges) == edges_in_C1 + edges_in_C2 + len(edgelist)

    def test_acyclicity_error(self):
        """Test acycicity."""
        n_nodes = 5
        prob = 0.1
        p = ProductionLineGraph()
        p.new_cell(name="C1")
        p.new_cell(name="C2")
        p.C1.add_random_module(n_nodes=n_nodes, p=prob)
        p.C2.add_random_module(n_nodes=n_nodes, p=prob)

        # create cycle
        edgelist = [
            ("C1_M1_1", "C2_M1_1"),
            ("C2_M1_1", "C2_M1_3"),
            ("C2_M1_3", "C1_M1_1"),
        ]
        p.connect_across_cells_manually(edges=edgelist)

        with pytest.raises(TypeError):
            print(p.graph.edges)

    def test_ground_truth_visible(self):
        """Test ground truth visible."""
        n_nodes = 5
        prob = 0.1
        p = ProductionLineGraph()
        p.new_cell(name="C1")
        p.new_cell(name="C2")
        p.new_cell(name="C3")
        p.C1.add_random_module(n_nodes=n_nodes, p=prob)
        p.C2.add_random_module(n_nodes=n_nodes, p=prob)
        p.C3.add_random_module(n_nodes=n_nodes, p=prob)
        p.connect_cells(forward_probs=[0.1])

        assert np.allclose(p.ground_truth, p.ground_truth_visible)
        assert len(p.visible_nodes()) == len(p.graph.nodes)
        assert len(p.hidden_nodes()) == 0
        assert len(p._pairs_with_hidden_confounders()) == 0
        assert len(p._pairs_with_hidden_mediators()) == 0

    def test_ground_truth_hidden(self):
        """TEst hidden gt."""
        TWO = 2
        edges1 = [(1, 2), (2, 3)]
        edges2 = [(1, 3), (2, 3)]
        edges3 = [(1, 2), (2, 3)]

        p = ProductionLineGraph()
        p.new_cell(name="C1")
        p.new_cell(name="C2")
        p.new_cell(name="C3")
        p.C1.add_module(nx.DiGraph(edges1), mark_hidden=[3])
        p.C2.add_module(nx.DiGraph(edges2), allow_in_edges=False)
        p.C3.add_module(nx.DiGraph(edges3), mark_hidden=[1, 2])

        edge_list = [
            ("C1_M1_3", "C2_M1_2"),
            ("C1_M1_3", "C2_M1_1"),
            ("C2_M1_3", "C3_M1_1"),
        ]
        p.connect_across_cells_manually(edge_list)

        assert p.hidden_nodes() == ["C1_M1_3", "C3_M1_1", "C3_M1_2"]

        ## pairs with mediators should be:
        # (C2_M1_3, C3_M1_3)

        assert p._pairs_with_hidden_mediators() == [("C2_M1_3", "C3_M1_3")]

        ## pairs with hidden confounder should be
        # {(C2_M1_1, C2_M1_2): C1_M1_3}
        assert p._pairs_with_hidden_confounders() == {("C2_M1_1", "C2_M1_2"): ["C1_M1_3"]}

        assert p.ground_truth_visible.loc[("C2_M1_1", "C2_M1_2")] == TWO
        assert p.ground_truth_visible.loc[("C2_M1_2", "C2_M1_1")] == TWO

    def test_input_cellgraph_directly(self):
        """Test input directly."""
        SIX = 6
        FOUR = 4
        dag1 = nx.DiGraph([(0, 1), (1, 2)])
        dag2 = nx.DiGraph([(3, 4), (3, 5)])

        testline = ProductionLineGraph()
        testline.new_cell(name="Station1")
        testline.Station1.input_cellgraph_directly(graph=dag1)
        testline.new_cell(name="Station2")
        testline.Station2.input_cellgraph_directly(graph=dag2)

        assert testline.num_nodes == SIX
        assert testline.num_edges == FOUR
        assert testline.sparsity == pytest.approx(4 / math.comb(6, 2))

    def test_drf_size(self):
        """TEst drf size."""
        testline = ProductionLineGraph()
        assert not testline.drf

    def test_drf_error(self):
        """TEst drf error."""
        testline = ProductionLineGraph()
        with pytest.raises(ValueError):
            testline.sample_from_drf()

    def test_from_nx(self):
        """Test from nx."""
        TWO = 2
        nx_graph = nx.DiGraph(
            [("1", "2"), ("1", "3"), ("1", "4"), ("2", "5"), ("2", "6"), ("5", "6")]
        )
        pd_graph = nx.to_pandas_adjacency(G=nx_graph)
        cell_mapper = {"cell1": ["1", "2", "3", "4"], "cell2": ["5", "6"]}
        pline_from_nx = ProductionLineGraph.from_nx(g=nx_graph, cell_mapper=cell_mapper)

        assert len(pline_from_nx.cells) == TWO
        assert set(pline_from_nx.cell1.nodes) == {
            "cell1_1",
            "cell1_2",
            "cell1_3",
            "cell1_4",
        }
        assert set(pline_from_nx.cell2.nodes) == {"cell2_5", "cell2_6"}
        with pytest.raises(ValueError):
            ProductionLineGraph.from_nx(pd_graph, cell_mapper=cell_mapper)

    def test_save_and_load_drf(self, tmp_path_factory):
        """Test save and load.

        Args:
            tmp_path_factory (_type_): _description_
        """
        basedir = tmp_path_factory.mktemp("data")
        filename = "drf.pkl"

        line1 = ProductionLineGraph()
        line1.drf = np.array([[1, 2, 3]])  # type: ignore
        line1.save_drf(filename=filename, location=basedir)

        line2 = ProductionLineGraph()
        line2.drf = ProductionLineGraph.load_drf(filename=filename, location=basedir)
        assert np.array_equal(line2.drf, np.array([[1, 2, 3]]))  # type: ignore

    def test_pickleability(self, tmp_path):
        """Test pickle.

        Args:
            tmp_path (_type_): _description_
        """
        SEVEN = 7
        TEN = 10
        # Arrange
        filename_path = os.path.join(tmp_path, "pline.pkl")
        pline = ProductionLineGraph()
        pline.new_cell(name="Station1")
        pline.new_cell(name="Station2")

        pline.Station1.add_random_module(n_nodes=7)
        pline.Station2.add_random_module(n_nodes=10)

        new_edge = ("S1_M1_0", "S2_M1_0")
        pline.connect_across_cells_manually(edges=[new_edge])

        # Act
        with open(filename_path, "wb") as fp:
            pickle.dump(pline, fp)

        with open(filename_path, "rb") as fp:
            pline_reloaded = pickle.load(fp)

        # Assert
        assert new_edge in pline_reloaded.edges
        assert len(pline.Station1.nodes) == SEVEN
        assert len(pline.Station2.nodes) == TEN

    def test_copy(self):
        """Test copy."""
        pline = ProductionLineGraph()
        pline.new_cell(name="Station1")
        pline.new_cell(name="Station2")

        pline.Station1.add_random_module(n_nodes=7)
        pline.Station2.add_random_module(n_nodes=10)

        copyline = pline.copy()

        assert set(pline.nodes) == set(copyline.nodes)
        assert set(pline.edges) == set(copyline.edges)
        assert set(pline.Station1.edges) == set(copyline.Station1.edges)
        assert set(pline.Station2.edges) == set(copyline.Station2.edges)
        assert pline.cell_order == copyline.cell_order

    def test_within_edges_with_empty_cells_raises_error(self):
        """Test error."""
        # Setup
        pline = ProductionLineGraph()

        # Act
        with pytest.raises(AssertionError):
            pline.within_adjacency

    def test_between_edges_with_empty_cells_raises_error(self):
        """Test error."""
        # Setup
        pline = ProductionLineGraph()

        # Act
        with pytest.raises(AssertionError):
            print(pline.between_adjacency)

    def test_within_edges_adjacency_matrix(self):
        """Test within amat."""
        # Setup
        nx_graph = nx.DiGraph(
            [("1", "2"), ("1", "3"), ("1", "4"), ("2", "5"), ("2", "6"), ("5", "6")]
        )
        cell_mapper = {"cell1": ["1", "2", "3", "4"], "cell2": ["5", "6"]}
        pline = ProductionLineGraph.from_nx(g=nx_graph, cell_mapper=cell_mapper)

        # Act
        within_amat = pline.within_adjacency

        # Assert

        pd.testing.assert_frame_equal(
            left=within_amat.loc[pline.cell1.nodes, pline.cell1.nodes],
            right=pline.cell1.ground_truth,
        )
        pd.testing.assert_frame_equal(
            left=within_amat.loc[pline.cell2.nodes, pline.cell2.nodes],
            right=pline.cell2.ground_truth,
        )

        assert (
            within_amat.loc["cell1_2", :].sum() == 0
            and pline.ground_truth.loc["cell1_2", :].sum() != 0
        )  # type: ignore

    def test_between_edges_adjacency_matrix(self):
        """Test between edges amat."""
        TWO = 2
        # Setup
        nx_graph = nx.DiGraph(
            [("1", "2"), ("1", "3"), ("1", "4"), ("2", "5"), ("2", "6"), ("5", "6")]
        )
        cell_mapper = {"cell1": ["1", "2", "3", "4"], "cell2": ["5", "6"]}
        pline = ProductionLineGraph.from_nx(g=nx_graph, cell_mapper=cell_mapper)

        # Act
        between_amat = pline.between_adjacency

        # Assert

        assert (
            between_amat.loc["cell1_2", :].sum() == TWO
            and pline.ground_truth.loc["cell1_2", :].sum() == TWO
        )  # type: ignore
        assert between_amat.loc[pline.cell1.nodes, pline.cell1.nodes].sum().sum() == 0
        assert between_amat.loc[pline.cell2.nodes, pline.cell2.nodes].sum().sum() == 0

    def test_interventional_drf_error(self):
        """Test interventional drf."""
        testline = ProductionLineGraph()
        with pytest.raises(ValueError):
            testline.sample_from_interventional_drf()
