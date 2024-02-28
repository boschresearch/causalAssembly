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
from __future__ import annotations

import itertools
import json
import logging
import pickle
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import requests
from matplotlib.collections import PatchCollection
from matplotlib.patches import BoxStyle, FancyBboxPatch
from networkx.readwrite import json_graph
from scipy.stats import gaussian_kde

from causalAssembly.dag_utils import _bootstrap_sample, tuples_from_cartesian_product
from causalAssembly.pdag import PDAG, dag2cpdag

logger = logging.getLogger(__name__)

DATA_SOURCE = "https://raw.githubusercontent.com/boschresearch/causalAssembly/main/data"
DATA_DATASET = f"{DATA_SOURCE}/data_sets/synthetic_data.csv"
DATA_GROUND_TRUTH = f"{DATA_SOURCE}/ground_truth/ground_truth.json"


@dataclass
class NodeAttributes:
    ALLOW_IN_EDGES = "allow_in_edges"
    HIDDEN = "is_hidden"


def _sample_from_drf(
    prod_object: ProductionLineGraph | ProcessCell, size=10, smoothed: bool = True
) -> pd.DataFrame:
    if not prod_object.drf:
        raise ValueError("Nothing to sample from. Learn DRF first!")
    sample_dict = {}
    for node in prod_object.causal_order:
        if isinstance(prod_object.drf[node], gaussian_kde):
            # Node has no parents, generate a sample using bootstrapping
            #
            if smoothed:
                sample_dict[node] = prod_object.drf[node].resample(
                    size=size, seed=prod_object.random_state
                )[0]
            else:
                sample_dict[node] = _bootstrap_sample(
                    rng=prod_object.random_state,
                    data=prod_object.drf[node].dataset[0],
                    size=size,
                )
        else:
            parents = prod_object.parents(of_node=node)
            new_data = pd.DataFrame({col: sample_dict[col] for col in parents})
            # new_data = pd.DataFrame(sample_dict[parents])
            forest = prod_object.drf[node]
            sample_dict[node] = forest.produce_sample(
                newdata=new_data, random_state=prod_object.random_state
            )
    new_df = pd.DataFrame(sample_dict)
    return new_df[prod_object.nodes]


class ProcessCell:
    """
    Representation of a single Production Line Cell
    (to model a station / a process in a production line
    environment).

    A Cell can contain multiple modules (sub-graphs, which are nx.DiGraph objects).

    Note that none of the term Cell, Process or Module has a strict definition.
    The convention is based on a production line, consisting of several cells which
    are to be modeled by means of smaller graphs (modules) by a user of the repository.

    """

    def __init__(self, name: str):
        self.name = name
        self.graph: nx.DiGraph = nx.DiGraph()

        self.description: str = ""  # description of the cell.

        self.__module_prefix = "M"  # M01 vs M1?
        self.modules: dict[str, nx.DiGraph] = dict()  # {'M1': nx.DiGraph, 'M2': nx.DiGraph}
        self.module_connectors: list[tuple] = list()

        self.is_eol = False
        self.random_state = None
        self.drf: dict = dict()

    @property
    def nodes(self) -> list[str]:
        """Nodes in the graph.

        Returns:
            list[str]
        """
        return list(self.graph.nodes())

    @property
    def edges(self) -> list[tuple]:
        """Edges in the graph.

        Returns:
            list[tuple]
        """
        return list(self.graph.edges())

    @property
    def num_nodes(self) -> int:
        """Number of nodes in the graph

        Returns:
            int
        """
        return len(self.nodes)

    @property
    def num_edges(self) -> int:
        """Number of edges in the graph

        Returns:
            int
        """
        return len(self.edges)

    @property
    def sparsity(self) -> float:
        """Sparsity of the graph

        Returns:
            float: in [0,1]
        """
        s = self.num_nodes
        return self.num_edges / s / (s - 1) * 2

    @property
    def ground_truth(self) -> pd.DataFrame:
        """Returns the current ground truth as
        pandas adjacency.

        Returns:
            pd.DataFrame: Adjacenccy matrix.
        """
        return nx.to_pandas_adjacency(self.graph, weight=None)

    @property
    def causal_order(self) -> list[str]:
        """Returns the causal order of the current graph.
        Note that this order is in general not unique.

        Returns:
            list[str]: Causal order
        """
        return list(nx.topological_sort(self.graph))

    def parents(self, of_node: str) -> list[str]:
        """Return parents of node in question.

        Args:
            of_node (str): Node in question.

        Returns:
            list[str]: parent set.
        """
        return list(self.graph.predecessors(of_node))

    def save_drf(self, filename: str, location: str = None):
        """Writes a drf dict to file. Please provide the .pkl suffix!

        Args:
            filename (str): name of the file to be written e.g. examplefile.pkl
            location (str, optional): path to file in case it's not located in
                the current working directory. Defaults to None.
        """

        if not location:
            location = Path().resolve()

        location_path = Path(location, filename)

        with open(location_path, "wb") as f:
            pickle.dump(self.drf, f)

    def add_module(
        self,
        graph: nx.DiGraph,
        allow_in_edges: bool = True,
        mark_hidden: bool | list = False,
    ) -> str:
        """Adds module to cell graph. Module has to be as nx.DiGraph object

        Args:
            graph (nx.DiGraph): Graph to add to cell.
            allow_in_edges (bool, optional):
                whether nodes in the module are allowed to
                have in-edges. Defaults to True.
            mark_hidden (bool | list, optional):
                If False all nodes' 'is_hidden' attribute is set to False.
                If list of node names is provided these get overwritten to True.
                Defaults to False.

        Returns:
            str: prefix of Module created
        """

        next_module_prefix = self.next_module_prefix()

        node_renaming_dict = {
            old_node_name: f"{self.name}_{next_module_prefix}_{old_node_name}"
            for old_node_name in graph.nodes()
        }

        self.modules[self.next_module_prefix()] = graph.copy()
        graph = nx.relabel_nodes(graph, node_renaming_dict)

        if allow_in_edges:  # for later: mark nodes to not have incoming edges
            nx.set_node_attributes(graph, True, NodeAttributes.ALLOW_IN_EDGES)
        else:
            nx.set_node_attributes(graph, False, NodeAttributes.ALLOW_IN_EDGES)

        nx.set_node_attributes(
            graph, False, NodeAttributes.HIDDEN
        )  # set all non-hidden -> visible by default
        if isinstance(mark_hidden, list):
            mark_hidden_renamed = [
                f"{self.name}_{next_module_prefix}_{new_name}" for new_name in mark_hidden
            ]
            overwrite_dict = {node: {NodeAttributes.HIDDEN: True} for node in mark_hidden_renamed}
            nx.set_node_attributes(
                graph, values=overwrite_dict
            )  # only overwrite the ones specified
        # TODO relabel attributes, i.e. name of the parents has changed now?
        # .update_attributes or so or keep and remove prefixes in bayesian network creation?
        self.graph = nx.compose(self.graph, graph)

        return next_module_prefix

    def input_cellgraph_directly(self, graph: nx.DiGraph, allow_in_edges: bool = False):
        """Allow to input graphs on a cell-level. This should only be done if the graph
        is already available for the entire cell, otherwise `add_module()` is preferred.

        Args:
            graph (nx.DiGraph): Cell graph to input
            allow_in_edges (bool, optional): Defaults to False.
        """
        if allow_in_edges:  # for later: mark nodes to not have incoming edges
            nx.set_node_attributes(graph, True, NodeAttributes.ALLOW_IN_EDGES)
        else:
            nx.set_node_attributes(graph, False, NodeAttributes.ALLOW_IN_EDGES)

        node_renaming_dict = {
            old_node_name: f"{self.name}_{old_node_name}" for old_node_name in graph.nodes()
        }
        graph = nx.relabel_nodes(graph, node_renaming_dict)

        self.graph = nx.compose(self.graph, graph)

    def sample_from_drf(self, size=10, smoothed: bool = True) -> pd.DataFrame:
        """Draw from the trained DRF.

        Args:
            size (int, optional): Number of samples to be drawn. Defaults to 10.
            smoothed (bool, optional): If set to true, marginal distributions will
                be sampled from smoothed bootstraps. Defaults to True.

        Returns:
            pd.DataFrame: Data frame that follows the distribution implied by the ground truth.
        """
        return _sample_from_drf(prod_object=self, size=size, smoothed=smoothed)

    def _generate_random_dag(self, n_nodes: int = 5, p: float = 0.1) -> nx.DiGraph:
        """
        Creates a random DAG by
        taking an arbitrary ordering of the specified number of nodes,
        and then considers edges from node i to j only if i < j.
        That constraint leads to DAGness by construction.

        Args:
            n_nodes (int, optional): Defaults to 5.
            p (float, optional): Defaults to .1.

        Returns:
            nx.DiGraph:
        """
        dag = nx.DiGraph()
        dag.add_nodes_from(range(0, n_nodes))

        causal_order = list(dag.nodes)
        self.random_state.shuffle(causal_order)

        all_forward_edges = itertools.combinations(causal_order, 2)
        edges = np.array(list(all_forward_edges))

        random_choice = self.random_state.choice([False, True], p=[1 - p, p], size=edges.shape[0])

        dag.add_edges_from(edges[random_choice])
        return dag

    def add_random_module(self, n_nodes: int = 7, p: float = 0.10):
        randomdag = self._generate_random_dag(n_nodes=n_nodes, p=p)
        self.add_module(graph=randomdag, allow_in_edges=True, mark_hidden=False)

    def connect_by_module(self, m1: str, m2: str, edges: list[tuple]):
        """Connect two modules (by name, e.g. M2, M4) of the cell by a list
        of edges with the original node names.

        Args:
            m1: str
            m2: str
            edges: list[tuple]: use the original node names before they have entered into the cell,
                i.e. not with Cy_Mx prefix
        """
        self.__verify_edges_are_allowed(m1=m1, m2=m2, edges=edges)

        node_prefix_m1 = f"{self.name}_{m1}"
        node_prefix_m2 = f"{self.name}_{m2}"

        new_edges = [
            (f"{node_prefix_m1}_{edge[0]}", f"{node_prefix_m2}_{edge[1]}") for edge in edges
        ]

        [self.module_connectors.append(edge) for edge in new_edges]

        self.graph.add_edges_from(new_edges)

    def connect_by_random_edges(self, sparsity: float = 0.1) -> nx.DiGraph:
        """
        Add random edges to graph according to proportion,
        with restriction specified in node attributes.

        Args:
            sparsity (float, optional): Sparsity parameter in (0,1). Defaults to 0.1.

        Raises:
            NotImplementedError: when node attributes are not set.
            TypeError: when resulting graph contains cycles.

        Returns:
            nx.DiGraph: DAG with new edges added.
        """

        arrow_head_candidates = get_arrow_head_candidates_from_graph(
            graph=self.graph, node_attributes_to_filter=NodeAttributes.ALLOW_IN_EDGES
        )

        arrow_tail_candidates = [node for node in self.nodes if node not in arrow_head_candidates]

        potential_edges = tuples_from_cartesian_product(
            l1=arrow_tail_candidates, l2=arrow_head_candidates
        )
        num_choices = int(np.ceil(sparsity * len(potential_edges)))

        ### choose edges uniformly according to sparsity parameter
        chosen_edges = [
            potential_edges[i]
            for i in self.random_state.choice(
                a=len(potential_edges), size=num_choices, replace=False
            )
        ]

        self.graph.add_edges_from(chosen_edges)

        if not nx.is_directed_acyclic_graph(self.graph):
            raise TypeError(
                "The randomly chosen edges induce cycles, this is not supposed to happen."
            )
        return self.graph

    def __repr__(self):
        return f"ProcessCell(name={self.name})"

    def __str__(self):
        cell_description = {
            "Cell Name: ": self.name,
            "Description:": self.description if self.description else "n.a.",
            "Modules:": self.no_of_modules,
            "Nodes: ": self.num_nodes,
        }
        s = str()
        for info, info_text in cell_description.items():
            s += f"{info:<14}{info_text:>5}\n"

        return s

    def __verify_edges_are_allowed(self, m1: str, m2: str, edges: list[tuple]):
        """Check whether all starting point nodes
        (first value in edge tuple) are allowed.

        Args:
            m1 (str): Module1
            m2 (str): Module2
            edges (list[tuple]): Edges

        Raises:
            ValueError: starting node not in M1
            ValueError: ending node not in M2
        """
        source_nodes = set([e[0] for e in edges])
        target_nodes = set([e[1] for e in edges])
        m1_nodes = set(self.modules.get(m1).nodes())
        m2_nodes = set(self.modules.get(m2).nodes())

        if not source_nodes.issubset(m1_nodes):
            raise ValueError(f"source nodes: {source_nodes} not include in {m1}s nodes: {m1_nodes}")
        if not target_nodes.issubset(m2_nodes):
            raise ValueError(f"target nodes: {target_nodes} not include in {m2}s nodes: {m2_nodes}")

    def next_module_prefix(self) -> str:
        """Return the next module prefix, e.g.
        if there are already 3 modules connected to the cell,
        will return module_prefix4

        Returns:
            str: module_prefix
        """
        return f"{self.__module_prefix}{self.no_of_modules + 1}"

    @property
    def module_prefix(self) -> str:
        return self.__module_prefix

    @module_prefix.setter
    def module_prefix(self, module_prefix: str):
        if not isinstance(module_prefix, str):
            raise ValueError("please only use strings as module prefix")

        self.__module_prefix = module_prefix

    @property
    def no_of_modules(self) -> int:
        return len(self.modules)

    def get_nodes_by_attribute(self, attr_name: str, submodule: str = None) -> list:
        pass

    def get_available_attributes(self):
        available_attributes = set()
        for node_tuple in self.graph.nodes(data=True):
            for attribute_name in node_tuple[1].keys():
                available_attributes.add(attribute_name)

        return list(available_attributes)

    def to_cpdag(self) -> PDAG:
        return dag2cpdag(dag=self.graph)

    def show(
        self,
        meta_desc: str = "",
    ):
        """Plots the cell graph by giving extra weight to nodes
        with high in- and out-degree.

        Args:
            meta_desc (str, optional): Defaults to "".

        """
        cmap = plt.get_cmap("cividis")
        fig, ax = plt.subplots()
        center: np.ndarray = np.array([0, 0])
        pos = nx.spring_layout(
            self.graph,
            center=center,
            seed=10,
            k=50,
        )

        max_in_degree = max([d for _, d in self.graph.in_degree()])
        max_out_degree = max([d for _, d in self.graph.out_degree()])

        nx.draw_networkx_nodes(
            self.graph,
            pos=pos,
            ax=ax,
            cmap=cmap,
            vmin=-0.2,
            vmax=1,
            node_color=[
                (d + 10) / (max_in_degree + 10) for _, d in self.graph.in_degree(self.nodes)
            ],
            node_size=[
                500 * (d + 1) / (max_out_degree + 1) for _, d in self.graph.out_degree(self.nodes)
            ],
        )

        nx.draw_networkx_edges(
            self.graph,
            pos=pos,
            ax=ax,
            alpha=0.2,
            arrowsize=8,
            width=0.5,
            connectionstyle="arc3,rad=0.3",
        )

        ax.text(
            center[0],
            center[1] + 1.2,
            self.name + f"\n{meta_desc}",
            horizontalalignment="center",
            fontsize=12,
        )

        ax.axis("off")

    def _plot_cellgraph(
        self,
        ax,
        node_color,
        node_size,
        center=np.array([0, 0]),
        with_edges=True,
        with_box=True,
        meta_desc="",
    ):
        """Plots the cell graph by giving extra weight to nodes
        with high in- and out-degree.

        Args:
            with_edges (bool, optional): Defaults to True.
            with_box (bool, optional): Defaults to True.
            meta_desc (str, optional): Defaults to "".
            center (_type_, optional): Defaults to np.array([0, 0]).
            fig_size (tuple, optional): Defaults to (2, 8).
        """
        cmap = plt.get_cmap("cividis")

        pos = nx.spring_layout(
            self.graph,
            center=center,
            seed=10,
            k=50,
        )

        nx.draw_networkx_nodes(
            self.graph,
            pos=pos,
            ax=ax,
            cmap=cmap,
            vmin=-0.2,
            vmax=1,
            node_color=node_color,
            node_size=node_size,
        )

        if with_edges:
            nx.draw_networkx_edges(
                self.graph,
                pos=pos,
                ax=ax,
                alpha=0.2,
                arrowsize=8,
                width=0.5,
                connectionstyle="arc3,rad=0.3",
            )

        ax.text(
            center[0],
            center[1] + 1.2,
            self.name + f"\n{meta_desc}",
            horizontalalignment="center",
            fontsize=12,
        )

        if with_box:
            ax.add_collection(
                PatchCollection(
                    [
                        FancyBboxPatch(
                            center - [2, 1],
                            4,
                            2.6,
                            boxstyle=BoxStyle("Round", pad=0.02),
                        )
                    ],
                    alpha=0.2,
                    color="gray",
                )
            )

        ax.axis("off")
        return pos


def choose_edges_from_cells_randomly(
    from_cell: ProcessCell,
    to_cell: ProcessCell,
    probability: float,
    rng: np.random.Generator,
) -> list[tuple[str, str]]:
    """
    From two given cells (graphs), we take the cartesian product (end up with
    from_cell.number_of_nodes x to_cell.number_of_nodes possible edges (node tuples).

    From this product we draw probability x cartesian product number of edges randomly.

    In case we have a float number, we ceil the value,
    e.g. 17.3 edges will lead to 18 edges drawn.

    Args:
        from_cell: ProcessCell from where we want the edges
        to_cell: ProcessCell to where we want the edges
        probability: between 0 and 1

    Returns:
        list[tuple[str, str]]: Chosen edges.
    """

    assert 0 <= probability <= 1.0

    arrow_tail_candidates = list(from_cell.graph.nodes)
    arrow_head_candidates = get_arrow_head_candidates_from_graph(graph=to_cell.graph)

    potential_edges = tuples_from_cartesian_product(
        l1=arrow_tail_candidates, l2=arrow_head_candidates
    )

    num_to_choose = int(np.ceil(probability * len(potential_edges)))

    chosen_edges = [
        potential_edges[i]
        for i in rng.choice(a=len(potential_edges), size=num_to_choose, replace=False)
    ]

    return chosen_edges


def get_arrow_head_candidates_from_graph(
    graph: nx.DiGraph, node_attributes_to_filter: str = NodeAttributes.ALLOW_IN_EDGES
) -> list[str]:
    """Returns all arrow head (nodes where an arrow points to) nodes as list of candidates.
    To later build a list of tuples of potential edges.

    Args:
        graph (nx.DiGraph): DAG
        node_attributes_to_filter (str, optional): see NodeAttributes.
            Defaults to NodeAttributes.ALLOW_IN_EDGES.

    Returns:
        list[str]: list of nodes
    """
    arrow_head_candidates = [
        node
        for node, allowed in nx.get_node_attributes(graph, node_attributes_to_filter).items()
        if allowed is True
    ]

    nodes_without_attributes = list(
        set(graph.nodes).difference(
            set(nx.get_node_attributes(graph, node_attributes_to_filter).keys())
        )
    )

    if len(arrow_head_candidates) == 0 and len(nodes_without_attributes) == 0:
        logger.warning(
            f"None of the nodes in cell {graph} \
            are allowed to have in-edges."
        )

    arrow_head_candidates.extend(nodes_without_attributes)

    return arrow_head_candidates


class ProductionLineGraph:
    """Blueprint of a Production Line Graph.

    A Production Line consists of multiple Cells, each Cell can contain multiple modules.
    Modules can be instantiated randomly or manually. Cellgraphs and linegraphs can be
    instantiated directly from nx.DiGraph objects. Similarly, edges can be drawn at random
    (obeying certain probability choices that can be set by the user) between cells/moduls
    or manually.

    Besides populating a production line with cell/module-graphs one can obtain
    semi-synthetic data obeying the standard causal assumptions:

        1. Markov Property
        2. Faithfulness

    This can be achieved by fitting distributional random forests to the line/cell-graphs
    and draw data from these. A random number stream is initiated when calling this class.
    If desired this can be overwritten manually.

    """

    def __init__(self):
        self._random_state = np.random.default_rng(seed=2023)
        self.cells: dict[str, ProcessCell] = dict()
        self.cell_prefix = "C"
        self.cell_connectors: list[tuple] = list()
        self.cell_connector_edges = list()
        self.cell_order = list()
        self.drf: dict = dict()

    @property
    def random_state(self):
        return self._random_state

    @random_state.setter
    def random_state(self, r: np.random.Generator):
        if not isinstance(r, np.random.Generator):
            raise AssertionError("Specify numpy random number generator object!")
        self._random_state = r

    @property
    def graph(self) -> nx.DiGraph:
        """
        Returns a nx.DiGraph object of the actual graph.

        The graph is only built HERE, i.e. all ProcessCells exist standalone in self.cells,
        with no connections between their nodes yet.

        The edges are stored in self.cell_connetor_edges, where they are added by random methods
        or by user (the dag builder) himself.

        ATTENTION: you can not work on self.graph and add manually edges, nodes and expect them to
        work.

        Returns nx.DiGraph

        -------

        """
        g = nx.DiGraph()
        for cell in self.cells.values():
            g = nx.compose(g, cell.graph)

        g.add_edges_from(self.cell_connector_edges)

        if not nx.is_directed_acyclic_graph(g):
            raise TypeError(
                "There are cycles in the graph, \
                this is not supposed to happen."
            )

        return g

    @property
    def nodes(self) -> list[str]:
        """Nodes in the graph.

        Returns:
            list[str]
        """
        return list(self.graph.nodes())

    @property
    def edges(self) -> list[tuple]:
        """Edges in the graph.

        Returns:
            list[tuple]
        """
        return list(self.graph.edges())

    @property
    def num_nodes(self) -> int:
        """Number of nodes in the graph

        Returns:
            int
        """
        return len(self.nodes)

    @property
    def num_edges(self) -> int:
        """Number of edges in the graph

        Returns:
            int
        """
        return len(self.edges)

    @property
    def sparsity(self) -> float:
        """Sparsity of the graph

        Returns:
            float: in [0,1]
        """
        s = self.num_nodes
        return self.num_edges / s / (s - 1) * 2

    @property
    def ground_truth(self) -> pd.DataFrame:
        """Returns the current ground truth as
        pandas adjacency.

        Returns:
            pd.DataFrame: Adjacenccy matrix.
        """
        return nx.to_pandas_adjacency(self.graph, weight=None)

    def _get_union_graph(self) -> nx.DiGraph:
        if not self.cells:
            raise AssertionError("Your pline has no cells. Within has no meaning.")
        union_graph = nx.DiGraph()
        for _, station_graph in self.cells.items():
            union_graph = nx.union(union_graph, station_graph.graph)
        return union_graph

    @property
    def within_adjacency(self) -> pd.DataFrame:
        """Returns adjacency matrix ignoring all
        between-cell edges.

        Returns:
            pd.DataFrame: adjacency matrix
        """
        union_graph = self._get_union_graph()
        return nx.to_pandas_adjacency(union_graph)

    @property
    def between_adjacency(self) -> pd.DataFrame:
        """Returns adjacency matrix ignoring all
        within-cell edges.

        Returns:
            pd.DataFrame: adjacency matrix
        """
        union_graph = self._get_union_graph()
        return nx.to_pandas_adjacency(nx.difference(self.graph, union_graph))

    @property
    def causal_order(self) -> list[str]:
        """Returns the causal order of the current graph.
        Note that this order is in general not unique.

        Returns:
            list[str]: Causal order
        """
        return list(nx.topological_sort(self.graph))

    def parents(self, of_node: str) -> list[str]:
        """Return parents of node in question.

        Args:
            of_node (str): Node in question.

        Returns:
            list[str]: parent set.
        """
        return list(self.graph.predecessors(of_node))

    def to_cpdag(self) -> PDAG:
        return dag2cpdag(dag=self.graph)

    def get_nodes_of_station(self, station_name: str) -> list:
        """Returns nodes in chosen Station.

        Args:
            station_name (str): name of station.

        Raises:
            AssertionError: if station name doesn't match pline.

        Returns:
            list: nodes in chosen station
        """
        if station_name not in self.cell_order:
            raise AssertionError("Station name not among cells.")

        return self.cells[station_name].nodes

    def __add_cell(self, cell: ProcessCell) -> ProcessCell:
        cell_names = [cell_name for cell_name in self.cells.keys()]

        if cell.is_eol and any([cell.is_eol for cell in self.cells.values()]):
            raise AssertionError(
                f"Cell: {[cell for cell in self.cells.values() if cell.is_eol]} "
                f"is already EOL Cell in ProductionLineGraph."
            )

        if cell.name not in cell_names:
            self.cells[cell.name] = cell

            return cell

        raise ValueError(f"A cell with name: {cell.name} is already in the Production Line.")

    def new_cell(self, name: str = None, is_eol: bool = False) -> ProcessCell:
        """Add a new cell to the production line.

        If no name is given, cell name is given by counting available cells + 1

        Args:
            name (str, optional): Defaults to None.
            is_eol (bool, optional): Whether cell is end-of-line. Defaults to False.

        Returns:
            ProcessCell
        """
        if name:
            c = ProcessCell(name=name)

        else:
            actual_no_of_cells = len(self.cells.values())
            c = ProcessCell(name=f"{self.cell_prefix}{actual_no_of_cells}")

        c.random_state = self.random_state

        c.is_eol = is_eol
        self.__add_cell(cell=c)
        self.cell_order.append(c.name)
        return c

    def connect_cells(
        self,
        forward_probs: list[float] = [0.1, 0.05],
    ):
        """Randomly connects cells in a ProductionLineGraph according to a forwards logic.

        Args:
            forward_probs (list[float], optional): Array of sparsity scalars of
                dimension max_forward. Defaults to [0.1, 0.05].
        """
        # assume cells are initiated in order
        # otherwise allow to change order

        max_forward = len(forward_probs)
        cells = list(self.cells.values())
        no_of_cells = len(self.cells)

        for cell_idx, cell in enumerate(cells):
            prob_it = 0  # prob it(erator)

            for forwards in range(1, max_forward + 1):
                forward_cell_idx = cell_idx + forwards

                if forward_cell_idx < no_of_cells:
                    forward_cell = cells[forward_cell_idx]
                    chosen_edges = choose_edges_from_cells_randomly(
                        from_cell=cell,
                        to_cell=forward_cell,
                        probability=forward_probs[prob_it],
                        rng=self.random_state,
                    )

                    prob_it += 1  # FIXME: a bit ugly and hard to read
                    self.cell_connector_edges.extend(chosen_edges)

            if eol_cell := self.eol_cell:
                eol_cell_idx = cells.index(eol_cell)

                if cell_idx + max_forward < eol_cell_idx:
                    eol_prob = forward_probs[-1]
                    chosen_eol_edges = choose_edges_from_cells_randomly(
                        from_cell=cell,
                        to_cell=eol_cell,
                        probability=eol_prob,
                        rng=self.random_state,
                    )

                    self.cell_connector_edges.extend(chosen_eol_edges)

    def copy(self) -> ProductionLineGraph:
        """Makes a full copy of the current
        ProductionLineGraph object

        Returns:
            ProductionLineGraph: copyied object.
        """
        copy_graph = ProductionLineGraph()
        for station in self.cell_order:
            copy_graph.new_cell(station)
            # make sure its sorted
            sorted_graph = nx.DiGraph()
            sorted_graph.add_nodes_from(
                sorted(self.cells[station].nodes, key=lambda x: int(x.rpartition("_")[2]))
            )
            sorted_graph.add_edges_from(self.cells[station].edges)
            copy_graph.cells[station].graph = sorted_graph

        between_cell_edges = nx.difference(self.graph, copy_graph.graph).edges()
        copy_graph.connect_across_cells_manually(edges=between_cell_edges)
        return copy_graph

    def connect_across_cells_manually(self, edges: list[tuple]):
        """Add edges manually across cells. You need to give the full name
        Args:
            edges (list[tuple]): list of edges to add
        """
        self.cell_connector_edges.extend(edges)

    @classmethod
    def get_ground_truth(cls) -> ProductionLineGraph:
        """Loads in the ground_truth as described in the paper:
        causalAssembly: Generating Realistic Production Data for
        Benchmarking Causal Discovery
        Returns:
            ProductionLineGraph: ground_truth for cells and line.
        """

        gt_response = requests.get(DATA_GROUND_TRUTH, timeout=5)
        ground_truth = json.loads(gt_response.text)

        assembly_line = json_graph.adjacency_graph(ground_truth)

        stations = ["Station1", "Station2", "Station3", "Station4", "Station5"]
        ground_truth_line = ProductionLineGraph()

        for station in stations:
            ground_truth_line.new_cell(station)
            station_nodes = [node for node in assembly_line.nodes if node.startswith(station)]
            station_subgraph = nx.subgraph(assembly_line, station_nodes)
            # make sure its sorted
            sorted_graph = nx.DiGraph()
            sorted_graph.add_nodes_from(
                sorted(station_nodes, key=lambda x: int(x.rpartition("_")[2]))
            )
            sorted_graph.add_edges_from(station_subgraph.edges)
            ground_truth_line.cells[station].graph = sorted_graph

        between_cell_edges = nx.difference(assembly_line, ground_truth_line.graph).edges()
        ground_truth_line.connect_across_cells_manually(edges=between_cell_edges)
        return ground_truth_line

    @classmethod
    def get_data(cls) -> pd.DataFrame:
        """Load in semi-synthetic data as described in the paper:
        causalAssembly: Generating Realistic Production Data for
        Benchmarking Causal Discovery
        Returns:
            pd.DataFrame: Data from which data should be generated.
        """
        return pd.read_csv(DATA_DATASET)

    @classmethod
    def from_nx(cls, g: nx.DiGraph, cell_mapper: dict[str, list]):
        """Convert nx.DiGraph to ProductionLineGraph. Requires a dict mapping
        where keys are cell names and values correspond to nodes within these cells.

        Args:
            g (nx.DiGraph): graph to be converted
            cell_mapper (dict[str, list]): dict to indicate what nodes belong to which cell

        Returns:
            ProductionLineGraph (ProductionLineGraph): the graph as a ProductionLineGraph object.
        """
        if not isinstance(g, nx.DiGraph):
            raise ValueError("Graph must be of type nx.DiGraph")
        pline = ProductionLineGraph()
        if cell_mapper:
            for cellname, cols in cell_mapper.items():
                pline.new_cell(name=cellname)
                cell_graph = nx.induced_subgraph(g, cols)
                pline.cells[cellname].input_cellgraph_directly(cell_graph, allow_in_edges=True)
        relabel_dict = {}
        for cellname, cols in cell_mapper.items():
            for col in cols:
                relabel_dict[col] = cellname + "_" + col

        g_rename = nx.relabel_nodes(g, relabel_dict)
        between_cell_edges = nx.difference(g_rename, pline.graph).edges()
        pline.connect_across_cells_manually(edges=between_cell_edges)
        return pline

    @classmethod
    def load_drf(cls, filename: str, location: str = None):
        """Loads a drf dict from a .pkl file into the workspace.

        Args:
            filename (str): name of the file e.g. examplefile.pkl
            location (str, optional): path to file in case it's not located
                in the current working directory. Defaults to None.

        Returns:
            DRF (dict): dict of trained drf objects
        """
        if not location:
            location = Path().resolve()

        location_path = Path(location, filename)

        with open(location_path, "rb") as drf:
            pickle_drf = pickle.load(drf)

        return pickle_drf

    @classmethod
    def load_pline_from_pickle(cls, filename: str, location: str = None):
        if not location:
            location = Path().resolve()

        location_path = Path(location, filename)

        with open(location_path, "rb") as pline:
            pickle_line = pickle.load(pline)

        if not isinstance(pickle_line, ProductionLineGraph):
            raise TypeError("You didn't refer to a ProductionLineGraph.")

        return pickle_line

    def save_drf(self, filename: str, location: str = None):
        """Writes a drf dict to file. Please provide the .pkl suffix!

        Args:
            filename (str): name of the file to be written e.g. examplefile.pkl
            location (str, optional): path to file in case it's not located in
                the current working directory. Defaults to None.
        """

        if not location:
            location = Path().resolve()

        location_path = Path(location, filename)

        with open(location_path, "wb") as f:
            pickle.dump(self.drf, f)

    def sample_from_drf(self, size=10, smoothed: bool = True) -> pd.DataFrame:
        """Draw from the trained DRF.

        Args:
            size (int, optional): Number of samples to be drawn. Defaults to 10.
            smoothed (bool, optional): If set to true, marginal distributions will
                be sampled from smoothed bootstraps. Defaults to True.

        Returns:
            pd.DataFrame: Data frame that follows the distribution implied by the ground truth.
        """
        return _sample_from_drf(prod_object=self, size=size, smoothed=smoothed)

    def hidden_nodes(self) -> list:
        """Returns list of nodes marked as hidden

        Returns:
            list: of hidden nodes
        """
        return [
            node
            for node, hidden in nx.get_node_attributes(self.graph, NodeAttributes.HIDDEN).items()
            if hidden is True
        ]

    def visible_nodes(self):
        return [node for node in self.nodes if node not in self.hidden_nodes()]

    @property
    def eol_cell(self) -> ProcessCell | None:
        """

        Returns ProcessCell: the EOL cell
            (if any single cell has attr .is_eol = True), otherwise returns None
        -------

        """
        for cell in self.cells.values():
            if cell.is_eol:
                return cell

    @property
    def ground_truth_visible(self) -> pd.DataFrame:
        """Generates a ground truth graph in form of a pandas adjacency matrix.
        Row and column names correspond to visible.
        The following integers can occur:

        amat[i,j] = 1 indicates i -> j
        amat[i,j] = 0 indicates no edge
        amat[i,j] = amat[j,i] = 2 indicates i <-> j and there exists a common hidden confounder

        Returns:
            pd.DataFrame: amat with values in {0,1,2}.
        """

        if len(self.hidden_nodes()) == 0:
            return self.ground_truth
        else:
            mediators = self._pairs_with_hidden_mediators()
            confounders = self._pairs_with_hidden_confounders()

            # here 1 indicates that ROWS has edge to COLUMNS!
            amat = nx.to_pandas_adjacency(self.graph)
            amat_visible = amat.loc[self.visible_nodes(), self.visible_nodes()]

            for pairs in mediators:
                amat_visible.loc[pairs] = 1

            # reverse = lambda tuples: tuples[::-1]

            def reverse(tuples):
                """
                Simple function to reverse tuple order

                Args:
                    tuples (tuple): tuple to reverse order

                Returns:
                    tuple: tuple in reversed order
                """
                new_tup = tuples[::-1]
                return new_tup

            for pair, _ in confounders.items():
                amat_visible.loc[pair] = 2
                amat_visible.loc[reverse(pair)] = 2

            return amat_visible

    def show(self, meta_description: list | None = None, fig_size: tuple = (15, 8)):
        """Plot full assembly line

        Args:
            meta_description (list | None, optional): Specify additional cell info.
                Defaults to None.
            fig_size (tuple, optional): Adjust depending on number of cells.
                Defaults to (15, 8).

        Raises:
            AssertionError: Meta list entry needs to exist for each cell!
        """
        fig, ax = plt.subplots(figsize=fig_size)

        pos = {}

        if meta_description is None:
            meta_description = ["" for _ in range(len(self.cells))]

        if len(meta_description) != len(self.cells):
            raise AssertionError("Meta list entry needs to exist for each cell!")

        max_in_degree = max([d for _, d in self.graph.in_degree()])
        max_out_degree = max([d for _, d in self.graph.out_degree()])

        for i, (station_name, meta_desc) in enumerate(zip(self.cell_order, meta_description)):
            pos.update(
                self.cells[station_name]._plot_cellgraph(
                    ax=ax,
                    with_edges=False,
                    with_box=True,
                    meta_desc=meta_desc,
                    center=np.array([8 * i, 0]),
                    node_color=[
                        (d + 10) / (max_in_degree + 10)
                        for _, d in self.graph.in_degree(self.get_nodes_of_station(station_name))
                    ],
                    node_size=[
                        500 * (d + 1) / (max_out_degree + 1)
                        for _, d in self.graph.out_degree(self.get_nodes_of_station(station_name))
                    ],
                )
            )

        nx.draw_networkx_edges(
            self.graph,
            pos=pos,
            ax=ax,
            alpha=0.2,
            arrowsize=8,
            width=0.5,
            connectionstyle="arc3,rad=0.3",
        )

    def __str__(self):
        s = "ProductionLine\n\n"
        for cell in self.cells:
            s += f"{cell}\n"
        return s

    def __getattr__(self, attrname):
        if attrname not in self.cells.keys():
            raise AttributeError(f"{attrname} is not a valid attribute (cell name?)")
        return self.cells[attrname]

    # https://docs.python.org/3/library/pickle.html#pickle-protocol
    # TODO why is .cells enough, are the other member vars directly pickable?
    def __getstate__(self):
        return (self.__dict__, self.cells)

    def __setstate__(self, state):
        self.__dict__, self.cells = state

    @classmethod
    def via_cell_number(cls, n_cells: int, cell_prefix: str = "C"):
        """Inits a ProductionLineGraph with predefined number of cells, e.g. n_cells = 3

        Will create empty  C0, C1 and C2 as cells if no other cell_prefix is given.

        Args:
            n_cells (int): Number of cells the graph will have
            cell_prefix (str, optional): If you like other cell names pass them here.
                Defaults to "C".

        """
        pl = cls()
        pl.cell_prefix = cell_prefix

        [pl.new_cell() for _ in range(n_cells)]

        return pl

    def _pairs_with_hidden_mediators(self):
        """
        Return pairs of nodes with hidden mediators present.

        Args:
            graph (nx.DiGraph): DAG
            visible (list): list of visible nodes

        Returns:
            list: list of tuples with pairs of nodes with hidden mediator
        """
        any_paths = []
        visible = self.visible_nodes()
        hidden_all = self.hidden_nodes()
        confounders = [node.pop() for _, node in self._pairs_with_hidden_confounders().items()]
        hidden = [node for node in hidden_all if node not in confounders]
        for i, _ in enumerate(visible):
            for j, _ in enumerate(visible):
                for path in sorted(nx.all_simple_paths(self.graph, visible[i], visible[j])):
                    any_paths.append(path)

        pairs_with_hidden_mediators = [
            (ls[0], ls[-1]) for ls in any_paths if np.all(np.isin(ls[1:-1], hidden)) and len(ls) > 2
        ]

        return pairs_with_hidden_mediators

    def _pairs_with_hidden_confounders(self) -> dict:
        """
        Returns node-pairs that have a common hidden confounder

        Returns:
            dict: Dictionary with keys equal to tuples of node-pairs
            and values their corresponding hidden confounder(s)
        """
        confounder_pairs = {}
        visible = self.visible_nodes()
        pair_order_list = list(itertools.combinations(visible, 2))

        for node1, node2 in pair_order_list:
            ancestors1 = nx.ancestors(self.graph, node1)
            ancestors2 = nx.ancestors(self.graph, node2)
            if np.all(
                np.concatenate(
                    (
                        np.isin(list(ancestors1), visible, invert=True),
                        np.isin(list(ancestors2), visible, invert=True),
                    ),
                    axis=None,
                )
            ):  # annoying way of doing this. List comparison doesn't allow elementwise eval
                confounder = ancestors1.intersection(ancestors2)
                if confounder:  # only populate if set is non-empty
                    confounder_pairs[(node1, node2)] = confounder
            else:
                direct_parents1 = set(self.graph.predecessors(node1))
                direct_parents2 = set(self.graph.predecessors(node2))
                direct_confounder = [
                    node
                    for node in list(direct_parents1.intersection(direct_parents2))
                    if node not in visible
                ]
                if direct_confounder:
                    confounder_pairs[(node1, node2)] = direct_confounder

        return confounder_pairs
