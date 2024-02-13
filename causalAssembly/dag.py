"""DAG class"""
from __future__ import annotations

import logging
import pickle
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde

from causalAssembly.dag_utils import _bootstrap_sample
from causalAssembly.pdag import PDAG, dag2cpdag

logger = logging.getLogger(__name__)


class DAG:
    """
    General class for dealing with directed acyclic graph i.e.
    graphs that are directed and must not contain any cycles.
    """

    def __init__(
        self,
        nodes: list | None = None,
        edges: list[tuple] | None = None,
    ):
        if nodes is None:
            nodes = []
        if edges is None:
            edges = []

        self._nodes = set(nodes)
        self._edges = set()
        self._parents = defaultdict(set)
        self._children = defaultdict(set)
        self.drf: dict = dict()
        self._random_state: np.random.Generator = np.random.default_rng(seed=2023)

        for edge in edges:
            self._add_edge(*edge)

    def _add_edge(self, i, j):
        self._nodes.add(i)
        self._nodes.add(j)
        self._edges.add((i, j))

        # Check if graph is acyclic
        # TODO: Make check really after each edge is added?
        if not self.is_acyclic():
            raise ValueError(
                "The edge set you provided \
                induces one or more cycles.\
                Check your input!"
            )

        self._children[i].add(j)
        self._parents[j].add(i)

    @property
    def random_state(self):
        return self._random_state

    @random_state.setter
    def random_state(self, r: np.random.Generator):
        if not isinstance(r, np.random.Generator):
            raise AssertionError("Specify numpy random number generator object!")
        self._random_state = r

    def add_edge(self, edge: tuple[str, str]):
        """Add edge to DAG

        Args:
            edge (tuple[str, str]): Edge to add
        """
        self._add_edge(*edge)

    def add_edges_from(self, edges: list[tuple[str, str]]):
        """Add multiple edges to DAG

        Args:
            edges (list[tuple[str, str]]): Edges to add
        """
        for edge in edges:
            self.add_edge(edge=edge)

    def children(self, of_node: str) -> list[str]:
        """Gives all children of node `node`.

        Args:
            node (str): node in current DAG.

        Returns:
            list: of children.
        """
        if of_node in self._children.keys():
            return list(self._children[of_node])
        else:
            return []

    def parents(self, of_node: str) -> list[str]:
        """Gives all parents of node `node`.

        Args:
            node (str): node in current DAG.

        Returns:
            list: of parents.
        """
        if of_node in self._parents.keys():
            return list(self._parents[of_node])
        else:
            return []

    def induced_subgraph(self, nodes: list[str]) -> DAG:
        """Returns the induced subgraph on the nodes in `nodes`.

        Args:
            nodes (list[str]): List of nodes.

        Returns:
            DAG: Induced subgraph.
        """
        edges = [(i, j) for i, j in self.edges if i in nodes and j in nodes]
        return DAG(nodes=nodes, edges=edges)

    def is_adjacent(self, i: str, j: str) -> bool:
        """Return True if the graph contains an directed
        edge between i and j.

        Args:
            i (str): node i.
            j (str): node j.

        Returns:
            bool: True if i->j or i<-j
        """
        return (j, i) in self.edges or (i, j) in self.edges

    def is_clique(self, potential_clique: set) -> bool:
        """
        Check every pair of node X potential_clique is adjacent.
        """
        return all(self.is_adjacent(i, j) for i, j in combinations(potential_clique, 2))

    def is_acyclic(self) -> bool:
        """Check if the graph is acyclic.

        Returns:
            bool: True if graph is acyclic.
        """
        nx_dag = self.to_networkx()
        return nx.is_directed_acyclic_graph(nx_dag)

    @classmethod
    def from_pandas_adjacency(cls, pd_amat: pd.DataFrame) -> DAG:
        """Build DAG from a Pandas adjacency matrix.

        Args:
            pd_amat (pd.DataFrame): input adjacency matrix.

        Returns:
            DAG
        """
        assert pd_amat.shape[0] == pd_amat.shape[1]
        nodes = pd_amat.columns

        all_connections = []
        start, end = np.where(pd_amat != 0)
        for idx, _ in enumerate(start):
            all_connections.append((pd_amat.columns[start[idx]], pd_amat.columns[end[idx]]))

        temp = [set(i) for i in all_connections]
        temp2 = [arc for arc in all_connections if temp.count(set(arc)) > 1]

        dir_edges = [edge for edge in all_connections if edge not in temp2]

        return DAG(nodes=nodes, edges=dir_edges)

    def remove_edge(self, i: str, j: str):
        """Removes edge in question

        Args:
            i (str): tail
            j (str): head

        Raises:
            AssertionError: if edge does not exist
        """
        if (i, j) not in self.edges:
            raise AssertionError("Edge does not exist in current DAG")

        self._edges.discard((i, j))
        self._children[i].discard(j)
        self._parents[j].discard(i)

    def remove_node(self, node):
        """Remove a node from the graph"""
        self._nodes.remove(node)

        self._edges = {(i, j) for i, j in self._edges if i != node and j != node}

        for child in self._children[node]:
            self._parents[child].remove(node)

        for parent in self._parents[node]:
            self._children[parent].remove(node)

        self._parents.pop(node, "I was never here")
        self._children.pop(node, "I was never here")

    @property
    def adjacency_matrix(self) -> pd.DataFrame:
        """Returns adjacency matrix where the i,jth
        entry being one indicates that there is an edge
        from i to j. A zero indicates that there is no edge.

        Returns:
            pd.DataFrame: adjacency matrix
        """
        amat = pd.DataFrame(
            np.zeros([self.num_nodes, self.num_nodes]),
            index=self.nodes,
            columns=self.nodes,
        )
        for edge in self.edges:
            amat.loc[edge] = 1
        return amat

    def vstructs(self) -> set:
        """Retrieve v-structures

        Returns:
            set: set of all v-structures
        """
        vstructures = set()
        for node in self._nodes:
            for p1, p2 in combinations(self._parents[node], 2):
                if p1 not in self._parents[p2] and p2 not in self._parents[p1]:
                    vstructures.add((p1, node))
                    vstructures.add((p2, node))
        return vstructures

    def copy(self):
        """Return a copy of the graph"""
        return DAG(nodes=self._nodes, edges=self._edges)

    def show(self):
        """Plot DAG."""
        graph = self.to_networkx()
        pos = nx.circular_layout(graph)
        nx.draw(graph, pos=pos, with_labels=True)

    def to_networkx(self) -> nx.DiGraph:
        """Convert to networkx graph.

        Returns:
            nx.MultiDiGraph: Graph with directed and undirected edges.
        """
        nx_dag = nx.DiGraph()
        nx_dag.add_nodes_from(self.nodes)
        nx_dag.add_edges_from(self.edges)

        return nx_dag

    @property
    def nodes(self) -> list:
        """Get all nods in current DAG.

        Returns:
            list: list of nodes.
        """
        return sorted(list(self._nodes))

    @property
    def num_nodes(self) -> int:
        """Number of nodes in current DAG.

        Returns:
            int: Number of nodes
        """
        return len(self._nodes)

    @property
    def num_edges(self) -> int:
        """Number of directed edges
        in current DAG.

        Returns:
            int: Number of directed edges
        """
        return len(self._edges)

    @property
    def sparsity(self) -> float:
        """Sparsity of the graph

        Returns:
            float: in [0,1]
        """
        s = self.num_nodes
        return self.num_edges / s / (s - 1) * 2

    @property
    def edges(self) -> list[tuple]:
        """Gives all directed edges in
        current DAG.

        Returns:
            list[tuple]: List of directed edges.
        """
        return list(self._edges)

    @property
    def causal_order(self) -> list[str]:
        """Returns the causal order of the current graph.
        Note that this order is in general not unique.

        Returns:
            list[str]: Causal order
        """
        return list(nx.lexicographical_topological_sort(self.to_networkx()))

    @property
    def max_in_degree(self) -> int:
        """Maximum in-degree of the graph.

        Returns:
            int: Maximum in-degree
        """
        return max(len(self._parents[node]) for node in self._nodes)

    @property
    def max_out_degree(self) -> int:
        """Maximum out-degree of the graph.

        Returns:
            int: Maximum out-degree
        """
        return max(len(self._children[node]) for node in self._nodes)

    @classmethod
    def from_nx(cls, nx_dag: nx.DiGraph) -> DAG:
        """Convert to DAG from nx.DiGraph.

        Args:
            nx_dag (nx.DiGraph): DAG in question.

        Raises:
            TypeError: If DAG is not nx.DiGraph

        Returns:
            DAG
        """
        if not isinstance(nx_dag, nx.DiGraph):
            raise TypeError("DAG must be of type nx.DiGraph")
        return DAG(nodes=list(nx_dag.nodes), edges=list(nx_dag.edges))

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
        return _sample_from_drf(graph=self, size=size, smoothed=smoothed)

    def to_cpdag(self) -> PDAG:
        return dag2cpdag(dag=self.to_networkx())

    @classmethod
    def load_drf(cls, filename: str, location: str = None) -> dict:
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


def _sample_from_drf(graph: DAG, size=10, smoothed: bool = True) -> pd.DataFrame:
    if not graph.drf:
        raise ValueError("Nothing to sample from. Learn DRF first!")
    sample_dict = {}
    for node in graph.causal_order:
        if isinstance(graph.drf[node], gaussian_kde):
            # Node has no parents, generate a sample using bootstrapping
            #
            if smoothed:
                sample_dict[node] = graph.drf[node].resample(size=size, seed=graph.random_state)[0]
            else:
                sample_dict[node] = _bootstrap_sample(
                    rng=graph.random_state,
                    data=graph.drf[node].dataset[0],
                    size=size,
                )
        else:
            parents = graph.parents(of_node=node)
            new_data = pd.DataFrame({col: sample_dict[col] for col in parents})
            # new_data = pd.DataFrame(sample_dict[parents])
            forest = graph.drf[node]
            sample_dict[node] = forest.produce_sample(
                newdata=new_data, random_state=graph.random_state
            )
    new_df = pd.DataFrame(sample_dict)
    return new_df[graph.nodes]
