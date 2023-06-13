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
import logging
from collections import defaultdict
from itertools import combinations

import networkx as nx
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class PDAG:
    """
    Class for dealing with partially directed graph i.e.
    graphs that contain both directed and undirected edges.
    """

    def __init__(
        self,
        nodes: list = [],
        dir_edges: list[tuple] = [],
        undir_edges: list[tuple] = [],
    ):
        self._nodes = set(nodes)
        self._undir_edges = set()
        self._dir_edges = set()
        self._parents = defaultdict(set)
        self._children = defaultdict(set)
        self._neighbors = defaultdict(set)
        self._undirected_neighbors = defaultdict(set)

        for dir_edge in dir_edges:
            self._add_dir_edge(*dir_edge)
        for unir_edge in undir_edges:
            self._add_undir_edge(*unir_edge)

    def _add_dir_edge(self, i, j):
        self._nodes.add(i)
        self._nodes.add(j)
        self._dir_edges.add((i, j))

        self._neighbors[i].add(j)
        self._neighbors[j].add(i)

        self._children[i].add(j)
        self._parents[j].add(i)

    def _add_undir_edge(self, i, j):
        self._nodes.add(i)
        self._nodes.add(j)
        self._undir_edges.add((i, j))

        self._neighbors[i].add(j)
        self._neighbors[j].add(i)

        self._undirected_neighbors[i].add(j)
        self._undirected_neighbors[j].add(i)

    def children(self, node: str) -> set:
        if node in self._children.keys():
            return self._children[node]
        else:
            return set()

    def parents(self, node: str) -> set:
        if node in self._parents.keys():
            return self._parents[node]
        else:
            return set()

    def neighbors(self, node: str) -> set:
        if node in self._neighbors.keys():
            return self._neighbors[node]
        else:
            return set()

    def undir_neighbors(self, node: str) -> set:
        if node in self._undirected_neighbors.keys():
            return self._undirected_neighbors[node]
        else:
            return set()

    def is_adjacent(self, i, j):
        """Return True if the graph contains an directed
        or undirected edge between i and j"""
        return any(
            (
                (j, i) in self.dir_edges or (j, i) in self.undir_edges,
                (i, j) in self.dir_edges or (i, j) in self.undir_edges,
            )
        )

    def is_clique(self, potential_clique: set) -> bool:
        """
        Check every pair of node X potential_clique is adjacent.
        """
        return all(self.is_adjacent(i, j) for i, j in combinations(potential_clique, 2))

    @classmethod
    def from_pandas(cls, pd_amat: pd.DataFrame):
        assert pd_amat.shape[0] == pd_amat.shape[1]
        nodes = pd_amat.columns

        all_connections = []
        start, end = np.where(pd_amat != 0)
        for idx, _ in enumerate(start):
            all_connections.append(
                (pd_amat.columns[start[idx]], pd_amat.columns[end[idx]])
            )

        temp = [set(i) for i in all_connections]
        temp2 = [arc for arc in all_connections if temp.count(set(arc)) > 1]
        undir_edges = [tuple(item) for item in set(frozenset(item) for item in temp2)]

        dir_edges = [edge for edge in all_connections if edge not in temp2]

        return PDAG(nodes=nodes, dir_edges=dir_edges, undir_edges=undir_edges)

    def remove_node(self, node):
        """Remove a node from the graph"""
        self._nodes.remove(node)

        self._dir_edges = {
            (i, j) for i, j in self._dir_edges if i != node and j != node
        }

        self._undir_edges = {
            (i, j) for i, j in self._undir_edges if i != node and j != node
        }

        for child in self._children[node]:
            self._parents[child].remove(node)
            self._neighbors[child].remove(node)

        for parent in self._parents[node]:
            self._children[parent].remove(node)
            self._neighbors[parent].remove(node)

        for u_nbr in self._undirected_neighbors[node]:
            self._undirected_neighbors[u_nbr].remove(node)
            self._neighbors[u_nbr].remove(node)

        self._parents.pop(node, "I was never here")
        self._children.pop(node, "I was never here")
        self._neighbors.pop(node, "I was never here")
        self._undirected_neighbors.pop(node, "I was never here")

    def to_dag(self) -> nx.DiGraph:
        """
        Algorithm as described in Chickering (2002):

            1. From PDAG P create DAG G containing all directed edges from P
            2. Repeat the following: Select node v in P s.t.
                i. v has no outgoing edges (children) i.e. \\(ch(v) = \\emptyset \\)

                ii. \\(neigh(v) \\neq \\emptyset\\)
                    Then \\( (pa(v) \\cup (neigh(v) \\) form a clique.
                    For each v that is in a clique and is part of an undirected edge in P
                    i.e. w - v, insert a directed edge w -> v in G.
                    Remove v and all incident edges from P and continue with next node.
                    Until all nodes have been deleted from P.

        Returns:
            nx.DiGraph: DAG that belongs to the MEC implied by the PDAG
        """

        pdag = self.copy()

        dag = nx.DiGraph()
        dag.add_nodes_from(pdag.nodes)
        dag.add_edges_from(pdag.dir_edges)

        if pdag.num_undir_edges == 0:
            return dag
        else:
            while pdag.nnodes > 0:
                # find node with (1) no directed outgoing edges and
                #                (2) the set of undirected neighbors is either empty or
                #                    undirected neighbors + parents of X are a clique
                found = False
                for node in pdag.nodes:
                    children = pdag.children(node)
                    neighbors = pdag.neighbors(node)
                    pdag._undirected_neighbors[node]
                    parents = pdag.parents(node)
                    potential_clique_members = neighbors.union(parents)

                    is_clique = pdag.is_clique(potential_clique_members)

                    if not len(children) and (not len(neighbors) or is_clique):
                        found = True
                        # add all edges of node as outgoing edges to dag
                        for edge in pdag.undir_edges:
                            if node in edge:
                                incident_node = set(edge) - {node}
                                dag.add_edge(*incident_node, node)

                        pdag.remove_node(node)
                        break

                if not found:
                    logger.warning("PDAG not extendible: Random DAG on skeleton drawn.")

                    dag = nx.from_pandas_adjacency(
                        self._amat_to_dag(), create_using=nx.DiGraph
                    )

                    break

            return dag

    @property
    def adjacency_matrix(self) -> pd.DataFrame:
        amat = pd.DataFrame(
            np.zeros([self.nnodes, self.nnodes]),
            index=self.nodes,
            columns=self.nodes,
        )
        for edge in self.dir_edges:
            amat.loc[edge[0], edge[1]] = 1
        for edge in self.undir_edges:
            amat.loc[edge[0], edge[1]] = amat.loc[edge[1], edge[0]] = 1
        return amat

    def _amat_to_dag(self) -> pd.DataFrame:
        """Transform the adjacency matrix of an PDAG to the adjacency
        matrix of a SOME DAG in the Markov equivalence class.

        Returns:
            pd.DataFrame: DAG, a member of the MEC.
        """
        pdag_amat = self.adjacency_matrix.to_numpy()

        p = pdag_amat.shape[0]
        skel = pdag_amat + pdag_amat.T
        skel[np.where(skel > 1)] = 1
        permute_ord = np.random.choice(a=p, size=p, replace=False)
        skel = skel[:, permute_ord][permute_ord]

        for i in range(1, p):
            for j in range(0, i + 1):
                if skel[i, j] == 1:
                    skel[i, j] = 0

        i_ord = np.sort(permute_ord)
        skel = skel[:, i_ord][i_ord]
        return pd.DataFrame(
            skel,
            index=self.adjacency_matrix.index,
            columns=self.adjacency_matrix.columns,
        )

    def vstructs(self):
        vstructs = set()
        for node in self._nodes:
            for p1, p2 in combinations(self._parents[node], 2):
                if p1 not in self._parents[p2] and p2 not in self._parents[p1]:
                    vstructs.add((p1, node))
                    vstructs.add((p2, node))
        return vstructs

    def copy(self):
        """Return a copy of the graph"""
        return PDAG(
            nodes=self._nodes, dir_edges=self._dir_edges, undir_edges=self._undir_edges
        )

    def show(self):
        """Plot PDAG."""
        graph = self.to_networkx()
        pos = nx.circular_layout(graph)
        nx.draw(graph, pos=pos, with_labels=True)

    def to_networkx(self) -> nx.MultiDiGraph:
        """Convert to networkx graph.

        Returns:
            nx.MultiDiGraph: Graph with directed and undirected edges.
        """
        nx_pdag = nx.MultiDiGraph(self.dir_edges)
        for edge in self.undir_edges:
            nx_pdag.add_edge(*edge)
            nx_pdag.add_edge(*edge[::-1])

        return nx_pdag

    @property
    def nodes(self):
        return sorted(list(self._nodes))

    @property
    def nnodes(self):
        return len(self._nodes)

    @property
    def num_undir_edges(self):
        return len(self._undir_edges)

    @property
    def num_dir_edges(self):
        return len(self._dir_edges)

    @property
    def num_adjacencies(self):
        return self.num_undir_edges + self.num_edges

    @property
    def undir_edges(self):
        return list(self._undir_edges)

    @property
    def dir_edges(self):
        return list(self._dir_edges)
