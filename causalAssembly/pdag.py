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
        nodes: list | None = None,
        dir_edges: list[tuple] | None = None,
        undir_edges: list[tuple] | None = None,
    ):
        if nodes is None:
            nodes = []
        if dir_edges is None:
            dir_edges = []
        if undir_edges is None:
            undir_edges = []

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
        """Gives all children of node `node`.

        Args:
            node (str): node in current PDAG.

        Returns:
            set: set of children.
        """
        if node in self._children.keys():
            return self._children[node]
        else:
            return set()

    def parents(self, node: str) -> set:
        """Gives all parents of node `node`.

        Args:
            node (str): node in current PDAG.

        Returns:
            set: set of parents.
        """
        if node in self._parents.keys():
            return self._parents[node]
        else:
            return set()

    def neighbors(self, node: str) -> set:
        """Gives all neighbors of node `node`.

        Args:
            node (str): node in current PDAG.

        Returns:
            set: set of neighbors.
        """
        if node in self._neighbors.keys():
            return self._neighbors[node]
        else:
            return set()

    def undir_neighbors(self, node: str) -> set:
        """Gives all undirected neighbors
        of node `node`.

        Args:
            node (str): node in current PDAG.

        Returns:
            set: set of undirected neighbors.
        """
        if node in self._undirected_neighbors.keys():
            return self._undirected_neighbors[node]
        else:
            return set()

    def is_adjacent(self, i: str, j: str) -> bool:
        """Return True if the graph contains an directed
        or undirected edge between i and j.

        Args:
            i (str): node i.
            j (str): node j.

        Returns:
            bool: True if i-j or i->j or i<-j
        """
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
    def from_pandas_adjacency(cls, pd_amat: pd.DataFrame) -> PDAG:
        """Build PDAG from a Pandas adjacency matrix.

        Args:
            pd_amat (pd.DataFrame): input adjacency matrix.

        Returns:
            PDAG
        """
        assert pd_amat.shape[0] == pd_amat.shape[1]
        nodes = pd_amat.columns

        all_connections = []
        start, end = np.where(pd_amat != 0)
        for idx, _ in enumerate(start):
            all_connections.append((pd_amat.columns[start[idx]], pd_amat.columns[end[idx]]))

        temp = [set(i) for i in all_connections]
        temp2 = [arc for arc in all_connections if temp.count(set(arc)) > 1]
        undir_edges = [tuple(item) for item in set(frozenset(item) for item in temp2)]

        dir_edges = [edge for edge in all_connections if edge not in temp2]

        return PDAG(nodes=nodes, dir_edges=dir_edges, undir_edges=undir_edges)

    def remove_edge(self, i: str, j: str):
        """Removes edge in question

        Args:
            i (str): tail
            j (str): head

        Raises:
            AssertionError: if edge does not exist
        """
        if (i, j) not in self.dir_edges and (i, j) not in self.undir_edges:
            raise AssertionError("Edge does not exist in current PDAG")

        self._undir_edges.discard((i, j))
        self._dir_edges.discard((i, j))
        self._children[i].discard(j)
        self._parents[j].discard(i)
        self._neighbors[i].discard(j)
        self._neighbors[j].discard(i)
        self._undirected_neighbors[i].discard(j)
        self._undirected_neighbors[j].discard(i)

    def undir_to_dir_edge(self, tail: str, head: str):
        """Takes a undirected edge and turns it into a directed one.
        tail indicates the starting node of the edge and head the end node, i.e.
        tail -> head.

        Args:
            tail (str): starting node
            head (str): end node

        Raises:
            AssertionError: if edge does not exist or is not undirected.
        """
        if (tail, head) not in self.undir_edges and (
            head,
            tail,
        ) not in self.undir_edges:
            raise AssertionError("Edge seems not to be undirected or even there at all.")
        self._undir_edges.discard((tail, head))
        self._undir_edges.discard((head, tail))
        self._neighbors[tail].discard(head)
        self._neighbors[head].discard(tail)
        self._undirected_neighbors[tail].discard(head)
        self._undirected_neighbors[head].discard(tail)

        self._add_dir_edge(i=tail, j=head)

    def remove_node(self, node):
        """Remove a node from the graph"""
        self._nodes.remove(node)

        self._dir_edges = {(i, j) for i, j in self._dir_edges if i != node and j != node}

        self._undir_edges = {(i, j) for i, j in self._undir_edges if i != node and j != node}

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
                    # pdag._undirected_neighbors[node]
                    parents = pdag.parents(node)
                    potential_clique_members = neighbors.union(parents)

                    is_clique = pdag.is_clique(potential_clique_members)

                    if not children and (not neighbors or is_clique):
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

                    dag = nx.from_pandas_adjacency(self._amat_to_dag(), create_using=nx.DiGraph)

                    break

            return dag

    @property
    def adjacency_matrix(self) -> pd.DataFrame:
        """Returns adjacency matrix where the i,jth
        entry being one indicates that there is an edge
        from i to j. A zero indicates that there is no edge.

        Returns:
            pd.DataFrame: adjacency matrix
        """
        amat = pd.DataFrame(
            np.zeros([self.nnodes, self.nnodes]),
            index=self.nodes,
            columns=self.nodes,
        )
        for edge in self.dir_edges:
            amat.loc[edge] = 1
        for edge in self.undir_edges:
            amat.loc[edge] = amat.loc[edge[::-1]] = 1
        return amat

    def _amat_to_dag(self) -> pd.DataFrame:
        """Transform the adjacency matrix of an PDAG to the adjacency
        matrix of a SOME DAG in the Markov equivalence class.

        Returns:
            pd.DataFrame: DAG, a member of the MEC.
        """
        pdag_amat = self.adjacency_matrix.to_numpy()

        p = pdag_amat.shape[0]
        ## amat to skel
        skel = pdag_amat + pdag_amat.T
        skel[np.where(skel > 1)] = 1
        ## permute skel
        permute_ord = np.random.choice(a=p, size=p, replace=False)
        skel = skel[:, permute_ord][permute_ord]

        ## skel to dag
        for i in range(1, p):
            for j in range(0, i + 1):
                if skel[i, j] == 1:
                    skel[i, j] = 0

        ## inverse permutation
        i_ord = np.sort(permute_ord)
        skel = skel[:, i_ord][i_ord]
        return pd.DataFrame(
            skel,
            index=self.adjacency_matrix.index,
            columns=self.adjacency_matrix.columns,
        )

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
        return PDAG(nodes=self._nodes, dir_edges=self._dir_edges, undir_edges=self._undir_edges)

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
        nx_pdag = nx.MultiDiGraph()
        nx_pdag.add_nodes_from(self.nodes)
        nx_pdag.add_edges_from(self.dir_edges)
        for edge in self.undir_edges:
            nx_pdag.add_edge(*edge)
            nx_pdag.add_edge(*edge[::-1])

        return nx_pdag

    def _meek_mec_enumeration(self, pdag: PDAG, dag_list: list):
        """Recursion algorithm which recursively applies the
        following steps:
            1. Orient the first undirected edge found.
            2. Apply Meek rules.
            3. Recurse with each direction of the oriented edge.
        This corresponds to Algorithm 2 in Wienöbst et al. (2023).

        Args:
            pdag (PDAG): partially directed graph in question.
            dag_list (list): list of currently found DAGs.

        References:
            Wienöbst, Marcel, et al. "Efficient enumeration of Markov equivalent DAGs."
            Proceedings of the AAAI Conference on Artificial Intelligence.
            Vol. 37. No. 10. 2023.
        """
        g_copy = pdag.copy()
        g_copy = self._apply_meek_rules(g_copy)  # Apply Meek rules

        undir_edges = g_copy.undir_edges
        if undir_edges:
            i, j = undir_edges[0]  # Take first undirected edge

        if not g_copy.undir_edges:
            # makes sure that flaoting nodes are preserved
            new_member = nx.DiGraph()
            new_member.add_nodes_from(g_copy.nodes)
            new_member.add_edges_from(g_copy.dir_edges)
            dag_list.append(new_member)
            return  # Add DAG to current list

        # Recursion first orientation:
        g_copy.undir_to_dir_edge(i, j)
        self._meek_mec_enumeration(pdag=g_copy, dag_list=dag_list)
        g_copy.remove_edge(i, j)

        # Recursion second orientation
        g_copy._add_dir_edge(j, i)
        self._meek_mec_enumeration(pdag=g_copy, dag_list=dag_list)

    def to_allDAGs(self) -> list[nx.DiGraph]:
        """Recursion algorithm which recursively applies the
        following steps:
            1. Orient the first undirected edge found.
            2. Apply Meek rules.
            3. Recurse with each direction of the oriented edge.
        This corresponds to Algorithm 2 in Wienöbst et al. (2023).

        References:
            Wienöbst, Marcel, et al. "Efficient enumeration of Markov equivalent DAGs."
            Proceedings of the AAAI Conference on Artificial Intelligence.
            Vol. 37. No. 10. 2023.
        """
        all_dags = []
        self._meek_mec_enumeration(pdag=self, dag_list=all_dags)
        return all_dags

    # use Meek's cpdag2alldag
    def _apply_meek_rules(self, G: PDAG) -> PDAG:
        """Apply all four Meek rules to a
        PDAG turning it into a CPDAG.

        Args:
            G (PDAG): PDAG to complete

        Returns:
            PDAG: completed PDAG.
        """
        # Apply Meek Rules
        cpdag = G.copy()
        cpdag = rule_1(pdag=cpdag)
        cpdag = rule_2(pdag=cpdag)
        cpdag = rule_3(pdag=cpdag)
        cpdag = rule_4(pdag=cpdag)
        return cpdag

    def to_random_dag(self) -> nx.DiGraph:
        """Provides a random DAG residing in the MEC.

        Returns:
            nx.DiGraph: random DAG living in MEC
        """
        to_dag_candidate = self.copy()

        while to_dag_candidate.num_undir_edges > 0:
            chosen_edge = to_dag_candidate.undir_edges[
                np.random.choice(to_dag_candidate.num_undir_edges)
            ]
            choose_orientation = [chosen_edge, chosen_edge[::-1]]
            node_i, node_j = choose_orientation[np.random.choice(len(choose_orientation))]

            to_dag_candidate.undir_to_dir_edge(tail=node_i, head=node_j)
            to_dag_candidate = to_dag_candidate._apply_meek_rules(G=to_dag_candidate)

        return nx.from_pandas_adjacency(to_dag_candidate.adjacency_matrix, create_using=nx.DiGraph)

    @property
    def nodes(self) -> list:
        """Get all nods in current PDAG.

        Returns:
            list: list of nodes.
        """
        return sorted(list(self._nodes))

    @property
    def nnodes(self) -> int:
        """Number of nodes in current PDAG.

        Returns:
            int: Number of nodes
        """
        return len(self._nodes)

    @property
    def num_undir_edges(self) -> int:
        """Number of undirected edges
        in current PDAG.

        Returns:
            int: Number of undirected edges
        """
        return len(self._undir_edges)

    @property
    def num_dir_edges(self) -> int:
        """Number of directed edges
        in current PDAG.

        Returns:
            int: Number of directed edges
        """
        return len(self._dir_edges)

    @property
    def num_adjacencies(self) -> int:
        """Number of adjacent nodes
        in current PDAG.

        Returns:
            int: Number of adjacent nodes
        """
        return self.num_undir_edges + self.num_dir_edges

    @property
    def undir_edges(self) -> list[tuple]:
        """Gives all undirected edges in
        current PDAG.

        Returns:
            list[tuple]: List of undirected edges.
        """
        return list(self._undir_edges)

    @property
    def dir_edges(self) -> list[tuple]:
        """Gives all directed edges in
        current PDAG.

        Returns:
            list[tuple]: List of directed edges.
        """
        return list(self._dir_edges)


def vstructs(dag: nx.DiGraph) -> set:
    """Retrieve all v-structures in a DAG.

    Args:
        dag (nx.DiGraph): DAG in question

    Returns:
        set: Set of all v-structures.
    """
    vstructures = set()
    for node in dag.nodes():
        for p1, p2 in combinations(list(dag.predecessors(node)), 2):  # get all parents of node
            if not dag.has_edge(p1, p2) and not dag.has_edge(p2, p1):
                vstructures.add((p1, node))
                vstructures.add((p2, node))
    return vstructures


def rule_1(pdag: PDAG) -> PDAG:
    """Given the following pattern X -> Y - Z. Orient Y - Z to Y -> Z
    if X and Z are non-adjacent (otherwise a new v-structure arises).

    Args:
        pdag (PDAG): PDAG before application of rule.

    Returns:
        PDAG: PDAG after application of rule.
    """
    copy_pdag = pdag.copy()
    for edge in copy_pdag.undir_edges:
        reverse_edge = edge[::-1]
        test_edges = [edge, reverse_edge]
        for tail, head in test_edges:
            orient = False
            undir_parents = copy_pdag.parents(tail)
            if undir_parents:
                for parent in undir_parents:
                    if not copy_pdag.is_adjacent(parent, head):
                        orient = True
            if orient:
                copy_pdag.undir_to_dir_edge(tail=tail, head=head)
                break
    return copy_pdag


def rule_2(pdag: PDAG) -> PDAG:
    """Given the following directed triple
    X -> Y -> Z where X - Z are indeed adjacent.
    Orient X - Z to X -> Z otherwise a cycle arises.

    Args:
        pdag (PDAG): PDAG before application of rule.

    Returns:
        PDAG: PDAG after application of rule.
    """
    copy_pdag = pdag.copy()
    for edge in copy_pdag.undir_edges:
        reverse_edge = edge[::-1]
        test_edges = [edge, reverse_edge]
        for tail, head in test_edges:
            orient = False
            undir_children = copy_pdag.children(tail)
            if undir_children:
                for child in undir_children:
                    if head in copy_pdag.children(child):
                        orient = True
            if orient:
                copy_pdag.undir_to_dir_edge(tail=tail, head=head)
                break
    return copy_pdag


def rule_3(pdag: PDAG) -> PDAG:
    """Orient X - Z to X -> Z, whenever there are two triples
    X - Y1 -> Z and X - Y2 -> Z such that Y1 and Y2 are non-adjacent.

    Args:
        pdag (PDAG): PDAG before application of rule.

    Returns:
        PDAG: PDAG after application of rule.
    """
    copy_pdag = pdag.copy()
    for edge in copy_pdag.undir_edges:
        reverse_edge = edge[::-1]
        test_edges = [edge, reverse_edge]
        for tail, head in test_edges:
            # if true that tail - node1 -> head and tail - node2 -> head
            # while {node1 U node2} = 0 then orient tail -> head
            orient = False
            if len(copy_pdag.undir_neighbors(tail)) >= 2:
                undir_n = copy_pdag.undir_neighbors(tail)
                selection = [
                    (node1, node2)
                    for node1, node2 in combinations(undir_n, 2)
                    if not copy_pdag.is_adjacent(node1, node2)
                ]
                if selection:
                    for node1, node2 in selection:
                        if head in copy_pdag.parents(node1).intersection(copy_pdag.parents(node2)):
                            orient = True
            if orient:
                copy_pdag.undir_to_dir_edge(tail=tail, head=head)
                break
    return pdag


def rule_4(pdag: PDAG) -> PDAG:
    """Orient X - Y1 to X -> Y1, whenever there are
    two triples with X - Z and X - Y1 <- Z and X - Y2 -> Z
    such that Y1 and Y2 are non-adjacent.

    Args:
        pdag (PDAG): PDAG before application of rule.

    Returns:
        PDAG: PDAG after application of rule.
    """
    copy_pdag = pdag.copy()
    for edge in copy_pdag.undir_edges:
        reverse_edge = edge[::-1]
        test_edges = [edge, reverse_edge]
        for tail, head in test_edges:
            orient = False
            if len(copy_pdag.undir_neighbors(tail)) > 0:
                undirected_n = copy_pdag.undir_neighbors(tail)
                for undir_n in undirected_n:
                    if tail in copy_pdag.children(undir_n):
                        children_select = list(copy_pdag.children(undir_n))
                        if children_select:
                            for parent in children_select:
                                if head in copy_pdag.children(parent):
                                    orient = True
            if orient:
                copy_pdag.undir_to_dir_edge(tail=tail, head=head)
                break
    return pdag


def dag2cpdag(dag: nx.DiGraph) -> PDAG:
    """Convertes a DAG into its unique CPDAG

    Args:
        dag (nx.DiGraph): DAG the CPDAG corresponds to.

    Returns:
        PDAG: unique CPDAG
    """
    copy_dag = dag.copy()
    # Skeleton
    skeleton = nx.to_pandas_adjacency(copy_dag.to_undirected())
    # v-Structures
    vstructures = vstructs(dag=copy_dag)

    for edge in vstructures:  # orient v-structures
        skeleton.loc[edge[::-1]] = 0

    pdag_init = PDAG.from_pandas_adjacency(skeleton)

    # Apply Meek Rules
    cpdag = rule_1(pdag=pdag_init)
    cpdag = rule_2(pdag=cpdag)
    cpdag = rule_3(pdag=cpdag)
    cpdag = rule_4(pdag=cpdag)

    return cpdag
