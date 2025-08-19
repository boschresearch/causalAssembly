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

import itertools
import logging

import networkx as nx
import numpy as np

logger = logging.getLogger(__name__)


def merge_dags(
    dag_to_insert: nx.DiGraph,
    target_dag: nx.DiGraph,
    mapping: dict,
    remove_in_edges_in_target_dag: bool = False,
) -> nx.DiGraph:
    """Dag_to_insert will be connected to target_tag via mapping dict.

    Args:
        dag_to_insert (nx.DiGraph): DAG to insert.
        target_dag (nx.DiGraph): DAG on which to map.
        mapping (dict): Mapping from insert to target dag e.g.
            {C1: D1, C5: D4} where node C1 from insert dag will
            be mapped to node D1 of target dag.
        remove_in_edges_in_target_dag (bool, optional): Defaults to False.

    Raises:
        ValueError: node does not exist in target_dag
        ValueError: node does not exist in dag_to_insert
    Returns:
        nx.DiGraph: merged DAG
    """
    for old_node_name, new_node_name in mapping.items():
        if new_node_name not in target_dag.nodes():
            raise ValueError(f"{new_node_name} does not exist in target_dag")
        if old_node_name not in dag_to_insert.nodes():
            raise ValueError(f"{old_node_name} does not exist in dag_to_insert")

    no_of_nodes_insert_dag = len(dag_to_insert.nodes())
    no_of_nodes_target_dag = len(target_dag.nodes())
    if no_of_nodes_insert_dag > no_of_nodes_target_dag:
        logger.warning(
            f"you are trying to merge a DAG of size={no_of_nodes_insert_dag} "
            f"into a smaller DAG of size={no_of_nodes_target_dag}. "
            f"If this is not intentional consider changing the order"
        )

    # remove all in edges on target node
    if remove_in_edges_in_target_dag:
        for node in mapping.values():
            e = target_dag.in_edges(node)
            target_dag.remove_edges_from(list(e))

    # rename nodes to glue together
    dag = nx.relabel_nodes(dag_to_insert, mapping)

    return nx.compose(dag, target_dag)


def merge_dags_via_edges(
    left_dag: nx.DiGraph,
    right_dag: nx.DiGraph,
    edges: list[tuple] | None = None,
    isolate_target_nodes: bool = False,
):
    """Merges two dags via a list of edges.

    Args:
        left_dag (nx.DiGraph): dag to merge to right_dag
        right_dag (nx.DiGraph):  dag to merge left_dag into
        edges (list[tuple], optional): list of edges that connect the two dags.
            Defaults to None.
        isolate_target_nodes (bool, optional): bool if True all incoming edges
            from the right_dag into the target node are removed:
            all influence from the left_dag, defined via edges list.
            Defaults to False.

    Raises:
        ValueError: source or target nodes are not available in left dag
        ValueError: source or target nodes are not available in right dag

    """
    if not edges:
        edges = list()
    source_nodes = set([t[0] for t in edges])
    target_nodes = set([t[1] for t in edges])

    if not source_nodes.issubset(set(left_dag.nodes)):
        raise ValueError(
            f"At least one of the source nodes: {source_nodes} "
            f"cannot be found in the left DAGs nodes: {left_dag.nodes}"
        )

    if not target_nodes.issubset(set(right_dag.nodes)):
        raise ValueError(
            f"At least one of the target nodes: {target_nodes} "
            f"cannot be found in right DAGs nodes: {right_dag.nodes}"
        )

    if isolate_target_nodes:
        for node in target_nodes:
            # cast to list to work with value not with reference
            edges_to_target_node = list(right_dag.in_edges(node))
            right_dag.remove_edges_from(edges_to_target_node)

    merged_dag = nx.compose(left_dag, right_dag)

    merged_dag.add_edges_from(edges, **{"connector": True})

    return merged_dag


def tuples_from_cartesian_product(l1: list, l2: list) -> list[tuple]:
    """Given two lists l1 and l2 this creates the cartesian product and returns all tuples.

    Args:
        l1 (list): First list of nodes
        l2 (list): Second list of nodes typically

    Returns:
        list[tuple]: list of edges typically

    Examples::

        l1 = [0,1,2]
        l2 = ['a','b','c']
        >>> tuples_from_cartesian_product(l1,l2)
        [(0,'a'), (0,'b'), (0,'c'), (1,'a'), (1,'b'), (1,'c'), (2,'a'), (2,'b'), (2,'c')]

    """
    return [
        (tail, head)
        for tail, head in itertools.product(
            l1,
            l2,
        )
    ]


def _bootstrap_sample(
    rng: np.random.Generator, data: np.ndarray, size: int | None = None
) -> np.ndarray:
    """Generate bootstrap sample, i.e.

    random sample with replacement of length `size` from 1-d array.

    Args:
        rng (np.random.Generator): Random number generator
        data (np.array): 1-d array
        size (int, optional): Size of bootstrap sample. If set to `None`,
        we set size to length of input array

    Returns:
        np.array: Bootstrap sample of data
    """
    if not size:
        size = len(data)
    idx = rng.choice(len(data), size, replace=True)
    bootstrap = data[idx]
    return bootstrap
