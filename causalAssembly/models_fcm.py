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

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from sympy import Eq, Expr, Number, Symbol, lambdify, ordered, symbols
from sympy.stats import sample as sympy_sample
from sympy.stats.rv import RandomSymbol

logger = logging.getLogger(__name__)


class FCM:
    """Class to define, intervene and sample from an FCM.

    Examples:
        ```python
        from sympy import symbols
        from sympy.stats import Normal, Uniform, Gamma

        x, y, z = symbols('x,y,z')

        eq_x = Eq(x, Uniform("noise", left=-1, right=1))
        eq_y = Eq(y, 2 * x ** 2 + Normal("error", 0, .5))
        eq_z = Eq(z, 9 * y * x * Gamma("some_name", .5, .5))

        eq_list = [eq_x, eq_y, eq_z]


        self = FCM(name='test', seed=2023)
        self.input_fcm(eq_list)
        self.draw(size=10)
        ```

    """

    def __init__(self, name: str | None = None, seed: int = 2023):
        self.name = name
        self._random_state = np.random.default_rng(seed=seed)
        self.__init_dag()
        self.last_df: pd.DataFrame
        self.__init_mutilated_dag()

    def __init_dag(self):
        self.graph = nx.DiGraph(name=self.name)

    def __init_mutilated_dag(self):
        self.mutilated_dags = dict()

    @property
    def source_nodes(self) -> list:
        """Returns source nodes in the current DAG.

        Returns:
            list: List of source nodes.
        """
        return [node for node in self.nodes if len(self.parents(of_node=node)) == 0]

    @property
    def causal_order(self) -> list[Symbol]:
        """Returns the causal order of the current graph.
        Note that this order is in general not unique. To
        ensure uniqueness, we additionally sort lexicograpically.

        Returns:
            list[Symbol]: Causal order
        """
        return list(nx.lexicographical_topological_sort(self.graph, key=lambda x: str(x)))

    @property
    def nodes(self) -> list[Symbol]:
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
    def interventions(self) -> list:
        """Returns all interventions performed on the original graph

        Returns:
            list: list of intervened upon nodes in do(x) notation.
        """
        return list(self.mutilated_dags.keys())

    def interventional_amat(self, which_intervention: int | str) -> pd.DataFrame:
        """Returns the adjacency matrix of a chosen mutilated DAG.

        Args:
            which_intervention (int | str): Integer count of your chosen intervention or
                literal string.

        Raises:
            ValueError: "The intervention you provide does not exist."

        Returns:
            pd.DataFrame: Adjacency matrix.
        """
        if isinstance(which_intervention, str) and which_intervention not in self.interventions:
            raise ValueError("The intervention you provide does not exist.")

        if isinstance(which_intervention, int) and which_intervention > len(self.interventions):
            raise ValueError("The intervention you index does not exist.")

        if isinstance(which_intervention, int):
            which_intervention = self.interventions[which_intervention]

        mutilated_dag = self.mutilated_dags[which_intervention].copy()
        return nx.to_pandas_adjacency(mutilated_dag, weight=None)

    def parents(self, of_node: Symbol) -> list[Symbol]:
        """Return parents of node in question.

        Args:
            of_node (str): Node in question.

        Returns:
            list[str]: parent set.
        """
        return list(self.graph.predecessors(of_node))

    def parents_of(self, node: Symbol, which_graph: nx.DiGraph) -> list[Symbol]:
        """Return parents of node in question for a chosen DAG.

        Args:
            node (Symbol): node whose parents to return.
            which_graph (nx.DiGraph): which graph along the interventions.

        Returns:
            list[Symbol]: list of parents.
        """
        return list(which_graph.predecessors(node))

    def causal_order_of(self, which_graph: nx.DiGraph) -> list[Symbol]:
        """Returns the causal order of the chosen graph.
        Note that this order is in general not unique. To
        ensure uniqueness, we additionally sort lexicograpically.

        Returns:
            list[Symbol]: Causal order
        """
        return list(nx.lexicographical_topological_sort(which_graph, key=lambda x: str(x)))

    def source_nodes_of(self, which_graph: nx.DiGraph) -> list:
        """Returns the source nodes of a chosen graph. This is mainly for
        choosing different mutilated DAGs.

        Args:
            which_graph (nx.DiGraph): DAG from which source nodes should be returned.

        Returns:
            list: List of nodes.
        """
        return [
            node
            for node in which_graph.nodes
            if len(self.parents_of(node=node, which_graph=which_graph)) == 0
        ]

    def input_fcm(self, fcm: list[Eq]):
        """
        Automatically builds up DAG according to the FCM fed in.
        Args:
            fcm (list): list of sympy equations generated as:
                    ```[python]
                    x,y = symbols('x,y')
                    term_x = Eq(x, Normal('x', 0,1))
                    term_y = Eq(y, 2*x**2*Normal('noise', 0,1))
                    fcm = [term_x, term_y]
                    ```
        """
        nodes_implied = [node.lhs.free_symbols.pop() for node in fcm]
        edges_implied = []
        for term in fcm:
            if not isinstance(term.rhs, RandomSymbol):
                if term.rhs.atoms(RandomSymbol):
                    edges_implied.extend(
                        [
                            (atom, term.lhs)
                            for atom in term.rhs.free_symbols
                            if str(atom) != str(term.rhs.atoms(RandomSymbol).pop())
                        ]
                    )
                else:
                    edges_implied.extend([(atom, term.lhs) for atom in term.rhs.atoms(Symbol)])

        g = self.graph
        g.add_nodes_from(nodes_implied)
        g.add_edges_from(edges_implied)

        term_dict = {}
        for term in fcm:
            term_dict[term.lhs] = {"term": term.rhs}

        nx.set_node_attributes(g, term_dict)

    def function_of(self, node: Symbol) -> dict:
        """Returns functional assignment for node in question.

        Args:
            node (Symbol): node corresponding to lhs.

        Returns:
            dict: key is node and value rhs of functional assignment.
        """
        if node not in self.graph.nodes:
            if node in [str(node) for node in self.nodes]:
                raise AssertionError(
                    "You probably defined a string. Node has to be a symbol, check out",
                    list(self.graph.nodes),
                )
            else:
                raise AssertionError("Node has to be in the graph")

        return {node: self.graph.nodes[node]["term"]}

    def display_functions(self) -> dict:
        """Displays all functional assignments inputted into the FCM.

        Returns:
            dict: Dict with keys equal to nodes and values equal to
                functional assignments.
        """
        fcm_dict = {}
        for node in self.causal_order:
            fcm_dict[node] = self.graph.nodes[node]["term"]

        return fcm_dict

    def sample(
        self,
        size: int,
        additive_gaussian_noise: bool = False,
        snr: None | float = 1 / 2,
        source_df: None | pd.DataFrame = None,
    ) -> pd.DataFrame:
        """Draw samples from the joint distribution that factorizes
        according to the DAG implied by the FCM fed in. To avoid
        unexpected/unintended behavior, avoid defining fully
        deterministic equation systems.
        If parameters in noise terms are additive and left unevaluated,
        they're set according to a chosen Signal-To-Noise (SNR) ratio.
        For convenience, you can add additive Gaussian noise to each equation.
        This will never overwrite any of the chosen noise distributions.
        You may also feed in a data frame for noise distributions (see below
        for more details).

        Args:
            size (int): Number of samples to draw.
            additive_gaussian_noise (bool, optional): This will attach additive
                Gaussian noise to all terms without a RandomSymbol that are not
                source nodes. It acts merely as a convenience option. Variance
                will then be chosen according to SNR. Defaults to False.
            snr (None | float, optional): Signal-to-noise ratio
                \\( SNR =  \\frac{\\text{Var}(\\hat{X})}{\\hat\\sigma^2}. \\).
                Defaults to 1/2.
            source_df (None | pd.DataFrame, optional): Data frame containing source node data.
                The sample size must be at least as large as the number of samples
                you'd like to draw. Defaults to None.

        Raises:
            AttributeError: if source node parameters are not given explicitly.
            ValueError: if source node sample size is too small.
            ValueError: if scale parameters are left unevaluated for non-additive terms.

        Returns:
            pd.DataFrame:  Data frame with rows of length `size` and columns equal to the
                number of nodes in the graph.
        """
        return self._sample(
            size=size,
            additive_gaussian_noise=additive_gaussian_noise,
            snr=snr,
            source_df=source_df,
            which_graph=self.graph,
        )

    def interventional_sample(
        self,
        size: int,
        which_intervention: str | int = 0,
        additive_gaussian_noise: bool = False,
        snr: None | float = 1 / 2,
        source_df: None | pd.DataFrame = None,
    ) -> pd.DataFrame:
        """Draw samples from the interventional distribution that factorizes
        according to the mutilated DAG after performing one or multiple
        interventions. Otherwise the method behaves similar to sampling from the
        non-interventional joint distribution. By default samples are drawn from the
        first intervention you performed. If you intervened upon more than one node,
        you'll have swith to another intervention for sampling from the corresponding
        interventional distribution.

        Args:
            size (int): Number of samples to draw.
            which_intervention (str | int): Which interventional distribution to draw
                from. We recommend using integer counts starting from zero. But you can
                also provide the literal string here, e.g. if you intervened on say two
                nodes `x,y` then you would need to provide here: `do([x, y])`.
            additive_gaussian_noise (bool, optional): This will attach additive Gaussian noise
                to all terms without a RandomSymbol that are not source nodes. It acts merely as
                a convenience option. Variance will then be chosen according to SNR.
                Defaults to False.
            snr (None | float, optional): Signal-to-noise ratio
                \\( SNR =  \\frac{\\text{Var}(\\hat{X})}{\\hat\\sigma^2}. \\). Defaults to 1/2.
            source_df (None | pd.DataFrame, optional): Data frame containing source node data.
                The sample size must be at least as large as the number of samples
                you'd like to draw. Defaults to None.

        Raises:
            NotImplementedError: Raised when `which_intervention` is not of correct form.

        Returns:
            pd.DataFrame: Data frame with rows of length `size` and columns equal to the
                number of nodes in the graph.
        """
        if isinstance(which_intervention, str):
            int_choice = which_intervention
        elif isinstance(which_intervention, int):
            int_choice = self.interventions[which_intervention]
        else:
            raise NotImplementedError(
                f"which_intervention has to be \
                the literal string or an integer \
                starting at count zero indicating \
                which intervention in {self.interventions} \
                to use."
            )

        return self._sample(
            size=size,
            additive_gaussian_noise=additive_gaussian_noise,
            snr=snr,
            source_df=source_df,
            which_graph=self.mutilated_dags[int_choice],
        )

    def _sample(
        self,
        size: int,
        which_graph: nx.DiGraph,
        additive_gaussian_noise: bool = False,
        snr: None | float = 1 / 2,
        source_df: None | pd.DataFrame = None,
    ) -> pd.DataFrame:
        """Draw samples from the joint distribution that factorizes
        according to the DAG implied by the FCM fed in. To avoid
        unexpected/unintended behavior, avoid defining fully
        deterministic equation systems.
        If parameters in noise terms are additive and left unevaluated,
        they're set according to a chosen Signal-To-Noise (SNR) ratio.
        For convenience, you can add additive Gaussian noise to each equation.
        This will never overwrite any of the chosen noise distributions.
        You may also feed in a data frame for noise distributions (see below
        for more details).

        Args:
            size (int): Number of samples to draw.
            additive_gaussian_noise (bool, optional): _description_. Defaults to False.
            snr (None | float, optional): Signal-to-noise ratio
                \\( SNR =  \\frac{\\text{Var}(\\hat{X})}{\\hat\\sigma^2}. \\).
                Defaults to 1/2.
            source_df (None | pd.DataFrame, optional): Data frame conaining source node data.
                The sample size must be at least as large as the number of samples
                you'd like to draw. Defaults to None.

        Raises:
            AttributeError: if source node parameters are not given explicitly.
            ValueError: if source node sample size is too small.
            ValueError: if scale parameters are left unevaluated for non-additive terms.

        Returns:
            pd.DataFrame:  Data frame with rows of lenght size and columns equal to the
                number of nodes in the graph.
        """

        if source_df is not None and not self.__source_df_condition(source_df):
            raise AssertionError("Names in source_df don't match nodenames in graph.")

        df = pd.DataFrame()
        for order in self.causal_order_of(which_graph=which_graph):
            if order in self.source_nodes_of(which_graph=which_graph):
                if source_df is not None and str(order) in source_df.columns:
                    if source_df[str(order)].shape[0] < size:
                        raise ValueError(
                            "Sample size of source node data must be at least \
                            as large as the number of samples you'd like to draw."
                        )
                    df[str(order)] = source_df[str(order)].sample(
                        n=size,
                        replace=False,
                        random_state=self._random_state,
                        ignore_index=True,
                    )

                elif isinstance(which_graph.nodes[order]["term"], RandomSymbol):
                    if not self.__distribution_parameters_explicit(order, which_graph=which_graph):
                        raise AttributeError("Source node parameters need to be given explicitly.")
                    df[str(order)] = sympy_sample(
                        which_graph.nodes[order]["term"],
                        seed=self._random_state,
                        size=size,
                    )

                elif isinstance(which_graph.nodes[order]["term"], Number):
                    df[str(order)] = np.repeat(which_graph.nodes[order]["term"], repeats=size)

                else:
                    raise NotImplementedError(
                        "Source nodes need to have a fully parameterized distribution, \
                        or need to be drawn from an appropriate data frame, or fixed to \
                        a single real number."
                    )
                continue

            fcm_expr = which_graph.nodes[order]["term"]

            if fcm_expr.atoms(RandomSymbol):
                if self.__distribution_parameters_explicit(order, which_graph=which_graph):
                    df[str(fcm_expr.atoms(RandomSymbol).pop())] = sympy_sample(
                        fcm_expr.atoms(RandomSymbol).pop(),
                        size=size,
                        seed=self._random_state,
                    )
                else:
                    df[str(fcm_expr.atoms(RandomSymbol).pop())] = np.zeros(size)

            df[str(order)] = self.__eval_expression(df=df, fcm_expr=fcm_expr)

            if fcm_expr.atoms(RandomSymbol) and not self.__distribution_parameters_explicit(
                order, which_graph=which_graph
            ):
                if not fcm_expr.is_Add:
                    raise ValueError(
                        "Noise term in "
                        + str(order)
                        + "="
                        + str(fcm_expr)
                        + " not additive. Scale parameter selection via SNR \
                        makes sense only for additive noise."
                    )
                logger.warning(
                    "I'll choose the noise scale in "
                    + str(order)
                    + "="
                    + str(fcm_expr)
                    + " according to the given SNR."
                )
                noise_var = df[str(order)].var() / snr
                df[str(order)] = df[str(order)] + sympy_sample(
                    fcm_expr.atoms(RandomSymbol)
                    .pop()
                    .subs(self.__unfree_symbol(fcm_expr), np.sqrt(noise_var)),
                    size=size,
                    seed=self._random_state,
                )

            if additive_gaussian_noise:
                if fcm_expr.atoms(RandomSymbol):
                    logger.warning(
                        "Noise already defined in "
                        + str(order)
                        + "="
                        + str(fcm_expr)
                        + ". I won't override this."
                    )
                else:
                    noise = symbols("noise")
                    noise_var = df[str(order)].var() / snr
                    df[str(noise)] = self._random_state.normal(
                        loc=0, scale=np.sqrt(noise_var), size=size
                    )
                    fcm_expr = which_graph.nodes[order]["term"] + noise
                    df[str(order)] = self.__eval_expression(df=df, fcm_expr=fcm_expr)

        self.last_df = df[[str(order) for order in self.causal_order]]

        return self.last_df

    def __unfree_symbol(self, fcm_expr) -> set[Symbol]:
        random_symbs = fcm_expr.atoms(Symbol).difference(fcm_expr.free_symbols)
        return {
            unfree
            for unfree in random_symbs
            if str(unfree) != str(fcm_expr.atoms(RandomSymbol).pop())
        }.pop()

    def __eval_expression(self, df: pd.DataFrame, fcm_expr: Expr) -> pd.DataFrame:
        """Eval given fcm_expression with the values in given dataframe

        Args:
            df (pd.DataFrame): Data frame.
            fcm_expr (Expr): Sympy expression.

        Returns:
            pd.DataFrame: Data frame after eval.
        """

        correct_order = list(ordered(fcm_expr.free_symbols))  # self.__return_ordered_args(fcm_expr)
        cols = [str(col) for col in correct_order]
        evaluator = lambdify(correct_order, fcm_expr, "scipy")

        return evaluator(*[df[col] for col in cols])

    def __distribution_parameters_explicit(self, order: Symbol, which_graph: nx.DiGraph) -> bool:
        """Returns true if distribution parameters
        are given explicitly, not symbolically.

        Args:
            order (node): node in graph

        Returns:
            bool:
        """
        return len(which_graph.nodes[order]["term"].free_symbols) == len(
            which_graph.nodes[order]["term"].atoms(Symbol)
        )

    def __source_df_condition(self, source_df: pd.DataFrame) -> bool:
        """Returns true if source_df colnames and graph nodenames agree.

        Args:
            source_df (None | pd.DataFrame): data frame containing source node data.

        Returns:
            bool: True if names agree
        """
        return {str(col) for col in source_df.columns}.issubset(
            {str(node) for node in self.source_nodes}
        )

    def intervene_on(self, nodes_values: dict[Symbol, RandomSymbol | float]):
        """Specify hard or soft intervention. If you want to intervene
        upon more than one node provide a list of nodes to intervene on
        and a list of corresponding values to set these nodes to.
        (see example). The mutilated dag will automatically be
        stored in `mutiliated_dags`.

        Args:
            nodes_values (dict[Symbol, RandomSymbol | float]): either single real
                number or RandmSymbol. If you provide more than one
                intervention just provide more key-value pairs.

        Raises:
            AssertionError: If node(s) are not in the graph

        Example:
            ```python
            x,y = symbols("x,y")
            eq_x = Eq(x, Gamma("source", 1,1))
            eq_y = Eq(y, 4*x**3 + Uniform("noise", left=-0.5, right=0.5))

            example_fcm = FCM()
            example_fcm.input_fcm([eq_x,eq_y])
            # Hard intervention
            example_fcm.intervene_on(nodes_values = {y : 4})
            # Soft intervention
            example_fcm.intervene_on(nodes_values = {y : Normal("noise",0,1)})

            ```
        """

        if not set(nodes_values.keys()).issubset(set(self.nodes)):
            raise AssertionError(
                "One or more nodes you want to intervene upon are not in the graph."
            )

        mutilated_dag = self.graph.copy()

        for node, value in nodes_values.items():
            intervention = Eq(node, value)
            old_incoming = self.parents(of_node=node)
            edges_to_remove = [(old, node) for old in old_incoming]
            mutilated_dag.remove_edges_from(edges_to_remove)
            mutilated_dag.nodes[node]["term"] = intervention.rhs

        self.mutilated_dags[
            f"do({list(nodes_values.keys())})"
        ] = mutilated_dag  # specifiying the same set twice will override

    def show(self, header: str | None = None, with_nodenames: bool = True) -> plt:
        """Plots the current DAG.

        Args:
            header (str | None, optional): Header for the DAG. Defaults to None.
            with_nodenames (bool, optional): Whether or not to use nodenames as
                labels in the plot. Defaults to True.

        Returns:
            plt: Plot of the DAG.
        """
        if header is None:
            header = ""
        return self._show(which_graph=self.graph, header=header, with_nodenames=with_nodenames)

    def show_mutilated_dag(
        self, which_intervention: str | int = 0, with_nodenames: bool = True
    ) -> plt:
        """Plot mutilated DAG

        Args:
            which_intervention (str | int, optional): Which interventional distribution
                should be represented by a DAG. Defaults to 0.
            with_nodenames (bool, optional): Whether or not to use nodenames as
                labels in the plot. Defaults to True.

        Returns:
            plt: Plot of the mutilated DAG.
        """
        if isinstance(which_intervention, int):
            which_intervention = self.interventions[which_intervention]

        return self._show(
            which_graph=self.mutilated_dags[which_intervention],
            header=which_intervention,
            with_nodenames=with_nodenames,
        )

    def _show(self, which_graph: nx.DiGraph, header: str, with_nodenames: bool):
        """Plots the graph by giving extra weight to nodes
        with high in- and out-degree.
        """
        cmap = plt.get_cmap("Blues")
        fig, ax = plt.subplots()
        center: np.ndarray = np.array([0, 0])
        pos = nx.spring_layout(
            which_graph,
            center=center,
            seed=10,
            k=50,
        )

        labels = {}
        for node in self.nodes:
            labels[node] = node

        max_in_degree = max([d for _, d in which_graph.in_degree()])
        max_out_degree = max([d for _, d in which_graph.out_degree()])

        nx.draw_networkx_nodes(
            which_graph,
            pos=pos,
            ax=ax,
            cmap=cmap,
            vmin=-0.2,
            vmax=1,
            node_color=[
                (d + 10) / (max_in_degree + 10) for _, d in which_graph.in_degree(self.nodes)
            ],
            node_size=[
                500 * (d + 1) / (max_out_degree + 1) for _, d in which_graph.out_degree(self.nodes)
            ],
        )

        if with_nodenames:
            nx.draw_networkx_labels(
                which_graph,
                pos=pos,
                labels=labels,
                font_size=8,
                font_color="w",
                alpha=0.4,
            )

        nx.draw_networkx_edges(
            which_graph,
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
            f"{header}",
            horizontalalignment="center",
        )

        ax.axis("off")
