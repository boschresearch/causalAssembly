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
import networkx as nx
import numpy as np
import pandas as pd
from sympy import Eq, Symbol, lambdify, symbols
from sympy.stats import sample
from sympy.stats.rv import RandomSymbol


class HandCrafted_FCM:
    """Class to define and sample from a FCM."""

    def __init__(self, name: str, seed: int):
        self.name = name
        self.graph: nx.DiGraph()
        self.rand = np.random.default_rng(seed)
        self.__init_dag()
        self.last_df = pd.DataFrame

    def __init_dag(self):
        self.graph = nx.DiGraph(name=self.name)

    @property
    def source_nodes(self):
        return [
            node
            for node in list(self.graph.nodes())
            if isinstance(self.graph.nodes[node]["term"], RandomSymbol)
        ]

    def input_fcm(self, fcm: list[Eq]):
        """
        Automatically builds up DAG according to the FCM fed in.
        Args:
            fcm (list): list of sympy equations generated as:
                    x,y = symbols('x,y')
                    term_x = Eq(x, Normal('x', 0,1))
                    term_y = Eq(y, 2*x)
                    fcm = [term_x, term_y]
        """
        edges_implied = []
        for term in fcm:
            if not isinstance(term.rhs, RandomSymbol):
                edges_implied.extend(
                    [(symb, term.lhs) for symb in term.rhs.atoms(Symbol)]
                )

        g = self.graph
        g.add_edges_from(edges_implied)

        term_dict = {}
        for term in fcm:
            term_dict[term.lhs] = {"term": term.rhs}

        nx.set_node_attributes(g, term_dict)

    def override_noise_distributions(
        self, override: dict
    ):  # allow to override default noise distributions
        pass

    def _unpack(self, df: pd.DataFrame) -> list[pd.Series]:
        return [df[col] for col in df.columns]

    def draw(
        self, size: int, add_noise: bool = True, snr: float = 2 / 3
    ) -> pd.DataFrame:
        """
        Draw samples from the joint distribution that factorizes
        according to the DAG implied by the FCM fed in. To ensure
        identifiability, add_noise should be set to `True`. Gaussian
        noise will then be added to each node where the variance is
        scaled according to the Signal-To-Noise (SNR) ration specified.
        The mean of the noise term is taken to be zero.

        Args:
            size (int): Number of samples to draw.
            add_noise (bool, optional): Whether additive noise is added.
                Defaults to True (recommended).
            snr (float, optional):
                \\( SNR =  \\frac{\\text{Var}(\\hat{X})}{\\hat\\sigma^2}. \\)
                Defaults to 2/3.

        Returns:
            pd.DataFrame: Data frame with rows = size and columns equal to the
                number of nodes in the graph.
        """
        df = pd.DataFrame()
        topological_order = list(nx.topological_sort(self.graph))

        for order in topological_order:
            if order in self.source_nodes:
                df[order] = sample(
                    self.graph.nodes[order]["term"], size=size, seed=self.rand
                )
            else:
                fcm_expr = self.graph.nodes[order]["term"]
                fcm_args = list(fcm_expr.atoms(Symbol))
                evaluator = lambdify(fcm_args, fcm_expr, "numpy")
                df[order] = evaluator(
                    *self._unpack(df[fcm_args])
                )
                if add_noise:
                    noise = symbols("noise")
                    noise_var = df[order].var() / snr
                    df[noise] = self.rand.normal(
                        loc=0, scale=np.sqrt(noise_var), size=size
                    )
                    fcm_expr = self.graph.nodes[order]["term"] + noise
                    fcm_args = list(fcm_expr.atoms(Symbol))
                    evaluator = lambdify(fcm_args, fcm_expr, "numpy")
                    df[order] = evaluator(
                        *self._unpack(df[fcm_args])
                    )
        self.last_df = df[topological_order]
        return df[topological_order]
