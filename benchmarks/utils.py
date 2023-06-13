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
from collections import defaultdict

import lingam
import networkx as nx
import numpy as np
import pandas as pd
from castle.algorithms import Notears
from castle.algorithms.anm import ANMNonlinear
from castle.algorithms.gradient.gran_dag import GraNDAG
from causallearn.search.ConstraintBased.PC import pc
from causallearn.search.ScoreBased.ExactSearch import bic_exact_search
from causallearn.search.ScoreBased.GES import ges
from sklearn.linear_model import LassoLarsIC, LinearRegression

from causalAssembly.metrics import DAGmetrics
from causalAssembly.models_dag import ProcessCell, ProductionLineGraph
from causalAssembly.pdag import PDAG


def run_benchmark(
    data_generator: ProductionLineGraph | ProcessCell,
    algorithms: list = [
        "astar",
        "pc",
        "ges",
        "lingam",
        "notears",
        "grandag",
        "anm",
        "snr",
    ],
    runs: int = 30,
    return_varsortability: bool = False,
    standardize_before_run: bool = True,
) -> dict:
    result_dict = defaultdict(list)
    final_dict = {}
    varsort = []
    for run in range(runs):
        df_run = data_generator.sample_from_drf(500, smoothed=True)
        if standardize_before_run:
            df_run = df_run / df_run.std()
        varsort.append(
            varsortability(data=df_run, ground_truth=data_generator.ground_truth)
        )
        sl = structure_learning_from_list(data=df_run, alg_list=algorithms)

        for alg, est in sl.items():
            assert est.shape[0] == est.shape[1] == df_run.shape[1]
            results = DAGmetrics(truth=data_generator.ground_truth, est=est)
            results.collect_metrics()

            df = pd.DataFrame(results.metrics, index=[run])
            result_dict[alg].append(df)

    for alg, _ in result_dict.items():
        final_dict[alg] = pd.concat(result_dict[alg], axis=0)
    if return_varsortability:
        final_dict["varsort"] = varsort
    return final_dict


def structure_learning_from_list(data: pd.DataFrame, alg_list):
    return_dict = {}
    if "astar" in alg_list:
        astar_est = fit_astar(data=data)
        return_dict["astar"] = astar_est
    if "lingam" in alg_list:
        lingam_est = fit_lingam(data=data)
        return_dict["lingam"] = lingam_est
    if "pc" in alg_list:
        pc_est = fit_pc(data=data)
        return_dict["pc"] = pc_est
    if "ges" in alg_list:
        ges_est = fit_ges(data=data)
        return_dict["ges"] = ges_est
    if "anm" in alg_list:
        anm_est = fit_anm(data=data)
        return_dict["anm"] = anm_est
    if "notears" in alg_list:
        notears_est = fit_notears(data=data)
        return_dict["notears"] = notears_est
    if "grandag" in alg_list:
        grandag_est = fit_grandag(data=data)
        return_dict["grandag"] = grandag_est
    if "snr" in alg_list:
        snr = fit_snr(data=data)
        return_dict["snr"] = snr

    return_dict["emptygraph"] = pd.DataFrame(
        np.zeros((data.shape[1], data.shape[1])),
        columns=data.columns,
        index=data.columns,
    )
    return return_dict


def to_dag_amat(cpdag_amat: pd.DataFrame) -> pd.DataFrame:
    """Turns PDAG into random member of the corresponding
    Markov equivalence class.

    Args:
        cpdag_amat (pd.DataFrame): PDAG representing the MEC

    Returns:
        pd.DataFrame: DAG as member of MEC.
    """

    pdag = PDAG.from_pandas(cpdag_amat)
    chosen_dag = pdag.to_dag()
    if not nx.is_directed_acyclic_graph(chosen_dag):
        raise TypeError("Graph provided is not a DAG!")

    return nx.to_pandas_adjacency(chosen_dag)


def causallearn2amat(causal_learn_graph: np.ndarray) -> np.ndarray:
    amat = np.zeros(causal_learn_graph.shape)
    for col in range(causal_learn_graph.shape[1]):
        for row in range(causal_learn_graph.shape[0]):
            if causal_learn_graph[row, col] == -1 and causal_learn_graph[col, row] == 1:
                amat[row, col] = 1
            if (
                causal_learn_graph[row, col] == -1
                and causal_learn_graph[col, row] == -1
            ):
                amat[row, col] = amat[col, row] = 1
    return amat


def fit_astar(data: pd.DataFrame) -> pd.DataFrame:
    astar, _ = bic_exact_search(data.to_numpy())
    return to_dag_amat(pd.DataFrame(astar, columns=data.columns, index=data.columns))


def fit_lingam(data: pd.DataFrame) -> pd.DataFrame:
    model = lingam.DirectLiNGAM()
    model.fit(data)
    return pd.DataFrame(
        (abs(model.adjacency_matrix_) > 0).astype(int),
        columns=data.columns,
        index=data.columns,
    )


def fit_pc(data: pd.DataFrame) -> pd.DataFrame:
    pcalg = pc(data.to_numpy(), node_names=list(data.columns), show_progress=False)
    pc_interim = causallearn2amat(pcalg.G.graph)
    return to_dag_amat(
        pd.DataFrame(pc_interim, columns=data.columns, index=data.columns)
    )


def fit_ges(data: pd.DataFrame) -> pd.DataFrame:
    gesalg = ges(data.to_numpy())
    ges_interim = causallearn2amat(gesalg["G"].graph)
    return to_dag_amat(
        pd.DataFrame(ges_interim, columns=data.columns, index=data.columns)
    )


def fit_anm(data: pd.DataFrame) -> pd.DataFrame:
    anm = ANMNonlinear(alpha=0.05)
    anm.learn(data=data.values)
    return pd.DataFrame(anm._causal_matrix, columns=data.columns, index=data.columns)


def fit_notears(data: pd.DataFrame) -> pd.DataFrame:
    notears = Notears()
    notears.learn(data=data.values)
    return pd.DataFrame(
        notears._causal_matrix, columns=data.columns, index=data.columns
    )


def fit_grandag(data: pd.DataFrame) -> pd.DataFrame:
    grandag = GraNDAG(input_dim=data.shape[1])
    grandag.learn(data=data.values)
    return pd.DataFrame(
        grandag._causal_matrix, columns=data.columns, index=data.columns
    )


# Taken from
# Reisach, A. G., Seiler, C., & Weichwald, S. (2021).
# Beware of the Simulated DAG! Causal Discovery Benchmarks May Be Easy To Game.
def fit_snr(data: pd.DataFrame) -> pd.DataFrame:
    """Take n x d data, order nodes by marginal variance and
    regresses each node onto those with lower variance, using
    edge coefficients as structure estimates."""
    X = data.to_numpy()
    LR = LinearRegression()
    LL = LassoLarsIC(criterion="bic")

    d = X.shape[1]
    W = np.zeros((d, d))
    increasing = np.argsort(np.var(X, axis=0))

    for k in range(1, d):
        covariates = increasing[:k]
        target = increasing[k]

        LR.fit(X[:, covariates], X[:, target].ravel())
        weight = np.abs(LR.coef_)
        LL.fit(X[:, covariates] * weight, X[:, target].ravel())
        W[covariates, target] = LL.coef_ * weight

    return pd.DataFrame(
        np.abs(W > 0).astype(int), columns=data.columns, index=data.columns
    )


# Taken from
# Reisach, A. G., Seiler, C., & Weichwald, S. (2021).
# Beware of the Simulated DAG! Causal Discovery Benchmarks May Be Easy To Game.
def varsortability(data: pd.DataFrame, ground_truth: pd.DataFrame, tol=1e-9):
    """Takes n x d data and a d x d adjaceny matrix,
    where the i,j-th entry corresponds to the edge weight for i->j,
    and returns a value indicating how well the variance order
    reflects the causal order."""
    X = data.to_numpy()
    W = ground_truth.to_numpy()
    E = W != 0
    Ek = E.copy()
    var = np.var(X, axis=0, keepdims=True)

    n_paths = 0
    n_correctly_ordered_paths = 0

    for _ in range(E.shape[0] - 1):
        n_paths += Ek.sum()
        n_correctly_ordered_paths += (Ek * var / var.T > 1 + tol).sum()
        n_correctly_ordered_paths += (
            1 / 2 * ((Ek * var / var.T <= 1 + tol) * (Ek * var / var.T > 1 - tol)).sum()
        )
        Ek = Ek.dot(E)

    return n_correctly_ordered_paths / n_paths
