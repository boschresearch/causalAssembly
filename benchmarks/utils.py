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

import logging
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from itertools import repeat
from typing import Any

import lingam
import networkx as nx
import numpy as np
import pandas as pd
from castle.algorithms import Notears
from castle.algorithms.gradient.gran_dag import GraNDAG
from causallearn.search.ConstraintBased.PC import pc
from causallearn.search.ScoreBased.GES import ges
from cdt.metrics import SID
from dodiscover.context_builder import make_context
from dodiscover.toporder.das import DAS
from dodiscover.toporder.score import SCORE
from sklearn.linear_model import LassoLarsIC, LinearRegression

from causalAssembly.metrics import DAGmetrics
from causalAssembly.models_dag import ProcessCell, ProductionLineGraph
from causalAssembly.pdag import PDAG, dag2cpdag

logger = logging.getLogger(__name__)


class BenchMarker:
    """Class to run causal discovery benchmarks.

    One instance can be
    the basis for several benchmark runs based on the same settings
    with different data generator objects.
    """

    def __init__(self):
        """Initializes the BenchMarker class."""
        self.algorithms: dict = {"snr": BenchMarker._fit_snr}
        self.collect_results: dict = {}
        self.num_runs: int
        self.prod_object: ProductionLineGraph | ProcessCell
        self.last_call: str
        self.n_select: int

    def include_pc(self):
        """Includes the PC-stable algorithm from the `causal-learn` package."""
        logger.info("PC algorithm added to benchmark routines.")
        self.algorithms["pc"] = BenchMarker._fit_pc

    def include_ges(self):
        """Includes GES from the `causal-learn` package."""
        logger.info("GES algorithm added to benchmark routines.")
        self.algorithms["ges"] = BenchMarker._fit_ges

    def include_notears(self):
        """Includes the NOTEARS algorithm from the `gcastle` package."""
        logger.info("NOTEARS added to benchmark routines.")
        self.algorithms["notears"] = BenchMarker._fit_notears

    def include_grandag(self):
        """Includes the Gran-DAG algorithm from the `gcastle` package."""
        logger.info("Gran-DAG added to benchmark routines.")
        self.algorithms["grandag"] = BenchMarker._fit_grandag

    def include_score(self):
        """Includes the SCORE algorithm from the `dodiscovery` package."""
        logger.info("SCORE algorithm added to benchmark routines.")
        self.algorithms["score"] = BenchMarker._fit_score

    def include_das(self):
        """Includes the DAS algorithm from the `dodiscovery` package."""
        logger.info("DAS algorithm added to benchmark routines.")
        self.algorithms["das"] = BenchMarker._fit_das

    def include_lingam(self):
        """Includes the DirectLiNGAM algorithm from the `lingam` package."""
        logger.info("Direct LiNGAM added to benchmark routines.")
        self.algorithms["lingam"] = BenchMarker._fit_lingam

    @staticmethod
    def _fit_pc(data: pd.DataFrame) -> PDAG:
        pcalg = pc(
            data.to_numpy(),
            node_names=list(data.columns),
            show_progress=False,
        )
        pc_interim = pd.DataFrame(
            BenchMarker._causallearn2amat(pcalg.G.graph),
            columns=data.columns,
            index=data.columns,
        )
        return PDAG.from_pandas_adjacency(pc_interim)

    @staticmethod
    def _causallearn2amat(causal_learn_graph: np.ndarray) -> np.ndarray:
        amat = np.zeros(causal_learn_graph.shape)
        for col in range(causal_learn_graph.shape[1]):
            for row in range(causal_learn_graph.shape[0]):
                if causal_learn_graph[row, col] == -1 and causal_learn_graph[col, row] == 1:
                    amat[row, col] = 1
                if causal_learn_graph[row, col] == -1 and causal_learn_graph[col, row] == -1:
                    amat[row, col] = amat[col, row] = 1
                if causal_learn_graph[row, col] == 1 and causal_learn_graph[col, row] == 1:
                    logger.warning(f"ambiguity found in {(row, col)}. I'll make it bidirected")
                    amat[row, col] = amat[col, row] = 1
        return amat

    @staticmethod
    def _fit_ges(data: pd.DataFrame) -> PDAG:
        gesalg = ges(data.to_numpy())
        ges_interim = pd.DataFrame(
            BenchMarker._causallearn2amat(gesalg["G"].graph),
            columns=data.columns,
            index=data.columns,
        )
        return PDAG.from_pandas_adjacency(ges_interim)

    # Taken from
    # Reisach, A. G., Seiler, C., & Weichwald, S. (2021).
    # Beware of the Simulated DAG! Causal Discovery Benchmarks May Be Easy To Game.
    @staticmethod
    def _fit_snr(data: pd.DataFrame) -> pd.DataFrame:
        """SNR algo.

        Take n x d data, order nodes by marginal variance and
        regresses each node onto those with lower variance, using
        edge coefficients as structure estimates.
        """
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

        return pd.DataFrame((np.abs(W) > 0).astype(int), columns=data.columns, index=data.columns)

    # Taken from
    # Reisach, A. G., Seiler, C., & Weichwald, S. (2021).
    # Beware of the Simulated DAG! Causal Discovery Benchmarks May Be Easy To Game.
    @staticmethod
    def varsortability(data: pd.DataFrame, ground_truth: pd.DataFrame, tol=1e-9):
        """Varsortability algo.

        Takes n x d data and a d x d adjaceny matrix,
        where the i,j-th entry corresponds to the edge weight for i->j,
        and returns a value indicating how well the variance order
        reflects the causal order.
        """
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

    @staticmethod
    def _fit_lingam(data: pd.DataFrame) -> pd.DataFrame:
        model = lingam.DirectLiNGAM()
        model.fit(data)
        return pd.DataFrame(
            (abs(model.adjacency_matrix_) > 0).astype(int),
            columns=data.columns,
            index=data.columns,
        )

    @staticmethod
    def _fit_notears(data: pd.DataFrame) -> pd.DataFrame:
        notears = Notears()
        notears.learn(data=data.values)
        return pd.DataFrame(notears._causal_matrix, columns=data.columns, index=data.columns)

    @staticmethod
    def _fit_grandag(data: pd.DataFrame) -> pd.DataFrame:
        grandag = GraNDAG(input_dim=data.shape[1])
        grandag.learn(data=data.values)
        return pd.DataFrame(grandag._causal_matrix, columns=data.columns, index=data.columns)

    @staticmethod
    def _emptygraph(size: int) -> np.ndarray:
        return np.zeros((size, size))

    @staticmethod
    def _fit_score(data: pd.DataFrame) -> pd.DataFrame:
        score = SCORE()
        ctxt = make_context()
        context = ctxt.variables(data.columns).build()
        score.learn_graph(data_df=data, context=context)
        return nx.to_pandas_adjacency(score.graph_)

    @staticmethod
    def _fit_das(data: pd.DataFrame) -> pd.DataFrame:
        score = DAS()
        ctxt = make_context()
        context = ctxt.variables(data.columns).build()
        score.learn_graph(data_df=data, context=context)
        return nx.to_pandas_adjacency(score.graph_)

    def run_benchmark(
        self,
        runs: int,
        prod_obj: ProductionLineGraph,
        n_select: int = 500,
        harmonize_via: str | None = "cpdag_transform",
        size_threshold: int = 50,
        parallelize: bool = False,
        n_workers: int = 4,
        seed_sequence: int = 1234,
        chunksize: int | None = None,
        external_dfs: list[pd.DataFrame | np.ndarray] | None = None,
        between_and_within_results: bool = False,
    ):
        """Run benchmark given the registered algorithms.

        Args:
            runs (int): number of simulation runs.
            prod_obj (ProductionLineGraph | ProcessCell): Data generator.
            n_select (int, optional): Number of samples to draw from `prod_obj`. Defaults to 500.
            harmonize_via (str | None, optional): Harmonization strategy.
                If CD-algorithms are included that output different types of graphs, i.e. the
                PC-algorithm outputs a CPDAG and the DirectLiNGAM a DAG one needs to harmonize. If
                set to `None`, no harmonization will be performed and the user needs to make sure
                that results are compatible. If `"cpdag_transform"` it selected all results
                including the ground truth will be transformed to the corresponding CPDAG. If
                `"best_dag_shd"` is selected, all DAGs in the implied MEC will be enumerated, the
                SHD calculated and the lowest (best) candidate DAG chosen. Defaults to
                "cpdag_transform".
            size_threshold (int) : size of threshold.
            parallelize (bool, optional): Whether to run on parallel processes. Defaults to False.
            n_workers (int, optional): If `parallelize = True` you need to assign the number
                of workers to prarallelize over. Defaults to 4.
            seed_sequence (int, optional): If `parallelize = True` you may choose the seed sequence
                handed down to every parallel process. Defaults to 1234.
            chunksize (int | None): If `parallelize = True` you may choose the
                chunksize for the parallelization. If `None`, it will be set to
                `runs / n_workers` or 1, whichever is larger. Defaults to None.
            external_dfs (list[pd.DataFrame | np.ndarray] | None, optional):
                If you want to use external dataframes for the benchmark runs, you can pass a list
                of dataframes or numpy arrays here. The length of the list must match the number of
                runs. If `None`, the `prod_obj` will be used to sample data.
                Defaults to None.
            between_and_within_results (bool, optional): If `True`, the benchmark will also
                return the within and between metrics for the `prod_obj` if it is a
                `ProductionLineGraph`. If `False`, only the metrics for the overall graph will be
                returned. Defaults to False.
        """
        self.num_runs = runs
        self.prod_object = prod_obj
        self.n_select = n_select
        if isinstance(prod_obj, ProductionLineGraph):
            self.last_call = f"full_line_{harmonize_via}"
        elif isinstance(prod_obj, ProcessCell):
            self.last_call = f"{prod_obj.name}_{harmonize_via}"
        varsort = []
        metrics = defaultdict(partial(defaultdict, list))
        within_between_metrics = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

        if parallelize:
            seed_seq = np.random.SeedSequence(seed_sequence)
            child_seed = seed_seq.spawn(runs)
            if not chunksize:
                chunksize = int(np.max([np.floor(runs / n_workers), 1]))

            with ProcessPoolExecutor(max_workers=n_workers) as parallelizer:
                for result in parallelizer.map(
                    BenchMarker.single_run,
                    range(runs),
                    repeat(prod_obj, times=runs),
                    repeat(self.algorithms, times=runs),
                    repeat(child_seed, times=runs),
                    chunksize=chunksize,
                ):
                    for alg_name in self.algorithms.keys():
                        for metric_name in result["Benchmarks"][alg_name].keys():
                            metrics[alg_name][metric_name].extend(
                                result["Benchmarks"][alg_name][metric_name]
                            )
                    varsort.append(result["Varsortability"])
        else:
            for run in range(runs):
                if external_dfs is not None and not len(external_dfs) == runs:
                    raise AssertionError("Dataframe list must be of length runs")

                if external_dfs is not None:
                    run_result = BenchMarker.single_run(
                        new_seed=None,
                        prod_obj=prod_obj,
                        algorithms=self.algorithms,
                        harmonize_via=harmonize_via,
                        n_select=n_select,
                        size_threshold=size_threshold,
                        external_df=external_dfs[run],
                        between_and_within_results=between_and_within_results,
                    )
                else:
                    run_result = BenchMarker.single_run(
                        new_seed=None,
                        prod_obj=prod_obj,
                        algorithms=self.algorithms,
                        harmonize_via=harmonize_via,
                        n_select=n_select,
                        size_threshold=size_threshold,
                        between_and_within_results=between_and_within_results,
                    )

                for alg_name in self.algorithms.keys():
                    for metric_name in run_result["Benchmarks"][alg_name].keys():
                        metrics[alg_name][metric_name].extend(
                            run_result["Benchmarks"][alg_name][metric_name]
                        )

                if between_and_within_results:
                    for alg_name in self.algorithms.keys():
                        for metric_name in run_result["Within_between_metrics"][alg_name].keys():
                            for which_one in ["within", "between"]:
                                within_between_metrics[alg_name][metric_name][which_one].extend(
                                    [
                                        run_result["Within_between_metrics"][alg_name][metric_name][
                                            which_one
                                        ]
                                    ]
                                )

                varsort.append(run_result["Varsortability"])

        empty_metrics = DAGmetrics(
            truth=prod_obj.ground_truth.values,
            est=self._emptygraph(size=prod_obj.num_nodes),
        )
        empty_shd = empty_metrics._shd()

        if between_and_within_results:
            self.collect_results[self.last_call] = {
                "Varsortability": varsort,
                "Benchmarks": metrics,
                "Emptygraph_shd": empty_shd,
                "Within_between_metrics": within_between_metrics,
            }
        else:
            self.collect_results[self.last_call] = {
                "Varsortability": varsort,
                "Benchmarks": metrics,
                "Emptygraph_shd": empty_shd,
            }

    @staticmethod
    def single_run(
        new_seed: int | None,
        prod_obj: ProductionLineGraph,
        algorithms: dict[str, Any],
        child_seed: None | np.random.SeedSequence = None,
        harmonize_via: None | str = "cpdag_transform",
        n_select: int = 500,
        size_threshold: int = 50,
        external_df: pd.DataFrame | np.ndarray | None = None,
        between_and_within_results: bool = False,
    ):
        """Single benchmark run.

        Args:
            new_seed (int | None): _description_
            prod_obj (ProductionLineGraph): _description_
            algorithms (dict[str, Any]): _description_
            child_seed (None | np.random.SeedSequence, optional): _description_. Defaults to None.
            harmonize_via (None | str, optional): _description_. Defaults to "cpdag_transform".
            n_select (int, optional): _description_. Defaults to 500.
            size_threshold (int, optional): _description_. Defaults to 50.
            external_df (pd.DataFrame | np.ndarray | None, optional):
                _description_. Defaults to None.
            between_and_within_results (bool, optional): _description_. Defaults to False.

        Raises:
            AssertionError: _description_
            AssertionError: _description_
            TypeError: _description_
            AssertionError: _description_

        Returns:
            _type_: _description_
        """
        metrics = defaultdict(partial(defaultdict, list))
        within_between_metrics = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
        if child_seed is not None:
            num_seeds = 10
            child_seeds = child_seed.spawn(num_seeds)
            if new_seed is None:
                new_seed = 0  # Or some default
                prod_obj.random_state = np.random.default_rng(child_seeds[new_seed])

        if external_df is not None:
            if isinstance(external_df, pd.DataFrame):
                df = external_df
            elif isinstance(external_df, np.ndarray):
                df = pd.DataFrame(external_df, columns=prod_obj.nodes)
        else:
            df = prod_obj.sample_from_drf(size=n_select)
        vsb = BenchMarker.varsortability(data=df, ground_truth=prod_obj.ground_truth)
        df = df / df.std()
        print(df.values.sum())
        for alg_name, alg in algorithms.items():
            result = alg(data=df)
            ground_truth = prod_obj.ground_truth
            if harmonize_via == "cpdag_transform":
                if isinstance(result, pd.DataFrame):
                    result = dag2cpdag(nx.from_pandas_adjacency(result, create_using=nx.DiGraph))
                ground_truth = dag2cpdag(prod_obj.graph)

                if not isinstance(result, PDAG) and not isinstance(ground_truth, PDAG):
                    raise AssertionError("something went wrong in CPDAG transformation")

                result = result.adjacency_matrix
                ground_truth = ground_truth.adjacency_matrix
            elif harmonize_via == "best_dag_shd":
                if isinstance(result, PDAG):
                    if result.nnodes <= size_threshold:
                        get_all_dags = result.to_allDAGs()
                        if len(get_all_dags) == 1:
                            result = nx.to_pandas_adjacency(get_all_dags[0])
                        else:
                            all_shds = []
                            for dag in get_all_dags:
                                dag_metrics = DAGmetrics(truth=prod_obj.graph, est=dag)
                                all_shds.append(dag_metrics._shd())

                            absolute_distance_to_mean = np.abs(
                                np.array(all_shds) - np.mean(all_shds)
                            )
                            random_index_choice = np.random.choice(
                                np.flatnonzero(
                                    absolute_distance_to_mean == np.min(absolute_distance_to_mean)
                                )
                            )
                            chosen_dag = get_all_dags[random_index_choice]
                            result = nx.to_pandas_adjacency(chosen_dag)
                    else:
                        chosen_dag = result.to_random_dag()
                        result = nx.to_pandas_adjacency(chosen_dag)

                if not isinstance(result, pd.DataFrame) and not isinstance(
                    ground_truth, pd.DataFrame
                ):
                    raise AssertionError("something went wrong in the best DAG selection")
            elif type(ground_truth) is not type(result):
                raise TypeError("ground truth and results need to have the same instance.")

            get_metrics = DAGmetrics(truth=ground_truth, est=result)
            my_metrics = get_metrics.collect_metrics()
            if harmonize_via == "best_dag_shd":
                target_dag = prod_obj.graph
                result_dag = nx.from_pandas_adjacency(df=result, create_using=nx.DiGraph)
                my_metrics["sid"] = int(SID(target=target_dag, pred=result_dag))
            if harmonize_via == "cpdag_transform":
                my_metrics["shd"] = get_metrics._shd(count_anticausal_twice=False)

            if between_and_within_results and harmonize_via == "best_dag_shd":
                if isinstance(prod_obj, ProcessCell):
                    raise AssertionError("between and within results only available for PLG")
                within_metrics = prod_obj.within_adjacency
                between_metrics = prod_obj.between_adjacency
                pline_mapper = {}
                for name, cell_graph in prod_obj.cells.items():
                    pline_mapper[name] = cell_graph.nodes
                result_plg = ProductionLineGraph.from_nx(
                    g=nx.from_pandas_adjacency(result, create_using=nx.DiGraph),
                    cell_mapper=pline_mapper,
                )

                get_within_metrics = DAGmetrics(
                    truth=within_metrics, est=result_plg.within_adjacency
                )
                within_metrics = get_within_metrics.collect_metrics()

                get_between_metrics = DAGmetrics(
                    truth=between_metrics, est=result_plg.between_adjacency
                )
                between_metrics = get_between_metrics.collect_metrics()

                # make dict
                w_b_dict = {"within": within_metrics, "between": between_metrics}

                for metric_name in ["precision", "recall"]:
                    for which_one, dct in w_b_dict.items():
                        within_between_metrics[alg_name][metric_name][which_one].append(
                            dct[metric_name]
                        )

            for metric_name, _ in my_metrics.items():
                metrics[alg_name][metric_name].append(my_metrics[metric_name])

        if between_and_within_results:
            return {
                "Varsortability": vsb,
                "Benchmarks": metrics,
                "Within_between_metrics": within_between_metrics,
            }
        else:
            return {"Varsortability": vsb, "Benchmarks": metrics}


def causallearn2amat(causal_learn_graph: np.ndarray) -> np.ndarray:
    """Causallearn object helper function.

    Args:
        causal_learn_graph (np.ndarray): causal lean output graph

    Returns:
        np.ndarray: adjacency matrix.
    """
    amat = np.zeros(causal_learn_graph.shape)
    for col in range(causal_learn_graph.shape[1]):
        for row in range(causal_learn_graph.shape[0]):
            if causal_learn_graph[row, col] == -1 and causal_learn_graph[col, row] == 1:
                amat[row, col] = 1
            if causal_learn_graph[row, col] == -1 and causal_learn_graph[col, row] == -1:
                amat[row, col] = amat[col, row] = 1
    return amat


def to_dag_amat(cpdag_amat: pd.DataFrame) -> pd.DataFrame:
    """Turns PDAG into random member of the corresponding Markov equivalence class.

    Args:
        cpdag_amat (pd.DataFrame): PDAG representing the MEC

    Returns:
        pd.DataFrame: DAG as member of MEC.
    """
    pdag = PDAG.from_pandas_adjacency(cpdag_amat)
    chosen_dag = pdag.to_dag()
    if not nx.is_directed_acyclic_graph(chosen_dag):
        raise TypeError("Graph provided is not a DAG!")

    return nx.to_pandas_adjacency(chosen_dag)
