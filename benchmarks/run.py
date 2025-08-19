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
import json
import logging
import os

from rpy2.robjects.packages import importr
from utils import BenchMarker

from causalAssembly.drf_fitting import fit_drf
from causalAssembly.models_dag import ProductionLineGraph

base_r_package = importr("base")
utils = importr("utils")
utils.install_packages("drf", repos="https://cloud.r-project.org")
utils.install_packages("SID")

logging.getLogger("py4j").setLevel(logging.ERROR)

n_select = 5000
runs = 100

path_to_benchmarks = os.path.join(os.path.dirname(__file__), "benchmarks.json")

if __name__ == "__main__":
    line_data = ProductionLineGraph.get_data()

    # take subsample for demonstration purposes
    df_line_data = line_data.sample(n_select, replace=False)

    # load in ground truth
    assembly_line = ProductionLineGraph.get_ground_truth()
    assembly_line.drf = fit_drf(graph=assembly_line, data=df_line_data)

    for cell in assembly_line.cell_order:
        assembly_line.cells[cell].drf = fit_drf(graph=assembly_line.cells[cell], data=df_line_data)

    benchmarks = BenchMarker()

    benchmarks.include_grandag()
    benchmarks.include_lingam()
    benchmarks.include_pc()
    benchmarks.include_notears()
    benchmarks.include_das()

    benchmarks.run_benchmark(runs=runs, prod_obj=assembly_line, harmonize_via="cpdag_transform")

    benchmarks.run_benchmark(
        runs=runs,
        prod_obj=assembly_line,
        harmonize_via="best_dag_shd",
        between_and_within_results=True,
    )

    for cell in assembly_line.cell_order:
        benchmarks.run_benchmark(
            runs=runs, prod_obj=assembly_line.cells[cell], harmonize_via="cpdag_transform"
        )
        benchmarks.run_benchmark(
            runs=runs, prod_obj=assembly_line.cells[cell], harmonize_via="best_dag_shd"
        )

    print("Benchmarks completed. I'll dump them to file")

    with open(path_to_benchmarks, "w", encoding="utf-8") as outfile:
        json.dump(benchmarks.collect_results, outfile)
