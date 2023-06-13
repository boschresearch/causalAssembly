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
import json
import os

from utils import run_benchmark

from causalAssembly.drf_fitting import fit_drf
from causalAssembly.models_dag import ProductionLineGraph

seed = 2023
n_select = 500

path_to_benchmarks = os.path.join(os.path.dirname(__file__), "benchmarks.json")

if __name__ == "__main__":
    df_line_data = ProductionLineGraph.get_data()

    # take subsample for demonstration purposes
    df_line_data = df_line_data.sample(
        n_select, random_state=seed, replace=False
    )

    # load in ground truth
    assembly_line = ProductionLineGraph.get_ground_truth()

    assembly_line.drf = fit_drf(prod_object=assembly_line, data=df_line_data)

    for cell in assembly_line.cell_order:
        assembly_line.cells[cell].drf = fit_drf(
            prod_object=assembly_line.cells[cell], data=df_line_data
        )

    benchmark_result_dict = {}
    generator_list = [assembly_line]
    generator_list.extend(
        [assembly_line.cells[cell] for cell in assembly_line.cell_order]
    )
    reps = [100] * 6
    metrics = ["shd", "precision", "recall", "f1"]
    naming = ["Full_Line"]
    naming.extend(assembly_line.cell_order)

    for idx, generator in enumerate(generator_list):
        # big loop over data_generators here and then plot
        benchmark_run = run_benchmark(
            data_generator=generator,
            algorithms=["pc", "lingam", "notears", "grandag", "snr"],
            runs=reps[idx],  # reduce for testing
            return_varsortability=False,
            standardize_before_run=True,
        )

        benchmarks = {}
        for metric in metrics:
            benchmarks_of_metric = {}
            for alg, result_df in benchmark_run.items():
                benchmarks_of_metric[alg] = result_df[metric]

            benchmarks[metric] = benchmarks_of_metric

        benchmark_result_dict[naming[idx]] = benchmarks

    print("Benchmarks completed. I'll dump them to file")

    with open(path_to_benchmarks, "w") as outfile:
        json.dump(benchmark_result_dict, outfile)
