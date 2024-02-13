# causalAssembly

[![License: AGPL v3](https://img.shields.io/badge/License-AGPL_v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)
[![Code style: ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/format.json)](https://github.com/astral-sh/ruff)


This repo provides details regarding $\texttt{causalAssembly}$, a causal discovery benchmark data tool based on complex production data.

## Authors
* [Konstantin Goebler](mailto:konstantin.goebler@de.bosch.com)
* [Steffen Sonntag](mailto:steffen.sonntag@de.bosch.com)

**Maintainer*:* [Konstantin Goebler](mailto:konstantin.goebler@de.bosch.com)

## Table of contents

* [How to install](#installing)
* [How to use](#using)
* [How to test](#testing)
* [How to contribute](#contributing)

## <a name="installing">How to install</a>

The package can be installed as follows

    pip install causalAssembly

[comment]: <> (git+https://github.com/boschresearch/causalAssembly.git)

## <a name="using">How to use</a>

This is how $\texttt{causalAssembly}$'s functionality may be used. Be sure to read the [documentation](https://boschresearch.github.io/causalAssembly/) for more in-depth details regarding available functions and classes.

In case you want to train a distributional random forests yourself (see [how to semisynthetsize](#how-to-semisynthesize)),
you need an R installation as well as the corresponding [drf](https://cran.r-project.org/web/packages/drf/index.html) R package.
Sampling has first been proposed in [[2]](#2).

*Note*: For Windows users the python package [rpy2](https://github.com/rpy2/rpy2) might cause issues.
        Please consult their [issue tracker](https://github.com/rpy2/rpy2/issues) on GitHub.

In order to sample semisynthetic data from $\texttt{causalAssembly}$, consider the following example:

```python
import pandas as pd

from causalAssembly.models_dag import ProductionLineGraph
from causalAssembly.drf_fitting import fit_drf

seed = 2023
n_select = 500

assembly_line_data = ProductionLineGraph.get_data()

# take subsample for demonstration purposes
assembly_line_data = assembly_line_data.sample(
    n_select, random_state=seed, replace=False
)

# load in ground truth
assembly_line = ProductionLineGraph.get_ground_truth()

# fit drf and sample for entire line
assembly_line.drf = fit_drf(assembly_line, data=assembly_line_data)
assembly_line_sample = assembly_line.sample_from_drf(size=n_select)

# fit drf and sample for station3
assembly_line.Station3.drf = fit_drf(assembly_line.Station3, data=assembly_line_data)
station3_sample = assembly_line.Station3.sample_from_drf(size=n_select)

```

### <a name="how-to-semisynthesize">How to semisynthesize</a>

In order to generate semisynthetic data for data sources outside the manufacturing
context, the class `DAG` may be used. We showcase all necessary steps in the example below using the well-known Sachs [[3](#3)] dataset.
Note, that the `cdt` package is only needed to get easy access to data and corresponding ground truth.

```python
import networkx as nx
from cdt.data import load_dataset

from causalAssembly.dag import DAG
from causalAssembly.drf_fitting import fit_drf

# load data set and available ground truth
s_data, s_graph = load_dataset("sachs")

# take subset for faster computation
s_data = s_data.sample(100, random_state=42)

print(nx.is_directed_acyclic_graph(s_graph))
cycles = nx.find_cycle(s_graph)
s_graph.remove_edge(*cycles[0])

if nx.is_directed_acyclic_graph(s_graph):
    # convert to DAG instance
    sachs_dag = DAG.from_nx(s_graph)

    # fit DRF to the conditional distributions implied by
    # the factorization over <s_graph>
    sachs_dag.drf = fit_drf(graph=sachs_dag, data=s_data)

    # sample new data from the trained DRFs
    dream_benchmark_data = sachs_dag.sample_from_drf(size=50)
    print(dream_benchmark_data.head())

```

### <a name="how-to-rand">How to generate random production DAGs</a>


The `ProductionLineGraph` class can further be used to generate completely random DAGs that follow an assembly line logic. Consider the following example:

```python

from causalAssembly.models_dag import ProductionLineGraph

example_line = ProductionLineGraph()

example_line.new_cell(name='Station1')
example_line.Station1.add_random_module()
example_line.Station1.add_random_module()

example_line.new_cell(name='Station2')
example_line.Station2.add_random_module(n_nodes=5)

example_line.new_cell(name='Station3', is_eol= True)
example_line.Station3.add_random_module()
example_line.Station3.add_random_module()

example_line.connect_cells(forward_probs= [.1])

example_line.show()

```
### <a name="how-to-fcm">How to generate FCMs</a>


$\texttt{causalAssembly}$ also allows creating structural causal models (SCM) or synonymously functional causal models (FCM). In particular, we employ symbolic programming to allow for a seamless interplay between readability and performance. The `FCM` class is completely general and inherits no production data logic. See the example below for construction and usage.


```python

import numpy as np
import pandas as pd
from sympy import Eq, Symbol, symbols
from sympy.stats import Gamma, Normal, Uniform, Exponential
from causalAssembly.models_fcm import FCM

# declare variables in FCM as symbols
v, w, x, y, z = symbols("v,w,x,y,z")
# declare symbol for the variance of a Gaussian
delta = Symbol("delta", positive=True)

# Set up FCM
# name for the noise terms is required but mainly for readability
# it gets evaluated equation-by-equation. Therefore repeating names is completely fine.
eq_x = Eq(x, Exponential("source_distribution", 0.5))
eq_v = Eq(v, Gamma("source_distribution", 1, 1))
eq_y = Eq(y, 2 * x**2 - 7 * v + Normal("error", 0, delta))
eq_z = Eq(z, 9 * y * x * Gamma("noise", 0.5, 1))
eq_w = Eq(w, 7 * v - z + Uniform("error", left=-0.5, right=0.5))

# Collect in a list
eq_list = [eq_v, eq_w, eq_x, eq_y, eq_z]

# Create instance
test_fcm = FCM()
# Input list of equations this automatically
# induces the DAG etc.
test_fcm.input_fcm(eq_list)

# There is an option to use real data for source node samples
source_df = pd.DataFrame(
    {
        "v": np.random.uniform(low=-0.1, high=0.71, size=10),
    },
    columns=["v"],
)

# Sample from joint distribution

print(test_fcm.sample(size=8, source_df=source_df))
test_fcm.show(header="No Intervention")


# Multiple hard and soft interventions:
test_fcm.intervene_on(nodes_values={z: 2, w: Normal("noise", 3, 1)})

print(test_fcm.interventional_sample(size=8, source_df=source_df))

# Some plotting
test_fcm.show_mutilated_dag()


```

### References

<a id="1">[1]</a>
Ćevid, D., Michel, L., Näf, J., Bühlmann, P., & Meinshausen, N. (2022). Distributional Random Forests: Heterogeneity Adjustment and Multivariate Distributional Regression. Journal of Machine Learning Research, 23(333), 1-79.

<a id="2">[2]</a>
Gamella, J.L, Taeb, A., Heinze-Deml, C., & Bühlmann, P. (2022). Characterization and greedy learning of Gaussian structural causal models under unknown noise interventions. arXiv preprint arXiv:2211.14897, 2022.

<a id="3">[3]</a>
Sachs, K., Perez, O., Pe'er, D., Lauffenburger, D. A., & Nolan, G. P. (2005). Causal protein-signaling networks derived from multiparameter single-cell data. Science, 308(5721), 523-529.

## <a name="testing">How to test</a>

In general we use pytest and the test suite can be executed locally via

    python -m pytest

## <a name="contributing">How to contribute?</a>

Please feel free to contact one of the authors in case you wish to contribute.

## <a name="3rd-party-licenses">Third-Party Licenses</a>

### Runtime dependencies

| Name                                                   | License                                                                             | Type       |
| ------------------------------------------------------ | ----------------------------------------------------------------------------------- | ---------- |
| [numpy](https://numpy.org/)                            | [BSD-3-Clause License](https://github.com/numpy/numpy/blob/master/LICENSE.txt)      | Dependency |
| [scipy](https://scipy.org/)                            | [BSD-3-Clause License](https://github.com/scipy/scipy/blob/main/LICENSE.txt)        | Dependency |
| [pandas](https://pandas.pydata.org/)                   | [BSD 3-Clause License](https://github.com/pandas-dev/pandas/blob/master/LICENSE)    | Dependency |
| [networkx](https://pypi.org/project/networkx/)         | [BSD-3-Clause License](https://github.com/networkx/networkx/blob/main/LICENSE.txt)  | Dependency |
| [matplotlib](https://github.com/matplotlib/matplotlib) | [Other](https://github.com/matplotlib/matplotlib/tree/main/LICENSE)                 | Dependency |
| [sympy](https://github.com/sympy/sympy)                | [BSD-3-Clause License](https://github.com/sympy/sympy/blob/master/LICENSE)          | Dependency |
| [rpy2](https://github.com/rpy2/rpy2)                   | [GNU General Public License v2.0](https://github.com/rpy2/rpy2/blob/master/LICENSE) | Dependency |
### Development dependency

| Name                                                            | License                                                                           | Type       |
| --------------------------------------------------------------- | --------------------------------------------------------------------------------- | ---------- |
| [mike](https://github.com/jimporter/mike)                       | [BSD-3-Clause License](https://github.com/jimporter/mike/blob/master/LICENSE)     | Dependency |
| [mkdocs](https://github.com/mkdocs/mkdocs)                      | [BSD-2-Clause License](https://github.com/mkdocs/mkdocs/blob/master/LICENSE)      | Dependency |
| [mkdocs-material](https://github.com/squidfunk/mkdocs-material) | [MIT License](https://github.com/squidfunk/mkdocs-material/blob/master/LICENSE)   | Dependency |
| [mkdocstrings[python]](https://github.com/mkdocstrings/python)  | [ISC License](https://github.com/mkdocstrings/python/blob/master/LICENSE)         | Dependency |
| [ruff](https://github.com/charliermarsh/ruff)                   | [MIT License](https://github.com/charliermarsh/ruff/blob/main/LICENSE)            | Dependency |
| [pytest](https://docs.pytest.org)                               | [MIT License](https://docs.pytest.org/en/latest/license.html)                     | Dependency |
| [pip-tools](https://github.com/jazzband/pip-tools)              | [BSD 3-Clause License](https://github.com/jazzband/pip-tools/blob/master/LICENSE) | Dependency |
