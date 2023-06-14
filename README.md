# causalAssembly

[![License: AGPL v3](https://img.shields.io/badge/License-AGPL_v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)

This repo provides details regarding a causal discovery benchmark data tool based on complex production data.

## Authors
* [Konstantin Goebler (TUM, CC/MFD2)](mailto:konstantin.goebler@de.bosch.com)
* [Steffen Sonntag (CR/APT4)](mailto:steffen.sonntag@de.bosch.com)

**Maintainer*:* [Martin Roth (CC/MFD2)](mailto:martin.roth2@de.bosch.com)

## Table of contents

* [How to install](#installing)
* [How to use](#using)
* [How to test](#testing)
* [How to contribute](#contributing)

## <a name="installing">How to install</a>

The package can be installed as follows

    pip install git+https://github.com/boschresearch/causalAssembly.git

## <a name="using">How to use</a>

This is how causalAssembly's functionality may be used. Be sure to read the [documentation](https://boschresearch.github.io/causalAssembly/) for more in-depth details and usages.

In case you want to train a distributional random forests yourself,
you need an R installation as well as the corresponding [drf](https://cran.r-project.org/web/packages/drf/index.html) R package.
Sampling has first been proposed in [[2]](#2).

*Note*: For Windows users the python package [rpy2](https://github.com/rpy2/rpy2) might cause issues.
        Please consult their [issue tracker](https://github.com/rpy2/rpy2/issues) on GitHub.

In order to fit DRFs and sample data, consider the following example:

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

The `ProductionLineGraph` class can also be used to generate completely random DAGs
that follow an assembly line logic. Consider the following example:

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

causalAssembly also allows to create functional causal model (FCM) and sample after specifying noise distributions.
For creating and sampling from handcrafted FCMs, a simple example would be:

```python

from causalAssembly.models_fcm import HandCrafted_FCM
from sympy import symbols, Eq
from sympy.stats import Uniform

x,y,z = symbols('x,y,z')

eq_x = Eq(x, Uniform('x', left=-1, right=1))
eq_y = Eq(y, 2*x**2 + 3)
eq_z = Eq(z, 9*y*x)

eq_list = [eq_x, eq_y, eq_z]

example_fcm = HandCrafted_FCM(name='example_fcm', seed= 2023)
example_fcm.input_fcm(eq_list)

print(example_fcm.graph.edges())

example_df = example_fcm.draw(size= 10, add_noise= True, snr= 2/3)
example_df.head()

```
### References

<a id="1">[1]</a>
Ćevid, D., Michel, L., Näf, J., Bühlmann, P., & Meinshausen, N. (2022). Distributional Random Forests: Heterogeneity Adjustment and Multivariate Distributional Regression. Journal of Machine Learning Research, 23(333), 1-79.

<a id="2">[2]</a>
Gamella, J.L, Taeb, A., Heinze-Deml, C., & Bühlmann, P. (2022). Characterization and greedy learning of Gaussian structural causal models under unknown noise interventions. arXiv preprint arXiv:2211.14897, 2022.


## <a name="testing">How to test</a>

In general we use pytest and the test suite can be executed locally via

    python -m pytest

## <a name="contributing">How to contribute?</a>

Please feel free to contact one of the authors in case you wish to contribute. 

## <a name="3rd-party-licenses">Third-Party Licenses</a>

### Runtime dependencies

| Name | License | Type |
|------|---------|------|
| [numpy](https://numpy.org/) | [BSD-3-Clause License](https://github.com/numpy/numpy/blob/master/LICENSE.txt) | Dependency |
| [scipy](https://scipy.org/) | [BSD-3-Clause License](https://github.com/scipy/scipy/blob/main/LICENSE.txt) | Dependency |
| [pandas](https://pandas.pydata.org/)|[BSD 3-Clause License](https://github.com/pandas-dev/pandas/blob/master/LICENSE)| Dependency |
| [networkx](https://pypi.org/project/networkx/)| [BSD-3-Clause License](https://github.com/networkx/networkx/blob/main/LICENSE.txt) | Dependency |
| [matplotlib](https://github.com/matplotlib/matplotlib)|[Other](https://github.com/matplotlib/matplotlib/tree/main/LICENSE)| Dependency |
| [sympy](https://github.com/sympy/sympy) | [BSD-3-Clause License](https://github.com/sympy/sympy/blob/master/LICENSE) | Dependency |
| [pydantic](https://github.com/pydantic/pydantic) | [MIT License](https://github.com/pydantic/pydantic/blob/main/LICENSE) | Dependency |
| [distinctipy](https://github.com/alan-turing-institute/distinctipy) | [MIT License](https://github.com/alan-turing-institute/distinctipy/blob/main/LICENSE) | Dependency |
| [rpy2](https://github.com/rpy2/rpy2) | [GNU General Public License v2.0](https://github.com/rpy2/rpy2/blob/master/LICENSE) | Dependency |
### Development dependency

| Name | License | Type |
|------|---------|------|
| [mike](https://github.com/jimporter/mike)|[BSD-3-Clause License](https://github.com/jimporter/mike/blob/master/LICENSE)| Dependency |
| [mkdocs](https://github.com/mkdocs/mkdocs)|[BSD-2-Clause License](https://github.com/mkdocs/mkdocs/blob/master/LICENSE)| Dependency |
| [mkdocs-material](https://github.com/squidfunk/mkdocs-material)|[MIT License](https://github.com/squidfunk/mkdocs-material/blob/master/LICENSE)| Dependency |
| [mkdocstrings[python]](https://github.com/mkdocstrings/python)|[ISC License](https://github.com/mkdocstrings/python/blob/master/LICENSE)| Dependency |
| [ruff](https://github.com/charliermarsh/ruff) | [MIT License](https://github.com/charliermarsh/ruff/blob/main/LICENSE) | Dependency |
| [pytest](https://docs.pytest.org)| [MIT License](https://docs.pytest.org/en/latest/license.html) | Dependency|
| [pip-tools](https://github.com/jazzband/pip-tools) | [BSD 3-Clause License](https://github.com/jazzband/pip-tools/blob/master/LICENSE) | Dependency |
