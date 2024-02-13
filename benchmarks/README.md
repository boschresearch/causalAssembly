# Benchmarks

[![License: AGPL v3](https://img.shields.io/badge/License-AGPL_v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)

Follow these instructions in order to reproduce the benchmarks conducted in the paper $\texttt{causalAssembly}$: Generating Realistic Production
Data for Benchmarking Causal Discovery.

## Authors
* [Konstantin Goebler (TUM, VM/MFD2)](mailto:konstantin.goebler@de.bosch.com)

## Table of contents

* [How to install](#installing)
* [How to use](#using)

## <a name="installing">How to install</a>

To install the required dependencies, run the following command:

    pip install -r requirements.txt

## <a name="using">How to use</a>

You may need to adjust the paths to where you stored ground truth and data set. To run benchmarks, execute the file

    python3 run.py

## <a name="3rd-party-licenses">Third-Party Licenses</a>

### Additional dependencies for benchmarks

| Name                                                                         | License                                                                                | Type       |
| ---------------------------------------------------------------------------- | -------------------------------------------------------------------------------------- | ---------- |
| [gCastle ](https://github.com/huawei-noah/trustworthyAI/tree/master/gcastle) | [Apache License 2.0](https://github.com/huawei-noah/trustworthyAI/blob/master/LICENSE) | Dependency |
| [causal-learn](https://github.com/py-why/causal-learn)                       | [MIT License](https://github.com/py-why/causal-learn/blob/main/LICENSE)                | Dependency |
| [lingam](https://github.com/cdt15/lingam)                                    | [MIT License](https://github.com/cdt15/lingam/blob/master/LICENSE)                     | Dependency |
| [varsortability](https://github.com/Scriddie/Varsortability)                 | [MIT License](https://github.com/Scriddie/Varsortability/blob/main/LICENSE)            | Reference  |
| [dodiscover](https://github.com/py-why/dodiscover/tree/main)                 | [MIT License](https://github.com/py-why/dodiscover/blob/main/LICENSE)                  | Dependency |
