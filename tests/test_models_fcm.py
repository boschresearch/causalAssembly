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

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal
from sympy import Eq, Symbol, symbols
from sympy.stats import Gamma, Normal, Uniform

from causalAssembly.models_fcm import FCM


class TestFCM:
    @pytest.fixture(scope="class")
    def example_fcm(self):
        x, y, z = symbols("x,y,z")

        eq_x = Eq(x, Uniform("error", left=-1, right=1))
        eq_y = Eq(y, 2 * x)
        eq_z = Eq(z, x + y)

        eq_list = [eq_x, eq_y, eq_z]

        example_fcm = FCM(name="example_fcm", seed=2023)
        example_fcm.input_fcm(eq_list)

        return example_fcm

    @pytest.fixture(scope="class")
    def medium_example_fcm(self) -> FCM:
        v, x, y, z = symbols("v,x,y,z")

        eq_x = Eq(x, Normal("error", 0, 1))
        eq_v = Eq(v, Gamma("error", 1, 1))
        eq_y = Eq(y, 2 * x**2 - 7 * v + Normal("error", 0, 0.2))
        eq_z = Eq(z, 9 * y * x * Gamma("noise", 0.5, 1))

        eq_list = [eq_v, eq_x, eq_y, eq_z]

        test_fcm = FCM(name="testing", seed=2023)
        test_fcm.input_fcm(eq_list)
        return test_fcm

    def test_instance_is_created(self):
        h = FCM(name="mymodel", seed=1234)
        assert isinstance(h, FCM)

    def test_input_fcm_works(self, example_fcm):
        # Act
        x, y = symbols("x,y")

        # Assert
        assert len(example_fcm.source_nodes) == 1
        assert example_fcm.num_nodes == 3
        assert example_fcm.num_edges == 3
        assert (x, y) in example_fcm.edges

    def test_empty_graph_works(self):
        # Arrange
        x, y, z = symbols("x,y,z")

        eq_x = Eq(x, Normal("exogenous", 0, 2))
        eq_y = Eq(y, 3)
        eq_z = Eq(z, Gamma("exogenous", 1, 1))

        # Act
        test_fcm = FCM()
        test_fcm.input_fcm([eq_x, eq_y, eq_z])

        df = test_fcm.sample(size=5)

        # Assert

        assert test_fcm.num_nodes == 3
        assert test_fcm.num_edges == 0
        assert df.shape == (5, 3)

    def test_draw_without_noise_works(self, example_fcm):
        # Act
        df = example_fcm.sample(size=10, additive_gaussian_noise=False)

        # Assert
        assert len(df) == 10

    def test_draw_with_noise_works(self, example_fcm):
        # Act
        df_without_noise = example_fcm.sample(size=10, additive_gaussian_noise=False)
        df_with_noise = example_fcm.sample(size=10, additive_gaussian_noise=True)

        # Assert
        # that the columns for the two drawn distributions are not equal / all close
        for col in df_without_noise.columns:
            assert not np.allclose(df_without_noise[col], df_with_noise[col])

    def test_draw_from_dataframe(self, example_fcm: FCM):
        # Arrange
        source_df = pd.DataFrame()
        source_df["x"] = [0, 1.0, 10.0]

        # Act
        sampled_df_from_source = example_fcm.sample(
            size=3, source_df=source_df, additive_gaussian_noise=False
        )

        sampled_df = example_fcm.sample(size=3, additive_gaussian_noise=False)

        # Assert
        assert not sampled_df_from_source.equals(sampled_df)
        assert sampled_df_from_source["y"].equals(2 * sampled_df_from_source["x"])
        assert sampled_df_from_source["z"].equals(
            sampled_df_from_source["x"] + sampled_df_from_source["y"]
        )

    def test_draw_from_wrong_dataframe_raises_assertionerror(self, example_fcm):
        source_df = pd.DataFrame()
        source_df["AAA"] = [0, 1.0, 10.0]

        # Assert
        with pytest.raises(AssertionError):
            example_fcm.sample(size=3, source_df=source_df, additive_gaussian_noise=False)

    def test_specify_individual_noise(self):
        # Arrange
        x, y, z = symbols("x,y,z")

        eq_x = Eq(x, Normal("error", 0, 1))
        eq_y = Eq(y, 2 * x**2 + Normal("error", 0, 0.2))
        eq_z = Eq(z, 9 * y * x * Gamma("noise", 0.5, 1))

        eq_list = [eq_x, eq_y, eq_z]

        test_fcm = FCM(name="testing", seed=2023)
        test_fcm.input_fcm(eq_list)

        # Act

        df = test_fcm.sample(size=10)

        # Assert

        assert test_fcm.num_edges == 3
        assert df.shape == (10, 3)

    def test_order_in_eval_always_correct(self):
        # Arrange
        v, w, x, y, z = symbols("v,w,x,y,z")
        eq_v = Eq(v, Uniform("noise", left=0.2, right=0.8))
        eq_w = Eq(w, Normal("noise", 0, 1))
        eq_x = Eq(x, 27 - v)
        eq_y = Eq(y, 2 * x - 4 * w)
        eq_z = Eq(z, (x + y) / v)

        eq_list = [eq_v, eq_w, eq_x, eq_y, eq_z]

        # Act
        test_fcm = FCM()
        test_fcm.input_fcm(eq_list)
        df = test_fcm.sample(size=10)
        # Assert
        assert all(np.isclose(df["x"], 27 - df["v"]))
        assert all(np.isclose(df["y"], 2 * df["x"] - 4 * df["w"]))
        assert all(np.isclose(df["z"], (df["x"] + df["y"]) / df["v"]))

    def test_select_scale_parameter_via_snr(self):
        # Arrange
        x, y, z = symbols("x,y,z")
        sigma = Symbol("sigma", positive=True)

        eq_x = Eq(x, Normal("error", 0, 1))
        eq_y = Eq(y, 2 * x**2 + Normal("error", 0, sigma))
        eq_z = Eq(z, 9 * y * x * Gamma("noise", 0.5, 1))

        eq_list = [eq_x, eq_y, eq_z]

        test_fcm = FCM(name="testing", seed=2023)
        test_fcm.input_fcm(eq_list)

        # Act

        df = test_fcm.sample(size=10, snr=0.6)

        # Assert

        assert test_fcm.num_edges == 3
        assert df.shape == (10, 3)

    def test_select_scale_parameter_via_snr_gives_error_when_not_additive(self):
        # Arrange
        x, y, z = symbols("x,y,z")
        sigma = Symbol("sigma", positive=True)

        eq_x = Eq(x, Normal("error", 0, 1))
        eq_y = Eq(y, 2 * x**2 + Normal("error", 0, 1))
        eq_z = Eq(z, 9 * y * x * Gamma("noise", 0.5, sigma))

        eq_list = [eq_x, eq_y, eq_z]

        test_fcm = FCM(name="testing", seed=2023)
        test_fcm.input_fcm(eq_list)

        # Assert
        with pytest.raises(ValueError):
            test_fcm.sample(size=10, snr=0.6)

    def test_data_frame_with_fewer_columns_than_source_nodes(self, medium_example_fcm: FCM):
        # Act
        source_df = pd.DataFrame(
            {
                "v": np.random.uniform(low=-0.1, high=0.71, size=10),
            },
            columns=["v"],
        )

        df = medium_example_fcm.sample(size=10, snr=0.6, source_df=source_df)
        # Assert
        assert medium_example_fcm.num_edges == 4
        assert df.shape == (10, 4)

    def test_data_frame_too_few_rows(self, medium_example_fcm: FCM):
        # Act
        source_df = pd.DataFrame(
            {
                "v": np.random.uniform(low=-0.1, high=0.71, size=9),
            },
            columns=["v"],
        )

        # Assert
        with pytest.raises(ValueError):
            medium_example_fcm.sample(size=10, snr=0.6, source_df=source_df)

    def test_polynomial_equation_works(self):
        # Arrange
        x, y = symbols("x,y")

        eq_x = Eq(x, Normal("error", 0, 1))
        eq_y = Eq(y, x**2 - 2 * x + 5)

        test_fcm = FCM()
        test_fcm.input_fcm([eq_x, eq_y])
        # Act
        df = test_fcm.sample(size=5)
        assert all(df["y"] == df["x"] ** 2 - 2 * df["x"] + 5)

    def test_display_functions_works(self):
        # Arrange
        v, x, y, z = symbols("v,x,y,z")

        eq_x = Eq(x, Normal("error", 0, 1))
        eq_v = Eq(v, Gamma("error", 1, 1))
        eq_y = Eq(y, 2 * x**2 - 7 * v + Normal("error", 0, 0.2))
        eq_z = Eq(z, 9 * y * x * Gamma("noise", 0.5, 1))

        eq_list = [eq_v, eq_x, eq_y, eq_z]

        test_fcm = FCM(name="testing", seed=2023)
        test_fcm.input_fcm(eq_list)
        # Act
        functions_dict = test_fcm.display_functions()
        # Assert
        assert isinstance(functions_dict, dict)
        assert len(functions_dict) == 4
        assert str(functions_dict[x]) == str(functions_dict[v]) == "error"
        assert functions_dict[y] == eq_y.rhs
        assert functions_dict[z] == eq_z.rhs

    def test_show_individual_function(self):
        # Arrange
        v, x, y, z = symbols("v,x,y,z")

        eq_x = Eq(x, Normal("error", 0, 1))
        eq_v = Eq(v, Gamma("error", 1, 1))
        eq_y = Eq(y, 2 * x**2 - 7 * v + Normal("error", 0, 0.2))
        eq_z = Eq(z, 9 * y * x * Gamma("noise", 0.5, 1))

        eq_list = [eq_v, eq_x, eq_y, eq_z]

        test_fcm = FCM(name="testing", seed=2023)
        test_fcm.input_fcm(eq_list)
        # Assert
        assert isinstance(test_fcm.function_of(node=x), dict)
        with pytest.raises(AssertionError):
            test_fcm.function_of(node="x")
        with pytest.raises(AssertionError):
            test_fcm.function_of(node="m")
        assert test_fcm.function_of(node=y) == {y: eq_y.rhs}

    def test_single_hard_intervention(self):
        # Arrange
        x, y, z = symbols("x,y,z")

        eq_x = Eq(x, Normal("error", 0, 1))
        eq_y = Eq(y, 2 * x**2 + Normal("error", 0, 1))
        eq_z = Eq(z, 9 * y * x + Gamma("noise", 0.5, 1))

        eq_list = [eq_x, eq_y, eq_z]

        test_fcm = FCM()
        test_fcm.input_fcm(eq_list)

        # Act
        test_fcm.intervene_on({y: 2.5})

        test_df = test_fcm.interventional_sample(size=5, which_intervention=0)

        # Assert
        assert len(test_fcm.interventions) == 1
        assert test_fcm.interventions[0] == "do([y])"
        assert len(test_fcm.mutilated_dags) == 1
        assert len(test_fcm.mutilated_dags["do([y])"].edges()) == test_fcm.num_edges - 1
        assert np.isclose(test_df["y"].mean(), 2.5)

    def test_single_soft_intervention(self):
        # Arrange
        x, y, z = symbols("x,y,z")

        eq_x = Eq(x, Normal("error", 0, 1))
        eq_y = Eq(y, 2 * x**2 + Normal("error", 0, 1))
        eq_z = Eq(z, 9 * y * x + Gamma("noise", 0.5, 1))

        eq_list = [eq_x, eq_y, eq_z]

        test_fcm = FCM()
        test_fcm.input_fcm(eq_list)

        # Act
        test_fcm.intervene_on({z: Uniform("noise", left=-0.3, right=0.3)})

        test_df = test_fcm.interventional_sample(size=5, which_intervention=0)

        # Assert
        assert len(test_fcm.interventions) == 1
        assert test_fcm.interventions[0] == "do([z])"
        assert len(test_fcm.mutilated_dags) == 1
        assert len(test_fcm.mutilated_dags["do([z])"].edges()) == test_fcm.num_edges - 2
        assert -0.3 < test_df["z"].min() and test_df["z"].min() < 0.3

    def test_multiple_interventions_at_once(self):
        # Arrange
        v, x, y, z = symbols("v,x,y,z")

        eq_x = Eq(x, Uniform("error", left=0.3, right=0.8))
        eq_v = Eq(v, -7 * x + Uniform("noise", left=-0.3, right=0.3))
        eq_y = Eq(y, 2 * x**2 - 7 * v + Normal("error", 0, 0.2))
        eq_z = Eq(z, 9 * y * x * Gamma("noise", 0.5, 1))

        eq_list = [eq_v, eq_x, eq_y, eq_z]

        test_fcm = FCM()
        test_fcm.input_fcm(eq_list)

        # Act
        test_fcm.intervene_on({v: 5, z: Normal("error", 0, 1)})
        test_df = test_fcm.interventional_sample(size=5, which_intervention=0)

        # Assert
        assert len(test_fcm.interventions) == 1
        assert test_fcm.interventions[0] == "do([v, z])"
        assert len(test_fcm.mutilated_dags) == 1
        assert len(test_fcm.mutilated_dags["do([v, z])"].edges()) == test_fcm.num_edges - 3
        assert np.isclose(test_df["v"].mean(), 5)

    def test_multiple_interventions_sequentually(self):
        # Arrange
        v, x, y, z = symbols("v,x,y,z")

        eq_x = Eq(x, Uniform("error", left=0.3, right=0.8))
        eq_v = Eq(v, -7 * x + Uniform("noise", left=-0.3, right=0.3))
        eq_y = Eq(y, 2 * x**2 - 7 * v + Normal("error", 0, 0.2))
        eq_z = Eq(z, 9 * y * x * Gamma("noise", 0.5, 1))

        eq_list = [eq_v, eq_x, eq_y, eq_z]

        test_fcm = FCM()
        test_fcm.input_fcm(eq_list)

        # Act
        test_fcm.intervene_on({v: 5})
        test_df_1 = test_fcm.interventional_sample(size=5, which_intervention=0)

        test_fcm.intervene_on({z: Normal("error", 0, 1)})
        test_fcm.interventional_sample(size=5, which_intervention=1)

        # Assert
        assert len(test_fcm.interventions) == 2
        assert len(test_fcm.mutilated_dags) == 2
        assert len(test_fcm.mutilated_dags["do([v])"].edges()) == test_fcm.num_edges - 1
        assert len(test_fcm.mutilated_dags["do([z])"].edges()) == test_fcm.num_edges - 2
        assert np.isclose(test_df_1["v"].mean(), 5)

    def test_reproducability(self):
        # Arrange
        v, x, y, z, sigma = symbols("v,x,y,z,sigma")

        eq_x = Eq(x, Normal("error", 0, 1))
        eq_v = Eq(v, Gamma("error", 1, 1))
        eq_y = Eq(y, 2 * x**2 - 7 * v)
        eq_z = Eq(z, 9 * y * x)

        eq_list = [eq_x, eq_v, eq_y, eq_z]

        # Act
        test_fcm = FCM(name="testing", seed=2023)
        test_fcm.input_fcm(eq_list)

        source_df = pd.DataFrame(
            {
                "v": np.array([1, 2, 3, 4, 5, 6, 7, 8, 9]),
            },
            columns=["v"],
        )
        df_a = test_fcm.sample(size=5, snr=2 / 3, additive_gaussian_noise=True, source_df=source_df)

        test_fcm = FCM(name="testing", seed=2023)
        test_fcm.input_fcm(eq_list)

        df_b = test_fcm.sample(size=5, snr=2 / 3, additive_gaussian_noise=True, source_df=source_df)

        # Very slight chance that this could theoretically be equal
        df_c = test_fcm.sample(size=5, snr=2 / 3, additive_gaussian_noise=True, source_df=source_df)

        # Assert
        assert_frame_equal(df_a, df_b)
        assert not df_b.equals(df_c)
