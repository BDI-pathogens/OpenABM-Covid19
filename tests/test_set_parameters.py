import pytest
import pandas as pd
from COVID19.model import Parameters, Model


class TestSetObjects:
    def test_set_params_classic(self):
        all_params = pd.read_csv("tests/data/baseline_parameters.csv")
        p = Parameters(read_param_file=True, input_households="tests/data/baseline_household_demographics.csv", input_param_file="tests/data/baseline_parameters.csv",param_line_number=1)
        for key in all_params:
            value = all_params[key][0]
            assert p.get_param(key) == value, f"{key} was not set properly"

    def test_set_params_from_python(self):
        all_params = pd.read_csv("tests/data/baseline_parameters.csv")
        p = Parameters(read_param_file=False, input_households="tests/data/baseline_household_demographics.csv")
        for key in all_params:
            value = all_params[key][0]
            try:
                p.set_param(key, float(value))
            except TypeError:
                p.set_param(key, int(value))
            print(f"set {key}")
            assert p.get_param(key) == value, f"{key} was not set properly"
            p._read_household_demographics()
        print("finished setting params")

    def test_run_model_read_prama_file_false(self):
        all_params = pd.read_csv("tests/data/baseline_parameters.csv")
        p = Parameters(
            read_param_file=False,
            input_households="tests/data/baseline_household_demographics.csv",
        )
        for key in all_params:
            value = all_params[key][0]
            try:
                p.set_param(key, float(value))
            except TypeError:
                p.set_param(key, int(value))
        m = Model(p)
        m.one_time_step()
        assert m.one_time_step_results()["time"] == 1

    def test_run_model_read_prama_file_baseline_df(self):
        all_params = pd.read_csv("tests/data/baseline_parameters.csv")
        household_df = pd.read_csv("tests/data/baseline_household_demographics.csv")
        p = Parameters(
            read_param_file=False,
            input_households=household_df,
        )
        for key in all_params:
            value = all_params[key][0]
            try:
                p.set_param(key, float(value))
            except TypeError:
                p.set_param(key, int(value))
        m = Model(p)
        m.one_time_step()
        assert m.one_time_step_results()["time"] == 1
