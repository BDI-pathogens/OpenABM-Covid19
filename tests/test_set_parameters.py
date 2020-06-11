import pytest
import pandas as pd
from COVID19.model import Parameters, Model, OccupationNetworkEnum  


class TestSetObjects:
    def test_set_params_classic(self):
        all_params = pd.read_csv("tests/data/baseline_parameters.csv")
        p = Parameters(
            read_param_file=True,
            input_households="tests/data/baseline_household_demographics.csv",
            input_param_file="tests/data/baseline_parameters.csv",
            param_line_number=1,
        )
        for key in all_params:
            value = all_params[key][0]
            assert pytest.approx(p.get_param(key), value), f"{key} was not set properly"

    def test_set_params_from_python(self):
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
            assert pytest.approx(p.get_param(key), value), f"{key} was not set properly"
            p._read_household_demographics()

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

    # @pytest.mark.skip(reason="Test fails")
    def test_run_model_read_prama_file_baseline_df(self):
        all_params = pd.read_csv("tests/data/baseline_parameters.csv")
        household_df = pd.read_csv("tests/data/baseline_household_demographics.csv")
        p = Parameters(read_param_file=False, input_households=household_df,)
        for key in all_params:
            value = all_params[key][0]
            try:
                p.set_param(key, float(value))
            except TypeError:
                p.set_param(key, int(value))
        m = Model(p)
        m.one_time_step()
        assert m.one_time_step_results()["time"] == 1

    def test_set_params_occupation_network_by_age(self):
        p = Parameters(
            read_param_file=True,
            input_households="tests/data/baseline_household_demographics.csv",
            input_param_file="tests/data/baseline_parameters.csv",
            param_line_number=1,
        )
        p.set_param("lockdown_occupation_multiplier_primary_network", 100.0)
        p.set_param("lockdown_occupation_multiplier_working_network", 0.2)
        assert p.get_param("lockdown_occupation_multiplier_primary_network") == 100.0
        assert p.get_param("lockdown_occupation_multiplier_working_network") == 0.2

    def test_set_params_occupation_network_by_age_check_used(self):
        p = Parameters(
            read_param_file=True,
            input_households="tests/data/baseline_household_demographics.csv",
            input_param_file="tests/data/baseline_parameters.csv",
            param_line_number=1,
        )
        model = Model(p)
        non_scaled = [
            model.get_param(f"daily_fraction_work_used{age.name}")
            for age in OccupationNetworkEnum
        ]
        model.update_running_params("lockdown_on", 1)
        scaled = [
            model.get_param(f"daily_fraction_work_used{age.name}")
            for age in OccupationNetworkEnum
        ]
        multipliers = [
            model.get_param(f"lockdown_occupation_multiplier{age.name}")
            for age in OccupationNetworkEnum
        ]
        
        for non_scaled_i, scaled_i, m in zip(non_scaled, scaled, multipliers):
            assert non_scaled_i * m == scaled_i
        
        
        model.update_running_params("lockdown_occupation_multiplier_primary_network", 10.0)
        scaled = [
            model.get_param(f"daily_fraction_work_used{age.name}")
            for age in OccupationNetworkEnum
        ]
        multipliers = [
            model.get_param(f"lockdown_occupation_multiplier{age.name}")
            for age in OccupationNetworkEnum
        ]
        
        for non_scaled_i, scaled_i, m in zip(non_scaled, scaled, multipliers):
            assert non_scaled_i * m == scaled_i
