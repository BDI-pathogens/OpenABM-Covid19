#!/usr/bin/env python3
"""
Set of tests to try setting values in the python parameters object 
"""

import sys

sys.path.append("src/COVID19")
from model import Parameters, ModelParameterException, ParameterException
import pytest


class TestParameters(object):
    def test_set_parameters_arrays_init_to_zero(self):
        p = Parameters(input_households="notset.csv", read_param_file=False, read_hospital_param_file=False)
        assert p.get_param("population_40_49") == 0

    def test_set_parameters_arrays_set_single_value(self):
        p = Parameters(input_households="notset.csv", read_param_file=False, read_hospital_param_file=False)
        assert (
            p.get_param("population_40_49") == 0
        ), "Array memebers not intilialised to zero"
        p.set_param("population_40_49", 400)
        assert (
            p.get_param("population_40_49") == 400
        ), "Did not set pop group to 400"
        assert p.get_param("population_50_59") == 0

    def test_set_age_out_of_range(self):
        p = Parameters(input_households="notset.csv", read_param_file=False, read_hospital_param_file=False)
        with pytest.raises(ParameterException):
            p.set_param("population_80_89", 5000)
