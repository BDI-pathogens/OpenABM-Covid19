"""
Utility functions for examples for OpenABM-Covid19

Created: 17 April 2020
Author: roberthinch
"""
import sys
sys.path.append("../src/COVID19")

from COVID19.model import Model, Parameters, ModelParameterException
import COVID19.simulation as simulation

input_parameter_file = "../tests/data/baseline_parameters.csv"
parameter_line_number = 1
output_dir = "."
household_demographics_file = "../tests/data/baseline_household_demographics.csv"

def get_baseline_parameters():
    params = Parameters(input_parameter_file, parameter_line_number, output_dir, household_demographics_file)
    return params

def get_simulation( params ):
    params.set_param( "end_time", 500 )
    model = simulation.COVID19IBM(model = Model(params))
    sim = simulation.Simulation(env = model, end_time = params.get_param( "end_time" ) )
    return sim
