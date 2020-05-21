"""
Example script for outputting reasons for individuals being in quarantine
"""
import pandas as pd
from COVID19.model import Model, Parameters, ModelParameterException
import COVID19.simulation as simulation

input_parameter_file = "../tests/data/baseline_parameters.csv"
parameter_line_number = 1
output_dir = "."
household_demographics_file = "../tests/data/baseline_household_demographics.csv"

params = Parameters(input_parameter_file, parameter_line_number, 
    output_dir, household_demographics_file)

params.set_param( "end_time", 500 )
params.set_param( "n_total", 10000 )
params.set_param( "test_order_wait", 1 )
params.set_param( "test_result_wait", 1 )
params.set_param( "self_quarantine_fraction", 0.8 )
params.set_param( "quarantine_household_on_positive", 1 )
params.set_param( "quarantine_household_on_symptoms", 1 )
params.set_param( "test_on_symptoms", 1 )
params.set_param( "intervention_start_time", 42 )
params.set_param( "app_turn_on_time", 49 )
params.set_param( "trace_on_symptoms", 1 )
params.set_param( "trace_on_positive", 1 )
params.set_param( "quarantine_on_traced", 1 )

model = simulation.COVID19IBM(model = Model(params))
sim = simulation.Simulation(env = model, end_time = params.get_param( "end_time" ) )

t = 1
shapes = []
output = []; output_index = []
# Run the simulation
while t <= 200:
    sim.steps(1)
    # Write quarantine reasons file
    sim.env.model.write_quarantine_reasons()

    # Read in the file
    df_quarantine_reasons = pd.read_csv("quarantine_reasons_file_Run1.csv")
    
    df_sub = df_quarantine_reasons.groupby("quarantine_reason")["ID"].count().reset_index()
    df_sub["time"] = t
    output_index.append(df_sub)
    
    
    df_sub = df_quarantine_reasons.groupby("status")["ID"].count().reset_index()
    df_sub["time"] = t
    output.append(df_sub)
    t += 1

dff = pd.concat(output)
dff.to_csv("quarantine_by_status.csv", index = False)

dff = pd.concat(output_index)
dff.to_csv("quarantine_by_status_quarantine_reason.csv", index = False)
