
import numpy as np
import sys
sys.path.append("../src/COVID19")

from model import Model, Parameters, ModelParameterException
import simulation

def basic_example_dummy_agent():
    params = Parameters(
        "../tests/data/baseline_parameters.csv", 1, ".", 
        "../tests/data/baseline_household_demographics.csv")

    T = 15
    params.set_param("end_time", T)

    model = simulation.COVID19IBM(model = Model(params))
    agent = simulation.Agent() # Dummy agent
    sim = simulation.Simulation(env = model, agent = agent, end_time = T, verbose = True)

    # Run the model for 3 time steps, print output
    sim.steps(3)
    print(sim.results)

    # RUn for another 5 time steps, print output
    sim.steps(5)
    print(sim.results)


if __name__ == "__main__":
    # Example with a dummy agent (just running the model and not changing intervention params)
    basic_example_dummy_agent()
    