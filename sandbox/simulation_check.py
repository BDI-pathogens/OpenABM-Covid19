
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



class LockdownAt20(simulation.Agent):
    """
    Class representing a lockdown starting at time 20
    """
    def step(self, state):
        """
        User-defined step function called at each step of a model run
        
        Should return an action (current an empty dict)
        """
        action = {}
        
        if state["time"] == 20:
            action = {"lockdown_on": 1}
        
        return action


def basic_example_lockdown_agent():
    params = Parameters(
        "../tests/data/baseline_parameters.csv", 1, ".", 
        "../tests/data/baseline_household_demographics.csv")
    
    T = 35
    params.set_param("end_time", T)

    model = simulation.COVID19IBM(model = Model(params))
    agent = LockdownAt20()
    sim = simulation.Simulation(env = model, agent = agent, end_time = T, verbose = True)

    # Run the model for 25 time steps, print output
    sim.steps(25)
    print(sim.results)



class LockdownInfectedOnePercent(simulation.Agent):
    """
    Class representing a lockdown starting when 1% of the population are infected
    """
    def step(self, state):
        """
        Turn on lockdown if >1% of population are infected
        """
        action = {}
        
        if state["total_infected"]/10000. > 0.01: # fixme <- agents need to be aware of total popn
            action = {"lockdown_on": 1}
        
        return action


def basic_example_lockdown_infected_agent():
    params = Parameters(
        "../tests/data/baseline_parameters.csv", 1, ".", 
        "../tests/data/baseline_household_demographics.csv")
    
    T = 50
    params.set_param("end_time", T)
    params.set_param("n_total", 10000)

    model = simulation.COVID19IBM(model = Model(params))
    agent = LockdownInfectedOnePercent()
    sim = simulation.Simulation(env = model, agent = agent, end_time = T, verbose = True)

    # Run the model for 25 time steps, print output
    sim.steps(25)
    print(sim.results)


if __name__ == "__main__":
    # Example with a dummy agent (just running the model and not changing intervention params)
    basic_example_dummy_agent()
    
    # Example with lockdown being turned on at a specific time
    basic_example_lockdown_agent()
    
    # Example with lockdown being turned on at a state-dependent point in time (1% infected)
    basic_example_lockdown_infected_agent()
