"""
Class representing a simulation object to store model output and
run model across mutiple time steps or simulations

Created: 5 April 2020
Author: p-robot
"""

import copy
from collections import defaultdict


class Environment:
    """
    Class representing an environment object that defines the system 
    """

    def __init__(self, verbose=False):
        self._start = None

    def step(self, action):
        """
        Method called at the start of each simulation
        """
        reward = None
        next_state = None

        return reward, next_state

    def start_simulation(self):
        """
        Initialize the Environment object for the start of a simulation
        """
        return self.start

    @property
    def start(self):
        "Return the starting state"
        return self._start


class Agent:
    """
    Class representing an Agent object for dictating policy through time and 
    storing value function (that maps states of the model to actions).  
    """

    def __init__(self, verbose=False):
        pass

    def start_simulation(self, state):
        """
        Initialize the Agent object for the start of a simulation
        """
        return {}

    def step(self, state):
        """
        User-defined step function called at each step of a model run
        
        Should return an action (current an empty dict)
        """
        return {}


class Simulation:
    """
    Simulation object to run the model and store data across multiple simulations
    """

    def __init__(self, env, agent=Agent(), end_time=None, verbose=False):
        self.env = env
        self.agent = agent

        self.current_state = None
        self.current_action = None

        self.timestep = None
        self.simulation_number = None

        self.end_time = end_time  # fixme: add end_time to accessible params in model.get_param()

        # Containers for model output
        self.results = None
        self.results_all_simulations = []

        self.verbose = verbose

        self.timestep = 0
        self.sim_started = False

    def start_simulation(self):
        """Initialisation of the simulation; reset the model
        """
        if self.verbose:
            print("Starting simulation")

        # Set the current time step
        self.timestep = 0

        # Reset the model, fixme - add method for initialising the model
        self.current_state = self.env.start_simulation()

        # Reset the agent
        self.current_action = self.agent.start_simulation(self.current_state)

        # Append current state
        if self.results:
            self.results_all_simulations.append(copy.copy(self.results))

        self.results = defaultdict(list)
        self.sim_started = True

    def end_simulation(self):
        if self.sim_started:
            if self.verbose:
                print("Ending simulation")
            self.sim_started = False
            self.env.end_simulation()

    def steps(self, n_steps):
        """
        Run the model for a specific number of steps, starting from the current state,
        save data as model progresses.

        Arguments
        ---------
            n_steps: number of steps for which to call self.model.one_time_step()
        """

        if not self.sim_started:
            self.start_simulation()

        for ts in range(n_steps):
            if self.verbose:
                print("Current timestep:", self.timestep)

            next_state = self.env.step(self.current_action)
            next_action = self.agent.step(next_state)

            # Save the state of the model
            self.collect_results(next_state, next_action)

            if self.timestep < self.end_time:
                self.current_state = next_state
                self.current_action = next_action
                self.timestep += 1
            else:  # if at the end_time of the model then exit
                self.end_simulation()
                if self.verbose:
                    print("Reached end time of simulation before completing all steps")
                break

    def collect_results(self, state, action):
        """Collect model results at each step; fixme action is not currently stored
        """
        # Save results to a defaultdict
        for key, value in state.items():
            self.results[key].append(value)

    @property
    def is_terminal_state(self):
        "Is the current state the terminal state"
        return False


class COVID19IBM(Environment):
    """
    Environment subclass representing a COVID19 outbreak as defined in the COVID19-IBM model
    """

    def __init__(self, model):
        self.model = model
        super().__init__()

    def start_simulation(self):
        """
        Start a simulation, return the state of the system
        """
        pass

    def end_simulation(self):
        self.model._destroy()

    def step(self, action):
        """
        Run the simulation through one time step, return the state of the system
        """

        # If the action is non-empty, then update model parameters in the simulation
        if action:
            for param, value in action.items():
                self.model.update_running_params(param, value)

        self.model.one_time_step()
        return (self.model.one_time_step_results())
