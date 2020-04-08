"""
Class representing a simulation object to store model output and 
run model across mutiple time steps or simulations

Created: 5 April 2020
Author: p-robot
"""

import copy
from collections import defaultdict


class Simulation:
    """
    Simulation object to run the model and store data across multiple simulations
    """
    def __init__(self, model, end_time, verbose = False):
        
        self.model = model
        
        self.timestep = None
        self.simulation = None
        
        self.end_time = end_time # fixme: add end_time to accessible params in model.get_param()
        
        # Containers for model output
        self.results = None
        self.results_all_simulations = []
        
        self.verbose = verbose
        
        self.start_simulation()
    
    def start_simulation(self):
        """Initialisation of the simulation; reset the model
        """
        if self.verbose:
            print("Starting simulation")
        
        # Set the current time step
        self.timestep = 0
        
        # Reset the model, fixme - add method for initialising the model
        self.model._create()
        
        # Append current state
        if self.results:
            self.results_all_simulations.append(copy.copy(self.results))
        
        self.results = defaultdict(list)
        
    
    def steps(self, n_steps):
        """
        Run the model for a specific number of steps, starting from the current state,
        save data as model progresses.  
        
        Arguments
        ---------
            n_steps: number of steps for which to call self.model.one_time_step()
        """
        
        if self.timestep > self.end_time:
            self.start_simulation()
        
        self.simulation = 0
        
        for ts in range(n_steps):
            if self.verbose:
                print("Current timestep:", self.timestep)
            self.model.one_time_step()
            
            # Save the data from the model
            self.collect_results(self.model.one_time_step_results())
            
            # If at the end_time of the model, restart a new simulation
            if self.timestep >= self.end_time:
                self.simulation += 1
                self.start_simulation()
                
            else:
                self.timestep += 1
    
    def simulations(self, n_simulations):
        """
        Run the model for a specific number of simulations, starting from the current state,
        save data as model progresses.
        """
        
        if self.timestep >= self.end_time:
            self.start_simulation()
        
        
        for self.simulation in range(n_simulations):
            
            if self.verbose:
                print("Simulation number:", self.simulation)
            
            self.timestep = 0
            while( self.timestep <= self.end_time):
                
                if self.verbose:
                    print("Current timestep:", self.timestep)
                
                self.model.one_time_step()
            
                # Save the data from the model
                self.collect_results(self.model.one_time_step_results())
                
                self.timestep += 1
            
            self.start_simulation()
    
    def collect_results(self, data):
        """Collect model results at each step
        """
        # Save results to a defaultdict
        for key, value in data.items():
            self.results[key].append(value)
        