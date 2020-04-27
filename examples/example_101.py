"""
Example 101
Get the baseline parameters, model, run for a few time steps and print the output

Created: 17 April 2020
Author: roberthinch
"""

import example_utils as utils
import pandas as pd

# gets the baseline parameters
params = utils.get_baseline_parameters()

# change to run on a small population (much quicker)
params.set_param( "n_total", 10000 )

# get the simulation object
sim = utils.get_simulation( params )

# run the simulation for 10 days
sim.steps( 10 )

# print the basic output
timeseries = pd.DataFrame( sim.results )
print( timeseries )


