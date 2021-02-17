"""
Example 101
Get the baseline parameters, model, run for a few time steps and print the output

Created: 17 April 2020
Author: roberthinch
"""

import COVID19.model as abm

# get the model overiding a couple of params
model = abm.Model( params = { "n_total" : 10000, "end_time": 20 } )

# run the model
model.run()

# print the basic output
print( model.results )


