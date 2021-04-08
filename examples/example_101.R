#
# Example 101
# Get the baseline parameters, model, run for a few time steps and print the output
#
# Created: 8 April 2021
# Author: roberthinch
#

library( OpenABMCovid19)

# get the model overiding a couple of params
model = Model.new( params = list( n_total = 10000, end_time = 20 ) )

# run the model
Model.run( model )

# print the basic output
print( Model.results( model ) )


