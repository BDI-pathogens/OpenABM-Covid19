#
# Example 102 - Lockdown and Contract-Tracing
#
# This second example runs the simulation until 1% of the population has been
# infected. At which point a lockdown is imposed for 30 days. After which a
# number of other interventions are implemented such as digital contract-tracing.
#
# Created: 8 April 2021
# Author: roberthinch
#

library( OpenABMCovid19 )

model = Model.new( params = list( n_total = 50000, end_time = 500 ) )

# Run the model until there are 500 infections
Model.one_time_step( model )
while( Model.one_time_step_results( model )[["total_infected"]] < 500 )
  Model.one_time_step( model )

# Turn on lockdown and run the model for another 30 days
Model.update_running_params( model, "lockdown_on", 1 )
for( t in 1:30 )
  Model.one_time_step( model )

# Now turn off the lockdown and turn on digital contract tracing, with the following options.
#
# 1. 80% of people self-quarantine along with their household when they develop symptoms.
#2. Tracing happens as soon as somebody develops symptoms and contacts quarantine themselves.
#3. The households members of those traced also quarantine

# We then run the simulation for another 100 days.

Model.update_running_params( model, "lockdown_on", 0)

# 80% self-quarantine along with their households
Model.update_running_params( model, "self_quarantine_fraction", 0.8 )
Model.update_running_params( model, "quarantine_household_on_symptoms", 1 )

# turn on the app and quarantine those people who have been traced along with their households
Model.update_running_params( model, "app_turned_on", 1 )
Model.update_running_params( model, "quarantine_on_traced", 1 )
Model.update_running_params( model, "trace_on_symptoms", 1 )
Model.update_running_params( model, "quarantine_household_on_traced_symptoms", 1 )

# step forwrard another 100 days
for( t in 1:100 )
  Model.one_time_step( model )

# Plot total infected through time
results = Model.results(model)
plot(results$time, results$total_infected,
     type = "l", col = "blue", xlab = "Time", ylab = "Total infected")

