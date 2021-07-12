#
# Example Vaccination
#
# This examples demonstrates the effect of a vaccination programme started
# during a lockdown period
#
# The population is populated at the rate of 2% of adults the population a day
##
# Created: 8 April 2021
# Author: roberthinch
#

library( OpenABMCovid19)

# get the model overiding a couple of params
n_total = 10000
model  = Model.new( params = list( n_total = n_total ) )

# Run the model until there are 2% infections
model$one_time_step()
while( model$one_time_step_results()[["total_infected"]] < n_total * 0.02 )
  model$one_time_step()

# Turn on lockdown and run the model for another 30 days and vaccinated
Model.update_running_params( model, "self_quarantine_fraction", 0.75 )
model$update_running_params( "lockdown_on", 1 )

# define the vaccination programme
vaccine = model$add_vaccine( full_efficacy = 0.7, symptoms_efficacy = 0.8)
schedule = VaccineSchedule$new(
  frac_0_9 = 0,
  frac_10_19 = 0,
  frac_20_29 = 0.02,
  frac_30_39 = 0.02,
  frac_40_49 = 0.02,
  frac_50_59 = 0.02,
  frac_60_69 = 0.02,
  frac_70_79 = 0.02,
  frac_80 = 0.02,
  vaccine = vaccine
)

for( t in 1:30 )
{
  model$one_time_step( )
  model$vaccinate_schedule( schedule )
}

# Turn off lockdown and run the model for another 50 days and vaccinated
model$update_running_params( "lockdown_on", 0 )
for( t in 1:50 )
{
  model$one_time_step( )
  model$vaccinate_schedule( schedule )
}

results = model$results()
plot( results$time, results$total_infected)
