# Example Multi-Strain with Vaccination
#
# Demonstrates running with multiple strains and a vaccination program
#
# An initial epidemic with the base strain is allowed to run and then
# surpressed by a combination of lockdown and vaccination
#
# A new strain is seeded which is more transmissible, only 80% cross immunity
# and lower vaccine efficacy.
#
# A new epidemic in cases occurs but hospitalisations are limited due to
# the vaccines be very effecitve against severe symptoms for the new strain.
#
# Created: 9 Jun 2021
# Author: roberthinch


library( OpenABMCovid19 )
library( data.table )
library( plotly )

n_total = 200000
params = list(
  n_total          = n_total,
  max_n_strains    = 2,
  rebuild_networks = FALSE,
  infectious_rate  = 6.5,
  rng_seed = sample( 1:100, 1),
#  rng_seed = 1,
  app_users_fraction_0_9   = 0,
  app_users_fraction_10_19 = 0.2,
  app_users_fraction_20_29 = 0.5,
  app_users_fraction_30_39 = 0.5,
  app_users_fraction_40_49 = 0.5,
  app_users_fraction_50_59 = 0.5,
  app_users_fraction_60_69 = 0.5,
  app_users_fraction_70_79 = 0.25,
  app_users_fraction_80 = 0.2,
  quarantine_days = 1,
  days_of_interactions = 1
)
abm    = Model.new( params = params )
set.seed( params$rng_seed )

# add a new strain
transmission_multiplier = 1.4;
hospitalised_fraction   = c( 0.001, 0.001, 0.01,0.05, 0.05, 0.10, 0.15, 0.30, 0.5 )

strain_delta = abm$add_new_strain(
  transmission_multiplier = transmission_multiplier,
  hospitalised_fraction   = hospitalised_fraction )

# set cross-immunity between strains
cross_immunity_mat = matrix( c(
  c( 1.0, 0.8 ),
  c( 0.8, 1.0 )
), nrow = 2 )

abm$set_cross_immunity_matrix( cross_immunity_mat )

# add vaccine
full_efficacy_base      = 0.6
full_efficacy_delta     = 0.3
symptoms_efficacy_base  = 0.85
symptoms_efficacy_delta = 0.65
severe_efficacy_base    = 0.95
severe_efficacy_delta   = 0.90
vaccine = abm$add_vaccine(
  full_efficacy     = c( full_efficacy_base, full_efficacy_delta ),
  symptoms_efficacy = c( symptoms_efficacy_base, symptoms_efficacy_delta ),
  severe_efficacy   = c( severe_efficacy_base, severe_efficacy_delta ),
  time_to_protect   = 14,
  vaccine_protection_period = 365
)

# create a vaccine schedule
schedule = VaccineSchedule$new(
  frac_0_9 = 0.0,
  frac_10_19 = 0.0,
  frac_20_29 = 0.02,
  frac_30_39 = 0.02,
  frac_40_49 = 0.02,
  frac_50_59 = 0.02,
  frac_60_69 = 0.02,
  frac_70_79 = 0.02,
  frac_80 = 0.02,
  vaccine = vaccine
)

# run the model until 1% infected
abm$one_time_step()
while(  tail( abm$results()$total_infected, n = 1 ) < n_total * 0.01 )
  abm$one_time_step()

# add lockdown and start vaccination schedule
abm$update_running_params( "lockdown_on", TRUE )

# run the model for 70 time steps and vaccinate
for( t in 1:70 )
{
  if( t < 45 )
    abm$vaccinate_schedule( schedule )
  abm$one_time_step()
}


# turn off lockdown and put in some social distancing (10% reduction in transmission rates outside), then run for 50 time steps
abm$update_running_params( "lockdown_on", FALSE )
abm$update_running_params( "relative_transmission_occupation", 0.85 )
abm$update_running_params( "relative_transmission_random", 0.85 )
abm$update_running_params( "self_quarantine_fraction", 0.6 )
abm$update_running_params( "quarantine_household_on_symptoms", 1 )
abm$update_running_params( "manual_trace_on", FALSE )
abm$update_running_params( "app_turned_on", TRUE )
abm$update_running_params( "quarantine_on_traced", 1 )
abm$update_running_params( "trace_on_symptoms", 1 )
abm$run( 40 )

# seed delta strain in 20 random people
idx_seed = sample( 1:n_total, 50 )
for( seed in idx_seed )
  abm$seed_infect_by_idx( ID = seed, strain = strain_delta )

# run the model for 100 more time steps
abm$run( 100 )

results = as.data.table( abm$results() )
results[ , new_infected := total_infected - shift( total_infected )]

trans = as.data.table( abm$get_transmissions() )
alpha = trans[ strain_idx == 0, .(new_infected = .N ), by = "time_infected" ][ ,.( time = time_infected, new_infected)][ order( time)]
delta = trans[ strain_idx == 1, .(new_infected = .N ), by = "time_infected" ][ ,.( time = time_infected, new_infected)][ order( time)]

t_inf_hosp = abm$get_param( "mean_time_to_symptoms")+ abm$get_param( "mean_time_to_hospital")
expHosp = trans[ , .(expected_hospitalisation = sum( expected_hospitalisation)), by = "time_infected"][ ,.( time = time_infected + t_inf_hosp , expected_hospitalisation)][ order( time)]

p = plot_ly(
  alpha,
  x = ~time,
  y = ~new_infected,
  type = "scatter",
  mode = "lines",
  name = "alpha"
) %>%
add_lines(
  data = delta,
  name = "delta"
) %>%
add_lines(
  data = results,
  y = ~hospital_admissions,
  yaxis = "y2",
  name = "hospital admissions"
)%>%
add_lines(
  data = expHosp,
  y = ~expected_hospitalisation,
  yaxis = "y2",
  name = "expected hospital admissions"
) %>%
layout(
  yaxis2 = list(
    overlaying = "y",
    side = "right"
  )
)
show( p )




