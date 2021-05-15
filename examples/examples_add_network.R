library( OpenABMCovid19 )

# get the base model
n_total = 10000
m = Model.new(
  params = list(
    n_total = n_total,
    sd_infectiousness_multiplier = 0, # prevent super-spreading
    end_time = 20
) )

# create a network where everyone is connected to person 0
ID_1 = rep( 0, n_total-1)
ID_2 = 1:(n_total-1)
df_network = data.frame( ID_1, ID_2)

# add the network and print out info on all networks
m$add_user_network( df_network, name = "my network")
print( m$get_network_info() )

# infect person 0 and run for the simualtion
m$seed_infect_by_idx( 0 )
Model.run(m, verbose = FALSE )

# see how many people infected by person 0
trans = Model.get_transmissions(m)
print( sprintf( "Person 0 infected %d people", sum(trans$ID_source == 0) ) )

