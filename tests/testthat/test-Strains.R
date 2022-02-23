library(R6)
library( OpenABMCovid19)
library( testthat )
library( data.table )

test_that("test_multiple_strain_generation_time", {

  sd_tol <- 3

  base_params <-list(
    n_total = 10000,
    mean_infectious_period = 5.5,
    infectious_rate = 3,
    end_time = 50,
    max_n_strains = 2,
    n_seed_infection = 0
  )
  n_seed_infection <- 200
  mean_infectious_period_1 <- 3.5

  # create model and add strain
  abm     <- Model.new( params = base_params)
  strain1 <- abm$add_new_strain( mean_infectious_period = mean_infectious_period_1 )

  # seed infection in both strains
  inf_id <- sample( 1:base_params[["n_total"]], n_seed_infection * 2, replace = FALSE)
  for( idx in 1:n_seed_infection ) {
    abm$seed_infect_by_idx( inf_id[ idx ], strain_idx = 0 )
    abm$seed_infect_by_idx( inf_id[ idx + n_seed_infection ], strain_idx = 1 )
  }
  abm$run( verbose = FALSE )

  trans     <- as.data.table( abm$get_transmissions())
  gen_times <- trans[ time_infected_source == 0 & time_infected > 0, .( generation_time = mean( time_infected ), N = .N ), by = "strain_idx" ][order(strain_idx)]
  gen_times[ , tol := sd_tol / sqrt( N ) * abm$get_param( "sd_infectious_period")]
  gen_times[ , target := c( base_params[[ "mean_infectious_period" ]], mean_infectious_period_1 )]

  expect_lt( gen_times[ 1, abs( generation_time - target ) ], gen_times[ 1, tol ])
  expect_lt( gen_times[ 2, abs( generation_time - target ) ], gen_times[ 2, tol ])
} )
