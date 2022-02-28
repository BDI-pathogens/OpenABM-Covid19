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

test_that("test_multi_strain_disease_transition_times", {


  # Test that the mean and standard deviation of the transition times between
  # states agrees with the parameters for 2 strains

  std_error_limit <- 4
  eps <- 1e-4

  base_params <- list(
    n_total=50000,
    n_seed_infection=200,
    end_time=30,
    infectious_rate=6.0,
    max_n_strains=2,
    mean_time_to_symptoms=4.0,
    sd_time_to_symptoms=2.0,
    mean_time_to_hospital=1.0,
    mean_time_to_critical=1.0,
    mean_time_to_recover=20.0,
    sd_time_to_recover=8.0,
    mean_time_to_death=12.0,
    sd_time_to_death=5.0,
    mean_asymptomatic_to_recovery=15.0,
    sd_asymptomatic_to_recovery=5.0,
    mean_time_hospitalised_recovery=6,
    sd_time_hospitalised_recovery=3,
    mean_time_critical_survive=4,
    sd_time_critical_survive=2,
    hospitalised_fraction_0_9  =0.2,
    hospitalised_fraction_10_19=0.2,
    hospitalised_fraction_20_29=0.2,
    hospitalised_fraction_30_39=0.2,
    hospitalised_fraction_40_49=0.2,
    hospitalised_fraction_50_59=0.2,
    hospitalised_fraction_60_69=0.2,
    hospitalised_fraction_70_79=0.2,
    hospitalised_fraction_80   =0.2,
    critical_fraction_0_9  =0.50,
    critical_fraction_10_19=0.50,
    critical_fraction_20_29=0.50,
    critical_fraction_30_39=0.50,
    critical_fraction_40_49=0.50,
    critical_fraction_50_59=0.50,
    critical_fraction_60_69=0.50,
    critical_fraction_70_79=0.50,
    critical_fraction_80   =0.50,
    fatality_fraction_0_9  =0.50,
    fatality_fraction_10_19=0.50,
    fatality_fraction_20_29=0.50,
    fatality_fraction_30_39=0.50,
    fatality_fraction_40_49=0.50,
    fatality_fraction_50_59=0.50,
    fatality_fraction_60_69=0.50,
    fatality_fraction_70_79=0.50,
    fatality_fraction_80   =0.50
  )
  strain1_params <- list(
    mean_time_to_symptoms=6.0,
    sd_time_to_symptoms=3.0,
    mean_time_to_hospital=1.8,
    mean_time_to_critical=3.0,
    mean_time_to_recover=12.0,
    sd_time_to_recover=6.0,
    mean_time_to_death=24.0,
    sd_time_to_death=15,
    mean_asymptomatic_to_recovery=10.0,
    sd_asymptomatic_to_recovery=3.0,
    mean_time_hospitalised_recovery=10,
    sd_time_hospitalised_recovery=6,
    mean_time_critical_survive=8,
    sd_time_critical_survive=3
  )

  abm     <- Model.new( params = base_params)
  strain1 <- abm$add_new_strain( 1,
     mean_time_to_symptoms = strain1_params[[ "mean_time_to_symptoms" ]],
     sd_time_to_symptoms   = strain1_params[[ "sd_time_to_symptoms" ]],
     mean_time_to_hospital = strain1_params[[ "mean_time_to_hospital" ]],
     mean_time_to_critical = strain1_params[[ "mean_time_to_critical" ]],
     mean_time_to_recover  = strain1_params[[ "mean_time_to_recover" ]],
     sd_time_to_recover    = strain1_params[[ "sd_time_to_recover" ]],
     mean_time_to_death    = strain1_params[[ "mean_time_to_death" ]],
     sd_time_to_death      = strain1_params[[ "sd_time_to_death" ]],
     mean_asymptomatic_to_recovery   = strain1_params[[ "mean_asymptomatic_to_recovery" ]],
     sd_asymptomatic_to_recovery     = strain1_params[[ "sd_asymptomatic_to_recovery" ]],
     mean_time_hospitalised_recovery = strain1_params[[ "mean_time_hospitalised_recovery" ]],
     sd_time_hospitalised_recovery   = strain1_params[[ "sd_time_hospitalised_recovery" ]],
     mean_time_critical_survive      = strain1_params[[ "mean_time_critical_survive" ]],
     sd_time_critical_survive        = strain1_params[[ "sd_time_critical_survive" ]]
    )

  # seed infection in both strains
  inf_id <- sample( 1:base_params[["n_total"]], base_params[["n_seed_infection"]], replace = FALSE)
  for( idx in 1:base_params[["n_seed_infection"]] ) {
    abm$seed_infect_by_idx( inf_id[ idx  ], strain_idx = 1 )
  }
  abm$run( verbose = FALSE )

  df_trans <- as.data.table( abm$get_transmissions() )
  df_trans[ , t_p_s := time_symptomatic - time_infected ]
  df_t_p_s <- df_trans[ time_infected > 0 & time_asymptomatic < 0 ]
  t_p_s    <- df_t_p_s[ strain_idx == 0, t_p_s ]
  t_p_s_1  <- df_t_p_s[ strain_idx == 1, t_p_s ]
  expect_lt( abs( mean( t_p_s ) - base_params[[ "mean_time_to_symptoms" ]]), std_error_limit * sd( t_p_s ) / sqrt( length( t_p_s ) ) )
  expect_lt( abs( sd( t_p_s ) - base_params[[ "sd_time_to_symptoms" ]]), std_error_limit * sd( t_p_s ) / sqrt( length( t_p_s ) ) )
  expect_lt( abs( mean( t_p_s_1 ) - strain1_params[[ "mean_time_to_symptoms" ]]), std_error_limit * sd( t_p_s_1 ) / sqrt( length( t_p_s_1 ) ) )
  expect_lt( abs( sd( t_p_s_1 ) - strain1_params[[ "sd_time_to_symptoms" ]]), std_error_limit * sd( t_p_s_1 ) / sqrt( length( t_p_s_1 ) ) )

  # time showing symptoms until going to hospital
  df_trans[ , t_s_h := time_hospitalised - time_symptomatic ]
  df_t_s_h <- df_trans[ time_hospitalised > 0 ]
  t_s_h    <- df_t_s_h[ strain_idx == 0, t_s_h ]
  t_s_h_1  <- df_t_s_h[ strain_idx == 1, t_s_h ]
  expect_lt( abs( mean( t_s_h ) - base_params[[ "mean_time_to_hospital" ]]), std_error_limit * sd( t_s_h ) / sqrt( length( t_s_h ) ) + eps )
  expect_lt( abs( mean( t_s_h_1 ) - strain1_params[[ "mean_time_to_hospital" ]]), std_error_limit * sd( t_s_h_1 ) / sqrt( length( t_s_h_1 ) ) + eps )

  # time hospitalised until moving to the ICU
  df_trans[ , t_h_c := time_critical - time_hospitalised ]
  df_t_h_c <- df_trans[ time_critical > 0 ];
  t_h_c    <- df_t_h_c[ strain_idx == 0, t_h_c ];
  t_h_c_1  <- df_t_h_c[ strain_idx == 1, t_h_c ];
  expect_lt( abs( mean( t_h_c ) - base_params[[ "mean_time_to_critical" ]]), std_error_limit * sd( t_h_c ) / sqrt( length( t_h_c ) ) + eps )
  expect_lt( abs( mean( t_h_c_1 ) -  strain1_params[[ "mean_time_to_critical" ]]), std_error_limit * sd( t_h_c_1 ) / sqrt( length( t_h_c_1 ) ) + eps )

  # time from symptoms to recover if not hospitalised
  df_trans[ , t_s_r := time_recovered - time_symptomatic ]
  df_t_s_r <- df_trans[ time_recovered > 0 & time_asymptomatic < 0 & ( time_hospitalised < 0) ]
  t_s_r    <- df_t_s_r[ strain_idx == 0, t_s_r ]
  t_s_r_1  <- df_t_s_r[ strain_idx == 1, t_s_r ]
  expect_lt( abs( mean( t_s_r ) - base_params[[ "mean_time_to_recover" ]]), std_error_limit * sd( t_s_r ) / sqrt( length( t_s_r ) ) + eps )
  expect_lt( abs( sd( t_s_r ) - base_params[[ "sd_time_to_recover" ]]), std_error_limit * sd( t_s_r ) / sqrt( length( t_s_r ) ) + eps )
  expect_lt( abs( mean( t_s_r_1 ) - strain1_params[[ "mean_time_to_recover" ]]), std_error_limit * sd( t_s_r_1 ) / sqrt( length( t_s_r_1 ) ) + eps )
  expect_lt( abs( sd( t_s_r_1 ) - strain1_params[[ "sd_time_to_recover" ]]), std_error_limit * sd( t_s_r_1 ) / sqrt( length( t_s_r_1 ) ) + eps )

  # time from hospitalised to recover if don't got to ICU
  df_trans[ , t_h_r := time_recovered - time_hospitalised ]
  df_t_h_r <- df_trans[ time_recovered  > 0 &  time_critical < 0 &  time_hospitalised > 0 ]
  t_h_r    <- df_t_h_r[ strain_idx == 0, t_h_r ]
  t_h_r_1  <- df_t_h_r[ strain_idx == 1, t_h_r ]
  expect_lt( abs( mean( t_h_r ) - base_params[[ "mean_time_hospitalised_recovery" ]]), std_error_limit * sd( t_h_r ) / sqrt( length( t_h_r ) ) + eps )
  expect_lt( abs( sd( t_h_r ) - base_params[[ "sd_time_hospitalised_recovery" ]]), std_error_limit * sd( t_h_r ) / sqrt( length( t_h_r ) ) + eps )
  expect_lt( abs( mean( t_h_r_1 ) - strain1_params[[ "mean_time_hospitalised_recovery" ]]), std_error_limit * sd( t_h_r_1 ) / sqrt( length( t_h_r_1 ) ) + eps )
  expect_lt( abs( sd( t_h_r_1 ) - strain1_params[[ "sd_time_hospitalised_recovery" ]]), std_error_limit * sd( t_h_r_1 ) / sqrt( length( t_h_r_1 ) ) + eps )

  # time in ICU
  df_trans[ , t_c_r := time_hospitalised_recovering - time_critical ]
  df_t_c_r <- df_trans[  time_hospitalised_recovering  > 0 & time_critical > 0 ]
  t_c_r    <- df_t_c_r[ strain_idx == 0, t_c_r ]
  t_c_r_1  <- df_t_c_r[ strain_idx == 1, t_c_r ]
  expect_lt( abs( mean( t_c_r ) - base_params[[ "mean_time_critical_survive" ]]), std_error_limit * sd( t_c_r ) / sqrt( length( t_c_r ) ) + eps )
  expect_lt( abs( sd( t_c_r ) - base_params[[ "sd_time_critical_survive" ]]), std_error_limit * sd( t_c_r ) / sqrt( length( t_c_r ) ) + eps )
  expect_lt( abs( mean( t_c_r_1 ) - strain1_params[[ "mean_time_critical_survive" ]]), std_error_limit * sd( t_c_r_1 ) / sqrt( length( t_c_r_1 ) ) + eps )
  expect_lt( abs( sd( t_c_r_1 ) - strain1_params[[ "sd_time_critical_survive" ]]), std_error_limit * sd( t_c_r_1 ) / sqrt( length( t_c_r_1 ) ) + eps )

  # time from ICU to death
  df_trans[ , t_c_d := time_death - time_critical ]
  df_t_c_d <- df_trans[ time_death > 0 & time_critical > 0 ]
  t_c_d    <- df_t_c_d[ strain_idx ==0, t_c_d ]
  t_c_d_1  <- df_t_c_d[ strain_idx ==1, t_c_d ]
  expect_lt( abs( mean( t_c_d ) - base_params[[ "mean_time_to_death" ]] ), std_error_limit * sd( t_c_d ) / sqrt( length( t_c_d ) ) + eps )
  expect_lt( abs( sd( t_c_d ) - base_params[[ "sd_time_to_death" ]] ), std_error_limit * sd( t_c_d ) / sqrt( length( t_c_d ) ) + eps )
  expect_lt( abs( mean( t_c_d_1 ) - strain1_params[[ "mean_time_to_death" ]] ), std_error_limit * sd( t_c_d_1 ) / sqrt( length( t_c_d_1 ) ) + eps )
  expect_lt( abs( sd( t_c_d_1 ) - strain1_params[[ "sd_time_to_death" ]] ), std_error_limit * sd( t_c_d_1 ) / sqrt( length( t_c_d_1 ) ) + eps )

  # time from asymptomatic to recover
  df_trans[ , t_a_r := time_recovered - time_asymptomatic ]
  df_t_a_r <- df_trans[ time_recovered > 0 & time_asymptomatic > 0 ]
  t_a_r    <- df_t_a_r[ strain_idx == 0, t_a_r ]
  t_a_r_1  <- df_t_a_r[ strain_idx == 1, t_a_r ]
  expect_lt( abs( mean( t_a_r ) - base_params[[ "mean_asymptomatic_to_recovery" ]] ), std_error_limit * sd( t_a_r ) / sqrt( length( t_a_r ) ) + eps )
  expect_lt( abs( sd( t_a_r ) - base_params[[ "sd_asymptomatic_to_recovery" ]] ), std_error_limit * sd( t_a_r ) / sqrt( length( t_a_r ) ) + eps )
  expect_lt( abs( mean( t_a_r_1 ) - strain1_params[[ "mean_asymptomatic_to_recovery" ]] ), std_error_limit * sd( t_a_r_1 ) / sqrt( length( t_a_r_1 ) ) + eps )
  expect_lt( abs( sd( t_a_r_1 ) - strain1_params[[ "sd_asymptomatic_to_recovery" ]] ), std_error_limit * sd( t_a_r_1 ) / sqrt( length( t_a_r_1 ) ) + eps )

} )

