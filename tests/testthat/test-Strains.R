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
  inf_id <- sample( 0:(base_params[["n_total"]]-1), base_params[["n_seed_infection"]], replace = FALSE)
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

test_that("test_multi_strain_disease_outcome_proportions", {

  # Test that the fraction of infected people following each path for
  # the progression of the disease agrees with the parameters (multi-strain version)
  std_error_limit = 3

  test_params = list(
      n_total          = 150000,
      n_seed_infection = 200,
      end_time         = 80,
      rebuild_networks = 0,
      infectious_rate  = 6.0,
      max_n_strains    = 2,
      population_0_9   = 10000,
      population_10_19 = 10000,
      population_20_29 = 10000,
      population_30_39 = 10000,
      population_40_49 = 10000,
      population_50_59 = 10000,
      population_60_69 = 10000,
      population_70_79 = 10000,
      population_80    = 10000,
      fraction_asymptomatic_0_9   = 0.10,
      fraction_asymptomatic_10_19 = 0.10,
      fraction_asymptomatic_20_29 = 0.10,
      fraction_asymptomatic_30_39 = 0.15,
      fraction_asymptomatic_40_49 = 0.15,
      fraction_asymptomatic_50_59 = 0.15,
      fraction_asymptomatic_60_69 = 0.20,
      fraction_asymptomatic_70_79 = 0.20,
      fraction_asymptomatic_80    = 0.20,
      mild_fraction_0_9           = 0.20,
      mild_fraction_10_19         = 0.20,
      mild_fraction_20_29         = 0.20,
      mild_fraction_30_39         = 0.15,
      mild_fraction_40_49         = 0.15,
      mild_fraction_50_59         = 0.15,
      mild_fraction_60_69         = 0.10,
      mild_fraction_70_79         = 0.10,
      mild_fraction_80            = 0.10,
      hospitalised_fraction_0_9  =0.40,
      hospitalised_fraction_10_19=0.40,
      hospitalised_fraction_20_29=0.40,
      hospitalised_fraction_30_39=0.60,
      hospitalised_fraction_40_49=0.60,
      hospitalised_fraction_50_59=0.60,
      hospitalised_fraction_60_69=0.80,
      hospitalised_fraction_70_79=0.80,
      hospitalised_fraction_80   =0.80,
      critical_fraction_0_9  =0.60,
      critical_fraction_10_19=0.60,
      critical_fraction_20_29=0.60,
      critical_fraction_30_39=0.70,
      critical_fraction_40_49=0.70,
      critical_fraction_50_59=0.70,
      critical_fraction_60_69=0.80,
      critical_fraction_70_79=0.80,
      critical_fraction_80   =0.80,
      fatality_fraction_0_9  =0.40,
      fatality_fraction_10_19=0.40,
      fatality_fraction_20_29=0.40,
      fatality_fraction_30_39=0.30,
      fatality_fraction_40_49=0.30,
      fatality_fraction_50_59=0.30,
      fatality_fraction_60_69=0.20,
      fatality_fraction_70_79=0.20,
      fatality_fraction_80   =0.20
    )
    strain1_params = list(
      fraction_asymptomatic = c( 0.2, 0.2, 0.2, 0.1, 0.1, 0.1, 0.15, 0.15, 0.15 ),
      mild_fraction         = c( 0.1, 0.1, 0.1, 0.2, 0.2, 0.2, 0.15, 0.15, 0.15 ),
      hospitalised_fraction = c( 0.6, 0.6, 0.6, 0.8, 0.8, 0.8, 0.4, 0.4, 0.4 ),
      critical_fraction     = c( 0.8, 0.8, 0.8, 0.5, 0.5, 0.5, 0.6, 0.6, 0.6 ),
      fatality_fraction     = c( 0.3, 0.3, 0.3, 0.5, 0.5, 0.5, 0.8, 0.8, 0.8 )
    )

    fraction_asymptomatic = c(
      test_params[[ "fraction_asymptomatic_0_9" ]],
      test_params[[ "fraction_asymptomatic_10_19" ]],
      test_params[[ "fraction_asymptomatic_20_29" ]],
      test_params[[ "fraction_asymptomatic_30_39" ]],
      test_params[[ "fraction_asymptomatic_40_49" ]],
      test_params[[ "fraction_asymptomatic_50_59" ]],
      test_params[[ "fraction_asymptomatic_60_69" ]],
      test_params[[ "fraction_asymptomatic_70_79" ]],
      test_params[[ "fraction_asymptomatic_80" ]]
    )
    fraction_asymptomatic_1 = strain1_params[[ "fraction_asymptomatic" ]]

    mild_fraction = c(
      test_params[[ "mild_fraction_0_9" ]],
      test_params[[ "mild_fraction_10_19" ]],
      test_params[[ "mild_fraction_20_29" ]],
      test_params[[ "mild_fraction_30_39" ]],
      test_params[[ "mild_fraction_40_49" ]],
      test_params[[ "mild_fraction_50_59" ]],
      test_params[[ "mild_fraction_60_69" ]],
      test_params[[ "mild_fraction_70_79" ]],
      test_params[[ "mild_fraction_80" ]]
    )
    mild_fraction_1 = strain1_params[[ "mild_fraction" ]]

    hospitalised_fraction = c(
      test_params[[ "hospitalised_fraction_0_9" ]],
      test_params[[ "hospitalised_fraction_10_19" ]],
      test_params[[ "hospitalised_fraction_20_29" ]],
      test_params[[ "hospitalised_fraction_30_39" ]],
      test_params[[ "hospitalised_fraction_40_49" ]],
      test_params[[ "hospitalised_fraction_50_59" ]],
      test_params[[ "hospitalised_fraction_60_69" ]],
      test_params[[ "hospitalised_fraction_70_79" ]],
      test_params[[ "hospitalised_fraction_80" ]]
    )
    hospitalised_fraction_1 = strain1_params[[ "hospitalised_fraction"]]

    critical_fraction = c(
      test_params[[ "critical_fraction_0_9" ]],
      test_params[[ "critical_fraction_10_19" ]],
      test_params[[ "critical_fraction_20_29" ]],
      test_params[[ "critical_fraction_30_39" ]],
      test_params[[ "critical_fraction_40_49" ]],
      test_params[[ "critical_fraction_50_59" ]],
      test_params[[ "critical_fraction_60_69" ]],
      test_params[[ "critical_fraction_70_79" ]],
      test_params[[ "critical_fraction_80" ]]
    )
    critical_fraction_1 = strain1_params[[ "critical_fraction" ]]

    fatality_fraction = c(
      test_params[[ "fatality_fraction_0_9" ]],
      test_params[[ "fatality_fraction_10_19" ]],
      test_params[[ "fatality_fraction_20_29" ]],
      test_params[[ "fatality_fraction_30_39" ]],
      test_params[[ "fatality_fraction_40_49" ]],
      test_params[[ "fatality_fraction_50_59" ]],
      test_params[[ "fatality_fraction_60_69" ]],
      test_params[[ "fatality_fraction_70_79" ]],
      test_params[[ "fatality_fraction_80" ]]
    )
    fatality_fraction_1 = strain1_params[[ "fatality_fraction" ]]

    # create model and add the new strain
    abm     <- Model.new( params = test_params)
    strain1 <- abm$add_new_strain( 1,
         fraction_asymptomatic = strain1_params[[ "fraction_asymptomatic" ]],
         mild_fraction         = strain1_params[[ "mild_fraction" ]],
         hospitalised_fraction = strain1_params[[ "hospitalised_fraction" ]],
         critical_fraction     = strain1_params[[ "critical_fraction" ]],
         fatality_fraction     = strain1_params[[ "fatality_fraction" ]],
    )

    # seed infection in both strains
    inf_id <- sample( 0:(test_params[["n_total"]]-1), test_params[["n_seed_infection"]], replace = FALSE)
    for( idx in 1:test_params[["n_seed_infection"]] ) {
      abm$seed_infect_by_idx( inf_id[ idx  ], strain_idx = 1 )
    }
    abm$run( verbose = FALSE )

    df_trans <- as.data.table( abm$get_transmissions() )
    df_trans[ , ID := ID_recipient ]
    df_indiv <- as.data.table( abm$get_individuals() )
    df_indiv <- df_trans[ df_indiv, on = "ID" ]

    # fraction asymptomatic vs mild+symptomatc
    df_inf      <- df_indiv[ time_infected > 0 ]
    df_asym     <- df_indiv[ time_asymptomatic > 0 ]
    df_mild_p   <- df_indiv[ time_presymptomatic_mild > 0 ]
    df_sev_p    <- df_indiv[ time_presymptomatic_severe > 0 ]
    df_sev      <- df_indiv[ time_symptomatic_severe > 0 ]
    df_hosp     <- df_indiv[ time_hospitalised > 0 ]
    df_crit     <- df_indiv[ ( time_critical > 0 ) | ( time_death > 0 ) ]
    df_icu      <- df_indiv[ time_critical > 0 ]
    df_dead_icu <- df_indiv[ ( time_critical > 0 ) & ( time_death > 0 ) ]

    for( idx in 1:length( AgeGroupEnum) ) {
      age = idx -1 # internal OpenABM index for age groups is from 0 to 8

      N_inf        <- df_inf[ age_group == age    & strain_idx == 0, .N ]
      N_inf_1      <- df_inf[ age_group == age    & strain_idx == 1, .N ]
      N_asym       <- df_asym[ age_group == age   & strain_idx == 0, .N ]
      N_asym_1     <- df_asym[ age_group == age   & strain_idx == 1, .N ]
      N_sev_p      <- df_sev_p[ age_group == age  & strain_idx == 0, .N ]
      N_sev_p_1    <- df_sev_p[ age_group == age  & strain_idx == 1, .N ]
      N_mild_p     <- df_mild_p[ age_group == age & strain_idx == 0, .N ]
      N_mild_p_1   <- df_mild_p[ age_group == age & strain_idx == 1, .N ]
      N_sev        <- df_sev[ age_group == age    & strain_idx == 0, .N ]
      N_sev_1      <- df_sev[ age_group == age    & strain_idx == 1, .N ]
      N_hosp       <- df_hosp[ age_group == age   & strain_idx == 0, .N ]
      N_hosp_1     <- df_hosp[ age_group == age   & strain_idx == 1, .N ]
      N_crit       <- df_crit[ age_group == age   & strain_idx == 0, .N ]
      N_crit_1     <- df_crit[ age_group == age   & strain_idx == 1, .N ]
      N_icu        <- df_icu[ age_group == age    & strain_idx == 0, .N ]
      N_icu_1      <- df_icu[ age_group == age    & strain_idx == 1, .N ]
      N_dead_icu   <- df_dead_icu[ age_group == age & strain_idx == 0, .N ]
      N_dead_icu_1 <- df_dead_icu[ age_group == age & strain_idx == 1, .N ]

      expect_equal( N_inf, ( N_sev_p + N_asym + N_mild_p ), info = "missing infected people" )
      expect_equal( N_inf_1, ( N_sev_p_1 + N_asym_1 + N_mild_p_1 ), info = "missing infected people" )
      expect_lt( 50, N_asym,     label = "insufficient asymptomtatic to test" )
      expect_lt( 50, N_asym_1,   label = "insufficient asymptomtatic to test" )
      expect_lt( 50, N_mild_p,   label = "insufficient mild to test" )
      expect_lt( 50, N_mild_p_1, label = "insufficient mild to test" )
      expect_lt( 50, N_hosp,     label = "insufficient hospitalised to test" )
      expect_lt( 50, N_hosp_1,   label = "insufficient hospitalised to test" )
      expect_lt( 40, N_icu,      label = "insufficient ICU to test" )
      expect_lt( 40, N_icu_1,    label =" insufficient ICU to test" )

      expect_lt( abs( N_asym - N_inf * fraction_asymptomatic[idx]),       std_error_limit * sqrt( N_inf * fraction_asymptomatic[idx ] ),     label = "incorrect asymptomatics" )
      expect_lt( abs( N_asym_1 - N_inf_1 * fraction_asymptomatic_1[idx]), std_error_limit * sqrt( N_inf_1 * fraction_asymptomatic_1[idx ] ), label = "incorrect asymptomatics" )
      expect_lt( abs( N_mild_p - N_inf * mild_fraction[idx]),             std_error_limit * sqrt( N_inf * mild_fraction[idx ] ),             label = "incorrect milds" )
      expect_lt( abs( N_mild_p_1 - N_inf_1 * mild_fraction_1[idx]),       std_error_limit * sqrt( N_inf_1 * mild_fraction_1[idx ] ),         label = "incorrect milds" )
      expect_lt( abs( N_hosp - N_sev * hospitalised_fraction[idx]),       std_error_limit * sqrt( N_sev * hospitalised_fraction[idx ] ),     label = "incorrect hospitalises" )
      expect_lt( abs( N_hosp_1 - N_sev_1 * hospitalised_fraction_1[idx]), std_error_limit * sqrt( N_sev_1 * hospitalised_fraction_1[idx ] ), label = "incorrect hospitalises" )
      expect_lt( abs( N_crit - N_hosp * critical_fraction[idx]),          std_error_limit * sqrt( N_hosp * critical_fraction[idx ] ),        label = "incorrect criticals" )
      expect_lt( abs( N_crit_1 - N_hosp_1 * critical_fraction_1[idx]),    std_error_limit * sqrt( N_hosp_1 * critical_fraction_1[idx ] ),    label = "incorrect criticals" )
      expect_lt( abs( N_dead_icu - N_icu * fatality_fraction[idx]),       std_error_limit * sqrt( N_icu * fatality_fraction[idx ] ),         label = "incorrect fatalaties" )
      expect_lt( abs( N_dead_icu_1 - N_icu_1 * fatality_fraction_1[idx]), std_error_limit * sqrt( N_icu_1 * fatality_fraction_1[idx ] ),     label= "incorrect fatalaties" )
    }
} )

