library(R6)

setwd("..")

ag0_9   <- AgeGroupEnum[['_0_9']]
ag10_19 <- AgeGroupEnum[['_10_19']]
ag20_29 <- AgeGroupEnum[['_20_29']]
ag30_39 <- AgeGroupEnum[['_30_39']]
ag40_49 <- AgeGroupEnum[['_40_49']]
ag50_59 <- AgeGroupEnum[['_50_59']]
ag60_69 <- AgeGroupEnum[['_60_69']]
ag70_79 <- AgeGroupEnum[['_70_79']]
ag80    <- AgeGroupEnum[['_80']]

baseline_params = function()
{
  return(Parameters$new(
    input_param_file           = "data/baseline_parameters.csv",
    param_line_number          = 1,
    output_file_dir            = "data_test",
    input_households           = "data/baseline_household_demographics.csv",
    hospital_input_param_file  = "data/hospital_baseline_parameters.csv",
    hospital_param_line_number = 1,
    read_param_file            = TRUE,
    read_hospital_param_file   = TRUE
  ))
}

test_that("Model::initialize (params_object isn't R6Class)", {
  expect_error(Model$new(10))
})

test_that("Model::baseline_params", {
  # This test requires more than 4 GiB. So skip it on 32-bit machines.
  if(.Machine$sizeof.pointer < 8) {
    m <- Model.new()
    expect_true(is.null(m))
    return();
  }

  # This test is slow, uncomment the next line to skip this test:
  # return()
  m <- Model$new(baseline_params())
  expect_error(m$update_running_params('hospital_on', 0))
  expect_error(m$get_param('fatality_fraction_TYPO'))

  m$update_running_params('manual_trace_on', 1)
  expect_equal(m$get_param('manual_trace_on'), 1)
  m$update_running_params('manual_trace_on', 0)
  expect_equal(m$get_param('manual_trace_on'), 0)

  m$update_running_params('fatality_fraction_0_9', 0.5)
  m$update_running_params('fatality_fraction_20_29', 0.25)
  m$update_running_params('fatality_fraction_80', 0.75)
  expect_equal(m$get_param('fatality_fraction_0_9'), 0.5)
  expect_equal(m$get_param('fatality_fraction_20_29'), 0.25)
  expect_equal(m$get_param('fatality_fraction_80'), 0.75)

  expect_equal(m$get_risk_score(1, ag10_19, ag60_69), 1)
  m$set_risk_score(1, ag10_19, ag60_69, 0.5)
  expect_equal(m$get_risk_score(1, ag10_19, ag60_69), 0.5)

  expect_equal(m$get_risk_score_household(ag10_19, ag60_69), 1)
  m$set_risk_score_household(ag10_19, ag60_69, 0.5)
  expect_equal(m$get_risk_score_household(ag10_19, ag60_69), 0.5)

  # TODO(olegat) test add_user_network()
  # TODO(olegat) test add_user_network_random()
  # TODO(olegat) test get_network_info()
  # TODO(olegat) test delete_network()

  nw <- m$get_network_by_id(3)
  expect_equal( 3, nw$network_id() )
  expect_equal( 'Occupation working network (default)', nw$name() )
  expect_equal( 3916906, nw$n_edges() )
  expect_equal( 559558, nw$n_vertices() )
  expect_equal( 1, nw$skip_hospitalised() )
  expect_equal( 1, nw$skip_quarantined() )
  expect_equal( 1, nw$type() )
  expect_equal( 0.5, nw$daily_fraction() )

  expect_equal(NA, m$get_network_ids(0))
  expect_equal(c(0,1,2,3,4,5,6), m$get_network_ids(3))
  expect_equal(c(0,1,2,3,4,5,6), m$get_network_ids(100))

  df_app_user <- m$get_app_users()
  df_app_user[['app_user']] <- as.integer(!df_app_user[['app_user']])
  m$set_app_users(df_app_user)
  expect_equal(df_app_user, m$get_app_users())



  ##
  ## Begin rudimentary integration test with Simulation and COVID19IBM:
  ##

  ## 1) Check initial state at time = 0
  result_t0 <- c(
    'time' = 0,
    'lockdown' = 0,
    'test_on_symptoms' = 0,
    'app_turned_on' = 0,
    'total_infected' = 10,
    'total_infected_0_9' = 1,
    'total_infected_10_19' = 1,
    'total_infected_20_29' = 3,
    'total_infected_30_39' = 1,
    'total_infected_40_49' = 0,
    'total_infected_50_59' = 1,
    'total_infected_60_69' = 0,
    'total_infected_70_79' = 2,
    'total_infected_80' = 1,
    'total_case' = 0,
    'total_case_0_9' = 0,
    'total_case_10_19' = 0,
    'total_case_20_29' = 0,
    'total_case_30_39' = 0,
    'total_case_40_49' = 0,
    'total_case_50_59' = 0,
    'total_case_60_69' = 0,
    'total_case_70_79' = 0,
    'total_case_80' = 0,
    'total_death' = 0,
    'total_death_0_9' = 0,
    'total_death_10_19' = 0,
    'total_death_20_29' = 0,
    'total_death_30_39' = 0,
    'total_death_40_49' = 0,
    'total_death_50_59' = 0,
    'total_death_60_69' = 0,
    'total_death_70_79' = 0,
    'total_death_80' = 0,
    'daily_death' = 0,
    'daily_death_0_9' = 0,
    'daily_death_10_19' = 0,
    'daily_death_20_29' = 0,
    'daily_death_30_39' = 0,
    'daily_death_40_49' = 0,
    'daily_death_50_59' = 0,
    'daily_death_60_69' = 0,
    'daily_death_70_79' = 0,
    'daily_death_80' = 0,
    'n_presymptom' = 5,
    'n_asymptom' = 5,
    'n_quarantine' = 0,
    'n_tests' = 0,
    'n_symptoms' = 0,
    'n_hospital' = 0,
    'n_hospitalised_recovering' = 0,
    'n_critical' = 0,
    'n_death' = 0,
    'n_recovered' = 0,
    'hospital_admissions' = 0,
    'hospital_admissions_total' = 0,
    'hospital_to_critical_daily' = 0,
    'hospital_to_critical_total' = 0,
    'n_quarantine_infected' = 0,
    'n_quarantine_recovered' = 0,
    'n_quarantine_app_user' = 0,
    'n_quarantine_app_user_infected' = 0,
    'n_quarantine_app_user_recovered' = 0,
    'n_quarantine_events' = 0,
    'n_quarantine_release_events' = 0,
    'n_quarantine_events_app_user' = 0,
    'n_quarantine_release_events_app_user' = 0,
    'R_inst' = -1,
    'R_inst_05' = -1,
    'R_inst_95' = -1)
  expect_equal(m$one_time_step_results(), result_t0)


  ## 2) Run a simulation until time = 3
  sim <- Simulation$new( COVID19IBM$new(m) )
  sim$start_simulation()
  sim$steps(3)


  ## 3) Check the results of simulation
  result_t3 <- c(
    'time' = 3,
    'lockdown' = 0,
    'test_on_symptoms' = 0,
    'app_turned_on' = 0,
    'total_infected' = 18,
    'total_infected_0_9' = 1,
    'total_infected_10_19' = 1,
    'total_infected_20_29' = 3,
    'total_infected_30_39' = 2,
    'total_infected_40_49' = 1,
    'total_infected_50_59' = 3,
    'total_infected_60_69' = 3,
    'total_infected_70_79' = 2,
    'total_infected_80' = 2,
    'total_case' = 0,
    'total_case_0_9' = 0,
    'total_case_10_19' = 0,
    'total_case_20_29' = 0,
    'total_case_30_39' = 0,
    'total_case_40_49' = 0,
    'total_case_50_59' = 0,
    'total_case_60_69' = 0,
    'total_case_70_79' = 0,
    'total_case_80' = 0,
    'total_death' = 0,
    'total_death_0_9' = 0,
    'total_death_10_19' = 0,
    'total_death_20_29' = 0,
    'total_death_30_39' = 0,
    'total_death_40_49' = 0,
    'total_death_50_59' = 0,
    'total_death_60_69' = 0,
    'total_death_70_79' = 0,
    'total_death_80' = 0,
    'daily_death' = 0,
    'daily_death_0_9' = 0,
    'daily_death_10_19' = 0,
    'daily_death_20_29' = 0,
    'daily_death_30_39' = 0,
    'daily_death_40_49' = 0,
    'daily_death_50_59' = 0,
    'daily_death_60_69' = 0,
    'daily_death_70_79' = 0,
    'daily_death_80' = 0,
    'n_presymptom' = 10,
    'n_asymptom' = 6,
    'n_quarantine' = 0,
    'n_tests' = 0,
    'n_symptoms' = 2,
    'n_hospital' = 0,
    'n_hospitalised_recovering' = 0,
    'n_critical' = 0,
    'n_death' = 0,
    'n_recovered' = 0,
    'hospital_admissions' = 0,
    'hospital_admissions_total' = 0,
    'hospital_to_critical_daily' = 0,
    'hospital_to_critical_total' = 0,
    'n_quarantine_infected' = 0,
    'n_quarantine_recovered' = 0,
    'n_quarantine_app_user' = 0,
    'n_quarantine_app_user_infected' = 0,
    'n_quarantine_app_user_recovered' = 0,
    'n_quarantine_events' = 0,
    'n_quarantine_release_events' = 0,
    'n_quarantine_events_app_user' = 0,
    'n_quarantine_release_events_app_user' = 0,
    'R_inst' = -1,
    'R_inst_05' = -1,
    'R_inst_95' = -1)
  expect_equal(sim$current_state, result_t3)

  results_expected = list()
  results_expected$time <- c(1, 2, 3)
  results_expected$lockdown <- c(0, 0, 0)
  results_expected$test_on_symptoms <- c(0, 0, 0)
  results_expected$app_turned_on <- c(0, 0, 0)
  results_expected$total_infected <- c(10, 10, 18)
  results_expected$total_infected_0_9 <- c(1, 1, 1)
  results_expected$total_infected_10_19 <- c(1, 1, 1)
  results_expected$total_infected_20_29 <- c(3, 3, 3)
  results_expected$total_infected_30_39 <- c(1, 1, 2)
  results_expected$total_infected_40_49 <- c(0, 0, 1)
  results_expected$total_infected_50_59 <- c(1, 1, 3)
  results_expected$total_infected_60_69 <- c(0, 0, 3)
  results_expected$total_infected_70_79 <- c(2, 2, 2)
  results_expected$total_infected_80 <- c(1, 1, 2)
  results_expected$total_case <- c(0, 0, 0)
  results_expected$total_case_0_9 <- c(0, 0, 0)
  results_expected$total_case_10_19 <- c(0, 0, 0)
  results_expected$total_case_20_29 <- c(0, 0, 0)
  results_expected$total_case_30_39 <- c(0, 0, 0)
  results_expected$total_case_40_49 <- c(0, 0, 0)
  results_expected$total_case_50_59 <- c(0, 0, 0)
  results_expected$total_case_60_69 <- c(0, 0, 0)
  results_expected$total_case_70_79 <- c(0, 0, 0)
  results_expected$total_case_80 <- c(0, 0, 0)
  results_expected$total_death <- c(0, 0, 0)
  results_expected$total_death_0_9 <- c(0, 0, 0)
  results_expected$total_death_10_19 <- c(0, 0, 0)
  results_expected$total_death_20_29 <- c(0, 0, 0)
  results_expected$total_death_30_39 <- c(0, 0, 0)
  results_expected$total_death_40_49 <- c(0, 0, 0)
  results_expected$total_death_50_59 <- c(0, 0, 0)
  results_expected$total_death_60_69 <- c(0, 0, 0)
  results_expected$total_death_70_79 <- c(0, 0, 0)
  results_expected$total_death_80 <- c(0, 0, 0)
  results_expected$daily_death <- c(0, 0, 0)
  results_expected$daily_death_0_9 <- c(0, 0, 0)
  results_expected$daily_death_10_19 <- c(0, 0, 0)
  results_expected$daily_death_20_29 <- c(0, 0, 0)
  results_expected$daily_death_30_39 <- c(0, 0, 0)
  results_expected$daily_death_40_49 <- c(0, 0, 0)
  results_expected$daily_death_50_59 <- c(0, 0, 0)
  results_expected$daily_death_60_69 <- c(0, 0, 0)
  results_expected$daily_death_70_79 <- c(0, 0, 0)
  results_expected$daily_death_80 <- c(0, 0, 0)
  results_expected$n_presymptom <- c(5, 4, 10)
  results_expected$n_asymptom <- c(5, 5, 6)
  results_expected$n_quarantine <- c(0, 0, 0)
  results_expected$n_tests <- c(0, 0, 0)
  results_expected$n_symptoms <- c(0, 1, 2)
  results_expected$n_hospital <- c(0, 0, 0)
  results_expected$n_hospitalised_recovering <- c(0, 0, 0)
  results_expected$n_critical <- c(0, 0, 0)
  results_expected$n_death <- c(0, 0, 0)
  results_expected$n_recovered <- c(0, 0, 0)
  results_expected$hospital_admissions <- c(0, 0, 0)
  results_expected$hospital_admissions_total <- c(0, 0, 0)
  results_expected$hospital_to_critical_daily <- c(0, 0, 0)
  results_expected$hospital_to_critical_total <- c(0, 0, 0)
  results_expected$n_quarantine_infected <- c(0, 0, 0)
  results_expected$n_quarantine_recovered <- c(0, 0, 0)
  results_expected$n_quarantine_app_user <- c(0, 0, 0)
  results_expected$n_quarantine_app_user_infected <- c(0, 0, 0)
  results_expected$n_quarantine_app_user_recovered <- c(0, 0, 0)
  results_expected$n_quarantine_events <- c(0, 0, 0)
  results_expected$n_quarantine_release_events <- c(0, 0, 0)
  results_expected$n_quarantine_events_app_user <- c(0, 0, 0)
  results_expected$n_quarantine_release_events_app_user <- c(0, 0, 0)
  results_expected$R_inst <- c(-1 ,-1 ,-1)
  results_expected$R_inst_05 <- c(-1 ,-1 ,-1)
  results_expected$R_inst_95 <- c(-1 ,-1 ,-1)
  expect_equal(sim$results, results_expected)

  # test infections / vaccinations.
  expect_true(m$seed_infect_by_idx(0))
  expect_true(m$vaccinate_individual(2))
  expect_false(m$vaccinate_schedule(VaccineSchedule$new()))
})

test_that("Model::default params", {
  m = Model.new( params = list( n_total = 10000,
                                end_time = 10,
                                n_seed_infection = 5 ))

  if(.Machine$sizeof.pointer < 8) { # skip this test in 32-bit
    expect_true(is.null(m))
    return();
  }
  Model.run( m, verbose = FALSE  )
  res = Model.results( m )

  expect_equal( res$time[ 10 ], 10 )
  expect_gt( res$total_infected[ 10 ], 10 )
})

test_that("Model::multiple models", {
  # create one model
  m1 = Model.new( params = list( n_total = 10000, end_time = 10 ) )

  # create a second modek
  m2 = Model.new( params = list( n_total = 10000,
                                 end_time = 10,
                                 n_seed_infection = 5 ))

  if(.Machine$sizeof.pointer < 8) { # skip this test in 32-bit
    expect_true(is.null(m1))
    expect_true(is.null(m2))
    return();
  }
  m1$one_time_step()
  m2$one_time_step()

  # run the first model again and destroy (force garbage collection)
  Model.run( m1, verbose = FALSE  )
  res = Model.results( m1 )
  expect_equal( res$time[ 10 ], 10 )
  rm( m1 ); gc()

  # run the second  model again and destroy
  Model.run( m2, verbose = FALSE  )
  res = Model.results( m2 )
  expect_equal( res$time[ 10 ], 10 )
  rm( m2 );

  # create a third model and run
  m3 = Model.new( params = list( n_total = 10000, end_time = 10 ) )
  Model.run( m3, verbose = FALSE  )
  res = Model.results( m3 )
  expect_equal( res$time[ 10 ], 10 )
  rm( m3 ); gc()
})



