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

test_that("Model::setters and getters", {
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
  expect_equal( 3920994, nw$n_edges() )
  expect_equal( 560142, nw$n_vertices() )
  expect_equal( 1, nw$skip_hospitalised() )
  expect_equal( 1, nw$skip_quarantined() )
  expect_equal( 1, nw$type() )
  expect_equal( 0.5, nw$daily_fraction() )

  expect_equal(NA, m$get_network_ids(0))
  expect_equal(c(0,1,2), m$get_network_ids(3))
  expect_equal(c(0,1,2,3,4,5,6), m$get_network_ids(100))

  df_app_user <- m$get_app_users()
  df_app_user[['app_user']] <- as.integer(!df_app_user[['app_user']])
  m$set_app_users(df_app_user)
  expect_equal(df_app_user, m$get_app_users())

  one_step_result_t0 <- c(
    'time' = 0,
    'lockdown' = 0,
    'test_on_symptoms' = 0,
    'app_turned_on' = 0,
    'total_infected' = 5,
    'total_infected_0_9' = 0,
    'total_infected_10_19' = 0,
    'total_infected_20_29' = 0,
    'total_infected_30_39' = 2,
    'total_infected_40_49' = 0,
    'total_infected_50_59' = 2,
    'total_infected_60_69' = 0,
    'total_infected_70_79' = 1,
    'total_infected_80' = 0,
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
    'n_presymptom' = 4,
    'n_asymptom' = 1,
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
  expect_equal(m$one_time_step_result(), one_step_result_t0)

  m$one_time_step()
  m$one_time_step()
  m$one_time_step()

  one_step_result_t3 <- c(
    'time' = 3,
    'lockdown' = 0,
    'test_on_symptoms' = 0,
    'app_turned_on' = 0,
    'total_infected' = 8,
    'total_infected_0_9' = 0,
    'total_infected_10_19' = 0,
    'total_infected_20_29' = 0,
    'total_infected_30_39' = 3,
    'total_infected_40_49' = 0,
    'total_infected_50_59' = 2,
    'total_infected_60_69' = 1,
    'total_infected_70_79' = 2,
    'total_infected_80' = 0,
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
    'n_presymptom' = 6,
    'n_asymptom' = 1,
    'n_quarantine' = 0,
    'n_tests' = 0,
    'n_symptoms' = 1,
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
  expect_equal(m$one_time_step_result(), one_step_result_t3)
})
