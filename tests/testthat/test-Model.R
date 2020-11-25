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
  # TODO(olegat) test get_network_by_id()
  # TODO(olegat) test get_network_info()
  # TODO(olegat) test delete_network()

  df_app_user <- m$get_app_users()
  df_app_user[['app_user']] <- as.integer(!df_app_user[['app_user']])
  m$set_app_users(df_app_user)
  expect_equal(df_app_user, m$get_app_users())
})
