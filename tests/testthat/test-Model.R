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

test_that("Model::risk_score (set/get)", {
  m <- Model$new(baseline_params())
  expect_equal(m$get_risk_score(1, ag10_19, ag60_69), 1)
  m$set_risk_score(1, ag10_19, ag60_69, 0.5)
  expect_equal(m$get_risk_score(1, ag10_19, ag60_69), 0.5)
})

test_that("Model::risk_score_household (set/get)", {
  m <- Model$new(baseline_params())
  expect_equal(m$get_risk_score_household(ag10_19, ag60_69), 1)
  m$set_risk_score_household(ag10_19, ag60_69, 0.5)
  expect_equal(m$get_risk_score_household(ag10_19, ag60_69), 0.5)
})
