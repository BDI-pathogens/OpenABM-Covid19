library(R6)

setwd("..")

ag0_9   <- AgeGroups$ag0_9
ag10_19 <- AgeGroups$ag10_19
ag20_29 <- AgeGroups$ag20_29
ag30_39 <- AgeGroups$ag30_39
ag40_49 <- AgeGroups$ag40_49
ag50_59 <- AgeGroups$ag50_59
ag60_69 <- AgeGroups$ag60_69
ag70_79 <- AgeGroups$ag70_79
ag80    <- AgeGroups$ag80

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
