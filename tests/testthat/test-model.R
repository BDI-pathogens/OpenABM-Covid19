library(R6)

setwd("..")

test_that("Parameters is R6 class", {
  expect_equal(is.R6Class(Parameters), TRUE)
})

test_that("Parameters::initialize(read_hospital_param_file=FALSE, ...)", {
  p <- Parameters$new(
    input_param_file           = "data/baseline_parameters.csv",
    param_line_number          = 1,
    output_file_dir            = "data_test",
    input_households           = "data/baseline_household_demographics.csv",
    hospital_input_param_file  = NA_character_,
    hospital_param_line_number = 1,
    read_param_file            = TRUE,
    read_hospital_param_file   = FALSE
  )
  expect_true(is.R6(p))
  expect_s4_class(p$c_params, "_p_parameters")
})

test_that("Parameters::initialize(read_hospital_param_file=TRUE, ...)", {
  p <- Parameters$new(
    input_param_file           = "data/baseline_parameters.csv",
    param_line_number          = 1,
    output_file_dir            = "data_test",
    input_households           = "data/baseline_household_demographics.csv",
    hospital_input_param_file  = "data/hospital_baseline_parameters.csv",
    hospital_param_line_number = 1,
    read_param_file            = TRUE,
    read_hospital_param_file   = TRUE
  )
  expect_true(is.R6(p))
  expect_s4_class(p$c_params, "_p_parameters")
})
