library(R6)

setwd("..")

test_that("Parameters is R6 class", {
  expect_equal(is.R6Class(Parameters), TRUE)
})

test_that("Parameters::initialize (household demographics CSV file only)", {
  p <- Parameters$new(
    input_param_file           = NA_character_,
    param_line_number          = 1,
    output_file_dir            = "data_test",
    input_households           = "data/baseline_household_demographics.csv",
    hospital_input_param_file  = NA_character_,
    hospital_param_line_number = 1,
    read_param_file            = FALSE,
    read_hospital_param_file   = FALSE
  )
  expect_equal(p$c_params$input_param_file, "")
  expect_equal(p$c_params$param_line_number, 1)
  expect_equal(p$c_params$output_file_dir, "data_test")
  expect_equal(p$c_params$input_household_file, "data/baseline_household_demographics.csv")
  expect_equal(p$c_params$hospital_input_param_file, "")
  expect_equal(p$c_params$hospital_param_line_number, 1)
})

test_that("Parameters::initialize (no input CSV file)", {
  p <- Parameters$new(
    input_param_file           = NA_character_,
    param_line_number          = 1,
    output_file_dir            = "data_test",
    input_households           = "data/baseline_household_demographics.csv",
    hospital_input_param_file  = "data/hospital_baseline_parameters.csv",
    hospital_param_line_number = 1,
    read_param_file            = FALSE,
    read_hospital_param_file   = TRUE
  )
  expect_equal(p$c_params$input_param_file, "")
  expect_equal(p$c_params$param_line_number, 1)
  expect_equal(p$c_params$output_file_dir, "data_test")
  expect_equal(p$c_params$input_household_file, "data/baseline_household_demographics.csv")
  expect_equal(p$c_params$hospital_input_param_file, "data/hospital_baseline_parameters.csv")
  expect_equal(p$c_params$hospital_param_line_number, 1)
})

test_that("Parameters::initialize (no hospital CSV file)", {
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
  expect_equal(p$c_params$input_param_file, "data/baseline_parameters.csv")
  expect_equal(p$c_params$param_line_number, 1)
  expect_equal(p$c_params$output_file_dir, "data_test")
  expect_equal(p$c_params$input_household_file, "data/baseline_household_demographics.csv")
  expect_equal(p$c_params$hospital_input_param_file, "")
  expect_equal(p$c_params$hospital_param_line_number, 1)
})

test_that("Parameters::initialize (all CSV files)", {
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
  expect_equal(p$c_params$input_param_file, "data/baseline_parameters.csv")
  expect_equal(p$c_params$param_line_number, 1)
  expect_equal(p$c_params$output_file_dir, "data_test")
  expect_equal(p$c_params$input_household_file, "data/baseline_household_demographics.csv")
  expect_equal(p$c_params$hospital_input_param_file, "data/hospital_baseline_parameters.csv")
  expect_equal(p$c_params$hospital_param_line_number, 1)
})
