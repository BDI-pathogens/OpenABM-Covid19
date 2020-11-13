library(R6)

setwd("..")

Parameters$set("public", "test_read_household_demographics", function() {
  private$read_household_demographics()
})

Parameters$set("public", "test_get_REFERENCE_HOUSEHOLDS", function() {
  private$get_REFERENCE_HOUSEHOLDS()
})

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

test_that("Parameters:get_param / set_param", {
  p <- Parameters$new(
    input_households = "init_value.csv",
    read_param_file = FALSE)
  expect_equal(p$get_param('input_household_file'), "init_value.csv")
  p$set_param('input_household_file', "new_value.csv")
  expect_equal(p$get_param('input_household_file'), "new_value.csv")
})

test_that("Parameters::read_household_demographics (data frame)", {
  R_df <- read.csv("data/baseline_household_demographics.csv")
  p <- Parameters$new(
    input_households = R_df,
    read_param_file = FALSE, read_hospital_param_file = FALSE)
  p$test_read_household_demographics()
  C_df <- p$test_get_REFERENCE_HOUSEHOLDS()
  expect_true(all.equal(R_df, C_df))
})

test_that("Parameters::read_household_demographics (file path)", {
  p <- Parameters$new(
    input_households = "data/baseline_household_demographics.csv",
    read_param_file = FALSE, read_hospital_param_file = FALSE)
  p$test_read_household_demographics()
  R_df <- read.csv("data/baseline_household_demographics.csv")
  C_df <- p$test_get_REFERENCE_HOUSEHOLDS()
  expect_true(all.equal(R_df, C_df))
})

test_that("Parameters::set_demographic_household_table (success)", {
  p <- Parameters$new(
    input_param_file         = "data/baseline_parameters.csv",
    param_line_number        = 1,
    output_file_dir          = "data_test",
    input_households         = "data/baseline_household_demographics.csv",
    read_param_file          = TRUE,
    read_hospital_param_file = FALSE)
  n_total <- 10L
  p$c_params$n_total <- n_total
  ID <- seq(from = 1, to = n_total, by = 1)
  df <- data.frame('ID' = ID, 'age_group' = ID %% 9, 'house_no' = ID %/% 4)
  p$set_demographic_household_table(df)
  # TODO(olegat) write proper test expectations:
  expect_true(TRUE)
})

test_that("Parameters::set_demographic_household_table (invalid n_total)", {
  p <- Parameters$new(
    input_households = "data/baseline_household_demographics.csv",
    read_param_file = FALSE, read_hospital_param_file = FALSE)
  n_total <- 10L
  p$c_params$n_total <- 11L
  ID <- seq(from = 1, to = n_total, by = 1)
  df <- data.frame('ID' = ID, 'age_group' = ID %% 9, 'house_no' = ID %/% 4)
  expect_error(p$set_demographic_household_table(df))
})
