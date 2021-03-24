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

test_that("Parameters:get_param / set_param (single)", {
  p <- Parameters$new(
    input_households = "data/baseline_household_demographics.csv",
    read_param_file  = FALSE)
  expect_equal(p$get_param('input_household_file'), "data/baseline_household_demographics.csv")
  p$set_param('input_household_file', "new_value.csv")
  expect_equal(p$get_param('input_household_file'), "new_value.csv")
})

test_that("Parameters:get_param / set_param (multi)", {
  p <- Parameters$new(
    input_households = "data/baseline_household_demographics.csv",
    read_param_file  = FALSE)
  expect_error(p$set_param('fatality_fraction_TYPO', 0.75))
  expect_error(p$get_param('fatality_fraction_TYPO'))
  p$set_param('fatality_fraction_0_9', 0.5)
  p$set_param('fatality_fraction_20_29', 0.25)
  p$set_param('fatality_fraction_80', 0.75)
  expect_equal(p$get_param('fatality_fraction_0_9'), 0.5)
  expect_equal(p$get_param('fatality_fraction_20_29'), 0.25)
  expect_equal(p$get_param('fatality_fraction_80'), 0.75)
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

# Tests for Parameters::set_demographic_household_table are losely based off
# of examples/example_user_defined_networks.ipynb
test_that("Parameters::set_demographic_household_table (success)", {
  p <- Parameters$new(
    input_param_file         = "data/baseline_parameters.csv",
    param_line_number        = 1,
    output_file_dir          = "data_test",
    input_households         = "data/baseline_household_demographics.csv",
    read_param_file          = TRUE,
    read_hospital_param_file = FALSE)
  n_total <- 10L
  p$set_param('n_total', n_total)
  ID <- seq(from = 0, to = n_total - 1, by = 1)
  df <- data.frame('ID' = ID, 'age_group' = ID %% 9, 'house_no' = ID %/% 4)
  p$set_demographic_household_table(df)
  # TODO(olegat) write proper test expectations:
  expect_true(TRUE)
})

test_that("Parameters::set_demographic_household_table (invalid house_no)", {
  p <- Parameters$new(
    input_households = "data/baseline_household_demographics.csv",
    read_param_file = FALSE, read_hospital_param_file = FALSE)
  n_total <- 10L
  p$set_param('n_total', n_total)
  ID <- seq(from = 0, to = n_total - 1, by = 1)
  houses <- rep(-1, length(ID))
  df <- data.frame('ID' = ID, 'age_group' = ID %% 9, 'house_no' = houses)
  # TODO(olegat) this prints a message to stdout, which is ugly and distracting
  expect_error(p$set_demographic_household_table(df))
})

test_that("Parameters::set_demographic_household_table (invalid n_total)", {
  p <- Parameters$new(
    input_households = "data/baseline_household_demographics.csv",
    read_param_file = FALSE, read_hospital_param_file = FALSE)
  n_total <- 10L
  p$set_param('n_total', 11L)
  ID <- seq(from = 0, to = n_total - 1, by = 1)
  df <- data.frame('ID' = ID, 'age_group' = ID %% 9, 'house_no' = ID %/% 4)
  expect_error(p$set_demographic_household_table(df))
})

test_that("Parameters::set_demographic_household_table (missing columns)", {
  p <- Parameters$new(
    input_households = "data/baseline_household_demographics.csv",
    read_param_file = FALSE, read_hospital_param_file = FALSE)
  n_total <- 10L
  p$set_param('n_total', n_total)
  ID <- seq(from = 0, to = n_total - 1, by = 1)
  expect_error(p$set_demographic_household_table(data.frame(
    'age_group' = ID %% 9, 'house_no' = ID %/% 4)))
  expect_error(p$set_demographic_household_table(data.frame(
    'ID' = ID, 'house_no' = ID %/% 4)))
  expect_error(p$set_demographic_household_table(data.frame(
    'ID' = ID, 'age_group' = ID %% 9)))
})

# Tests for Parameters::set_occupation_network_table are losely based off
# of examples/example_multiple_working_sectors.ipynb
test_that("Parameters::set_occupation_network_table (success)", {
  p <- Parameters$new(
    input_households = "data/baseline_household_demographics.csv",
    read_param_file = FALSE, read_hospital_param_file = FALSE)

  n_total <- 100L
  n_networks <- 10L

  network_no <- seq(from = 0, to = n_networks - 1, by = 1)
  network_name <- c('primary', 'secondary', 'sector_1', 'sector_2', 'sector_3',
                    'sector_4', 'sector_5', 'sector_6', 'retired', 'elderly')
  age_type <- c(0, 0, 1, 1, 1, 1, 1, 1, 2, 2)
  mean_work_interaction <- c(10, 10, 7, 7, 7, 7, 7, 7, 3, 3)
  lockdown_multiplier <- rep(0.22, n_networks)
  occupation_networks <- data.frame(
    'network_no' = network_no,
    'age_type' = age_type,
    'mean_work_interaction' = mean_work_interaction,
    'lockdown_multiplier' = lockdown_multiplier,
    'network_id' = network_no,
    'network_name' = network_name)

  ID <- seq(from = 0, to = n_total - 1, by = 1)
  assignment <- ID %% n_networks
  occupation_network_assignment <-
    data.frame('ID' = ID, 'network_no' = assignment)

  p$set_param('n_total', n_total)
  p$set_occupation_network_table(
    occupation_network_assignment, occupation_networks)
  # TODO(olegat) write proper test expectations:
  expect_true(TRUE)
})

test_that("Parameters::set_occupation_network_table (invalid n_total)", {
  p <- Parameters$new(
    input_households = "data/baseline_household_demographics.csv",
    read_param_file = FALSE, read_hospital_param_file = FALSE)
  networks <- data.frame('network_no' = c(0), 'age_type' = c(0),
    'mean_work_interaction' = c(10), 'lockdown_multiplier' = c(0.22),
    'network_id' = c(0), 'network_name' = c('primary'))
  assignments <- data.frame('ID' = c(0), 'network_no' = c(0))
  p$set_param('n_total', 10)
  expect_error(p$set_occupation_network_table(networks, assignments))
})

test_that("Parameters::return_param_object", {
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
  # Check that parameters are unlocked:
  p$set_param('n_total', 10)
  expect_equal(p$get_param('n_total'), 10)

  # Lock parameters:
  expect_equal(p$return_param_object(), p$c_params)

  # Check that parameters are locked:
  # TODO(olegat) find a way to lock write operations through '$<-'
  # expect_error(p$c_params$n_total <- 32)
  expect_error(p$set_param('n_total', 32))
  expect_equal(p$c_params$n_total, 10)
  expect_equal(p$get_param('n_total'), 10)
})
