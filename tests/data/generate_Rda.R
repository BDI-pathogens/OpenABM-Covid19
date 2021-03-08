# Load CSV files in "tests/data/" and save them to "data/baseline.rda"

baseline_household_demographics <-
    read.csv("tests/data/baseline_household_demographics.csv", header=TRUE)

baseline_parameters <-
    read.csv("tests/data/baseline_parameters.csv", header=TRUE)

hospital_baseline_parameters <-
    read.csv("tests/data/hospital_baseline_parameters.csv", header=TRUE)


dir.create("data")
save(
    baseline_household_demographics,
    baseline_parameters,
    hospital_baseline_parameters,
    file = "data/baseline.rda")
