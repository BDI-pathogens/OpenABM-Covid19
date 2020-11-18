library(R6)

setwd("..")

test_that("Model::initialize (params_object isn't R6Class)", {
  expect_error(Model$new(10))
})

