library(R6)

Number <- R6Class( inherit = Environment,
  public = list(
    number = NULL,

    start_simulation = function() {
      self$number <- c(number = 0)
      return(super$start_simulation())
    },

    step = function(action) {
      self$number <- action(self$number)
      return(c(number = self$number[[1]]))
    }
  )
)

Fibonacci <- R6Class( inherit = Agent,
  public = list(
    previous.number = NULL,

    initialize = function(...) {
      super$initialize(verbose...)
    },

    # action for fib(n=0)
    action.0 = function(...) {
      return(1)
    },

    # action for fib(n=1)
    action.1 = function(...) {
      self$previous.number <- 1
      return(1)
    },

    # action for fib(n)
    action.n = function(current.number) {
      new.number <- self$previous.number + current.number
      self$previous.number <- current.number
      return(new.number)
    },

    start_simulation = function(state) {
      self$previous.number <- 0
      return(self$action.0)
    },

    step = function(state) {
      if (self$previous.number == 0) {
        return(self$action.1)
      }else {
        return(self$action.n)
      }
    }
  )
)

test_that("Simulation:fibonacci", {
  sim <- Simulation$new( env = Number$new(), agent = Fibonacci$new() )

  sim$start_simulation()
  sim$steps(7)
  sim$end_simulation()
  expect_equal(sim$timestep, 7)
  expect_equal(sim$results, list(number = c(1, 1, 2, 3, 5, 8, 13)))
  expect_equal(sim$results_all_simulations, list())

  sim$start_simulation()
  sim$steps(3)
  sim$end_simulation()
  expect_equal(sim$timestep, 3)
  expect_equal(sim$results, list(number = c(1, 1, 2)))
  expect_equal(sim$results_all_simulations, list(
    list(number = c(1, 1, 2, 3, 5, 8, 13))
  ))

  sim$start_simulation()
  expect_equal(sim$timestep, 0)
  expect_equal(sim$results_all_simulations, list(
    list(number = c(1, 1, 2, 3, 5, 8, 13)),
    list(number = c(1, 1, 2))
  ))
})
