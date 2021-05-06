# TODO(olegat) write tests for: Environment, Agent, Simulation, COVID19IBM

#' R6Class Environment
#' @description
#' Class representing an environment object that defines the system.
#'
#' See \code{\link{Simulation}} for examples.
Environment <- R6Class( classname = 'Environment', cloneable = FALSE,
  private = list( .start = FALSE ),

  public = list(
    #' @param verbose Log verbosity.
    initialize = function(verbose = FALSE) {
    },

    #' @description Method called at the start of each simulation
    #' @param action Action to perform.
    #' @return A vector in the format \code{c(reward=R,next_start=N)}.
    step = function(action) {
      return(list(reward = NA, next_state = NA))
    },

    #' @description End the simulation.
    end_simulation = function() {},

    #' @description Initialize the Environment object for the start of a
    #' simulation.
    #' @return Returns the starting state.
    start_simulation = function() {
      return(private$.start)
    },

    #' @description Get the starting state.
    #' @return Returns the starting state.
    start = function() {
      return(private$.start)
    }
  )
)

#' R6Class Agent
#' @description
#' Class representing an Agent object for dictating policy through time and
#' storing value function (that maps states of the model to actions).
#'
#' See \code{\link{Simulation}} for examples.
Agent <- R6Class( classname = 'Agent', cloneable = FALSE,
  public = list(
    #' @param verbose Log verbosity.
    initialize = function(verbose = FALSE) {
    },

    #' @description Initialize the Agent object for the start of a simulation/
    #' @param state Initial state.
    #' @return An action.
    start_simulation = function(state) {
      return(c())
    },

    #' @description Method called at the start of each simulation
    #' @param state New state.
    #' @return Should return an action (current an empty dict).
    step = function(state) {
      return(c())
    }
  )
)

#' R6Class Simulation
#' @description Simulation object to run the model and store data across
#' multiple simulations.
#' @examples
#' # Create a model using the baseline parameters included in the package.
#' # Note: This initialisation can take a few seconds.
#' model <- Model.new( params = list( n_total = 10000 ) )
#'
#' if (!is.null(model)) {
#'   # Begin simulation:
#'   env <- COVID19IBM$new( model )
#'   sim <- Simulation$new( env )
#'   sim$start_simulation()
#'   sim$steps(1) # Note: slow operation (takes a few seconds).
#'
#'   # Make changes to the model (environment)
#'   model$seed_infect_by_idx(0)
#'   model$vaccinate_individual(2)
#'   model$vaccinate_schedule( VaccineSchedule$new() )
#'
#'   # Resume simulation
#'   sim$steps(1) # Note: slow operation (takes a few seconds).
#'
#'   # Get results
#'   sim$results
#' }
Simulation <- R6Class( classname = 'Simulation', cloneable = FALSE,
  public = list(
    #' @field env Current environment
    #' (\code{\link{Environment}} R6 class object)
    env = NULL,

    #' @field agent Current agent
    #' (\code{\link{Agent}} R6 class object)
    agent = NULL,

    #' @field current_state Current environment state.
    current_state = NULL,

    #' @field current_action Current action from agent.
    current_action = NULL,

    #' @field simulation_number Simulation Number.
    simulation_number = NULL,

    #' @field end_time Time when the simulation should end.
    end_time = NULL,

    #' @field results Results of current simulation (list of vectors).
    results = NULL,

    #' @field results_all_simulations Results of all simulations
    #' (list of list of vectors).
    results_all_simulations = list(),

    #' @field verbose Log verbosity.
    verbose = NULL,

    #' @field timestep Current time step.
    timestep = 0,

    #' @field sim_started \code{TRUE} is the simulation has started,
    #' \code{FALSE} otherwise.
    sim_started = FALSE,

    #' @description Environment subclass representing a COVID19 outbreak as
    #' defined in the COVID19-IBM model
    #' @param env Instance of \code{\link{Environment}} R6Class
    #' @param agent Instance of \code{\link{Agent}} R6Class
    #' @param end_time End time for the simulation
    #' @param verbose Log verbosity.
    initialize = function(env, agent = Agent$new(), end_time = NULL,
                          verbose = FALSE)
    {
      self$env      <- env
      self$agent    <- agent
      self$end_time <- end_time
      self$verbose  <- verbose
    },

    #' @description Initialisation of the simulation; reset the model
    start_simulation = function() {
      if (self$verbose) {
        cat("Starting simulation\n")
      }

      # Set the current time step
      self$timestep <- 0

      # Reset the model, fixme - add method for initialising the model
      self$current_state <- self$env$start_simulation()

      # Reset the agent
      self$current_action <- self$agent$start_simulation(self$current_state)

        # Append current state
      if (length(self$results) > 0) {
        n <- length(self$results_all_simulations)
        self$results_all_simulations[[n + 1]] <- self$results
      }

      self$results <- list()
      self$sim_started <- TRUE
    },

    #' @description End the simulation
    end_simulation = function() {
      if (self$sim_started) {
        if (self$verbose) {
          cat("Ending simulation\n")
        }
        self$sim_started <- FALSE
        self$env$end_simulation()
      }
    },

    #' @description Run the model for a specific number of steps, starting from
    #' the current state, save data as model progresses.
    #' @param n_steps Number of steps for which to call
    #' \code{self$model.one_time_step()}
    steps = function(n_steps) {
      if (!self$sim_started) {
        self$start_simulation()
      }

      for (ts in 1:n_steps) {
        if (self$verbose) {
          cat("Current timestep: ", self$timestep, "\n")
        }

        next_state  <- self$env$step(self$current_action)
        next_action <- self$agent$step(next_state)

        # Save the state of the model
        self$collect_results(next_state, next_action)

        if (is.null(self$end_time) || self$timestep < self$end_time) {
          self$current_state <- next_state
          self$current_action <- next_action
          self$timestep <- (self$timestep + 1)
        } else {# if at the end_time of the model then exit
          self$end_simulation()
          if (self$verbose) {
            cat("Reached end time of simulation before completing all steps\n")
          }
          break
        }
      }
    },

    #' @description Collect model results at each step;
    #' fixme action is not currently stored
    #' @param state The state to collect results from.
    #' @param action Currently unused.
    collect_results = function(state, action) {
      # Save results to a named vector
      for (i in 1:length(state)) {
        key   <- names(state[i])
        value <- state[[i]]
        self$results[[key]] <- append(self$results[[key]], value)
      }
    },

    #' @description Get terminal state.
    #' @return Is the current state the terminal state
    is_terminal_state = function() { return(FALSE) }
  )
)

#' R6Class COVID19IBM
#' @description Environment subclass representing a COVID19 outbreak as defined
#' in the COVID19-IBM model.
#'
#' See \code{\link{Simulation}} for examples.
COVID19IBM <- R6Class( classname = 'COVID19IBM', cloneable = FALSE,
  inherit = Environment,

  public = list(
    #' @field model An instance of the \code{\link{Model}} class.
    model = NULL,

    #' @description Environment subclass representing a COVID19 outbreak as
    #' defined in the COVID19-IBM model
    #' @param model An R6 \code{\link{Model}} class instance.
    #' @param ... Parameters to pass to Environment super class.
    initialize = function(model, ...) {
      self$model = model
      super$initialize(...)
    },

    #' @description Start a simulation.
    #' @return The state of the system
    start_simulation = function() {
      return(super$start_simulation())
    },

    #' @description End the simulation. Destroy the model.
    end_simulation = function() {
      # R's garbage collector will call SWIG's destructor
      self$model = NULL
    },

    #' @description Run the simulation through one time step.
    #' @param action Vector of actions. The vector should contain names from
    #' \code{\link{SAFE_UPDATE_PARAMS}}
    #' @return The state of the system.
    step = function(action) {
      # If the action is non-empty, then update model parameters in the simulation
      if (is.vector(action)) {
        for (i in 1:length(action)) {
          param <- names(action[i])
          value <- action[[i]]
          self$model$update_running_params(param, value)
        }
      }

      self$model$one_time_step()
      return(self$model$one_time_step_results())
    }
  )
)

