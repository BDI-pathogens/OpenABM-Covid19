#' R6Class Model
#'
#' @description
#' Wrapper class for the \code{model} C struct (\emph{model.h}).
#'
#' @details
#' TODO(olegat) PLACEHOLDER Some explanations
#'
#' @seealso AgeGroups
Model <- R6Class( classname = 'Model', cloneable = FALSE,

  private = list(
    params_object = NA,

    is_running = FALSE,

    nosocomial = FALSE,

    c_params = NA,

    c_model = NA),

  public = list(
    #' @param params_object An object of type \code{Parameters}. The
    #' constructor will lock the parameter values (ie. \code{params_code}
    #' will become read-only).
    initialize = function(params_object)
    {
      if (!is.R6(params_object)) {
        stop("params_object is an a Parameters R6Class")
      }
      # Store the params object so it doesn't go out of scope and get freed
      private$params_object <- params_object
      # Create C parameters object
      private$c_params   <- params_object$return_param_object()
      private$c_model    <- create_model(private$c_params)
      private$nosocomial <- as.logical(self$get_param('hospital_on'))
    },

    #' @description Destroy the C struct
    finalize = function()
    {
      destroy_model(private$c_model)
    },

    #' @description Get a parameter value by name
    #' @param name of param
    #' @return value of stored param
    get_param = function(name)
    {
      # TODO(olegat)
    },

    #' @description
    #' A subset of parameters may be updated whilst the model is evaluating,
    #' these correspond to events. This function throws an error if
    #' \code{param} isn't safe to update.
    #' @param param name of parameter
    #' @param value value of parameter
    update_running_params = function(param, value)
    {
      # TODO(olegat)
    },

    #' @description Gets the value of the risk score parameter.
    #' Wrapper for C API \code{get_model_param_risk_score}.
    #' @param day Infection day
    #' @param age_inf Infector age group index, value between 0 and 8.
    #' See AgeGroups list
    #' @param age_sus Susceptible age group index, value between 0 and 8.
    #' See AgeGroups list
    #' @return The risk value.
    get_risk_score = function(day, age_inf, age_sus)
    {
      value <- get_model_param_risk_score(
        private$c_model, day, age_inf, age_sus)
      if (value < 0) {
        stop( "Failed to get risk score")
      }
      return(value)
    },

    #' @description Gets the value of the risk score household parameter.
    #' Wrapper for C API \code{get_model_param_risk_score_household}.
    #' @param age_inf Infector age group index, value between 0 and 8.
    #' See AgeGroups list
    #' @param age_sus Susceptible age group index, value between 0 and 8.
    #' See AgeGroups list
    #' @return The risk value.
    get_risk_score_household = function(age_inf, age_sus)
    {
      value <- get_model_param_risk_score_household(
        private$c_model, age_inf, age_sus)
      if (value < 0) {
        stop( "Failed to get risk score household")
      }
      return(value)
    },

    #' @description
    #' Gets the value of the risk score parameter.
    #' Wrapper for C API \code{set_model_param_risk_score}.
    #' @param day Infection day
    #' @param age_inf Infector age group index, value between 0 and 8.
    #' See AgeGroups list
    #' @param age_sus Susceptible age group index, value between 0 and 8.
    #' See AgeGroups list
    #' @param value The risk value
    set_risk_score = function(day, age_inf, age_sus, value)
    {
      ret <- set_model_param_risk_score(
        private$c_model, day, age_inf, age_sus, value)
      if (ret == 0) {
        stop( "Failed to set risk score")
      }
    },

    #' @description
    #' Gets the value of the risk score household parameter.
    #' Wrapper for C API \code{set_model_param_risk_score_household}.
    #' @param age_inf Infector age group index, value between 0 and 8.
    #' See AgeGroups list
    #' @param age_sus Susceptible age group index, value between 0 and 8.
    #' See AgeGroups list
    #' @param value The risk value.
    set_risk_score_household = function(age_inf, age_sus, value)
    {
      ret <- set_model_param_risk_score_household(
        private$c_model, age_inf, age_sus, value)
      if (ret == 0) {
        stop( "Failed to set risk score household")
      }
    },

    #' @description
    #' Creates a new user user network (destroy an old one if it exists).
    #' Wrapper for C API \code{add_user_network}.
    #' @param df_network Network data frame.
    #' @param interaction_type Must 0 (household), 1 (occupation), or 2 (random)
    #' @param skip_hospitalised If TRUE, skip hospitalisation
    #' @param skip_quarantine If TRUE, skip quarantine.
    #' @param daily_fraction Value between 0 and 1.
    #' @param name Name of the network.
    add_user_network = function(
      df_network,
      interaction_type = 1,
      skip_hospitalised = TRUE,
      skip_quarantine = TRUE,
      daily_fraction = 1.0,
      name = "user_network")
    {
      # TODO(olegat)
    },

    #' @description Get all app users. Wrapper for C API \code{get_app_user}.
    #' @return All app users.
    get_app_users = function()
    {
      # TODO(olegat)
    },

    #' @description Sets specific users to have or not have the app.
    #' Wrapper for C API \code{get_app_user}. Throws error on failure.
    #' @param df_app_users TODO(olegat) PLACEHOLDER
    set_app_users = function()
    {
      # TODO(olegat)
    },


    #' @description Move the model through one time step.
    #' Wrapper for C API \code{one_time_step}.
    one_time_step = function()
    {
      # TODO(olegat)
    },

    #' @description Get the results from one-time step.
    #' @return A data frame with 1 row (i.e. dictionary).
    one_time_step_result = function()
    {
      # TODO(olegat)
    },

    #' @description Write output files.
    #' Wrapper for C API \code{write_output_files}.
    write_output_files = function()
    {
      # TODO(olegat)
    },

    #' @description Write output files
    #' Wrapper for C API \code{write_individual_file}.
    write_individual_file = function()
    {
      # TODO(olegat)
    },

    #' @description Wrapper for C API \code{write_interactions}.
    write_interactions_file = function()
    {
      # TODO(olegat)
    },

    #' @description Wrapper for C API \code{write_trace_tokens_ts}.
    write_trace_tokens_timeseries = function()
    {
      # TODO(olegat)
    },

    #' @description Wrapper for C API \code{write_trace_tokens}.
    write_trace_tokens = function()
    {
      # TODO(olegat)
    },

    #' @description Wrapper for C API \code{write_transmissions}.
    write_transmissions = function()
    {
      # TODO(olegat)
    },

    #' @description Wrapper for C API \code{write_quarantine_reasons}.
    write_quarantine_reasons = function()
    {
      # TODO(olegat)
    },

    #' @description Wrapper for C API \code{write_occupation_network}.
    #' @param idx Network index.
    write_occupation_network = function(idx)
    {
      # TODO(olegat)
    },

    #' @description Wrapper for C API \code{write_household_network}.
    write_household_network = function()
    {
      # TODO(olegat)
    },

    #' @description Wrapper for C API \code{write_random_network}.
    write_random_network = function()
    {
      # TODO(olegat)
    },

    #' @description Wrapper for C API \code{print_individual}.
    #' @param idx Individual index.
    print_individual = function(idx)
    {
      # TODO(olegat)
    }
  )
)
