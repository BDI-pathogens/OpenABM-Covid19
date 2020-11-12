# check if a double is a whole number, and convert it into an integer.
# returns NA_integer_ if number is NA.
make_integer <- function(number) {
  if (is.na(number)) {
    return(NA_integer_)
  }
  if (!is.numeric(number)) {
    stop(paste(number, "is not numeric"))
  }
  whole_part <- as.integer(number)
  fraction_part <- number - whole_part
  if (fraction_part != 0) {
    stop(paste(number,"is not a whole number"))
  }
  return(whole_part)
}



#' R6Class Parameters
#'
#' @description
#' Wrapper class for the \code{parameters} C struct (\emph{params.h}).
#'
#' @details
#' PLACEHOLDER Some explanations
#'
#' @examples
#' # Load parameters from CSV files.
#' params <- OpenABMCovid19::Parameters$initialize(
#'    "input_parameters.csv", 1,
#'    "out_dir",
#'    "input_household.csv",
#'    "hospital_input_parameters.csv", 1
#' )
#'
#' # Edit params
#' params$c_params$rng_seed = 1234
#' params$c_params$n_total = 250000
#'
Parameters <- R6Class(
  'Parameters',

  private = list(
    read_and_check_from_file = function()
    {
      read_param_file( self$c_params )
    },

    read_hospital_param_file = function()
    {
      read_hospital_param_file( self$c_params )
    },

    read_household_demographics = function()
    {
      if (self$c_params$N_REFERENCE_HOUSEHOLDS != 0) { return(); }

      if (is.data.frame(self$household_df)) {
        # Move data from R dataframe to C memory
        N <- nrow(self$household_df)
        self$c_params$N_REFERENCE_HOUSEHOLDS <- N
        set_up_reference_household_memory( self$c_params )

        for (i in 1:N) {
          add_household_to_ref_households(
            self$c_params,
            i - 1,
            self$household_df[i,1],
            self$household_df[i,2],
            self$household_df[i,3],
            self$household_df[i,4],
            self$household_df[i,5],
            self$household_df[i,6],
            self$household_df[i,7],
            self$household_df[i,8],
            self$household_df[i,9]
          )
        }
      }
      else {
        # Tell C API to load CSV.
        self$c_params$N_REFERENCE_HOUSEHOLDS <-
          nrow(read.csv(self$c_params$input_household_file))
        read_household_demographics_file(self$c_params)
      }
    },

    #' Convert the C param REFERENCE_HOUSEHOLDS (2D int array) into into an R
    #' data frame. Currently this function is used just for testing.
    get_REFERENCE_HOUSEHOLDS = function()
    {
      # TODO(olegat): get_param('REFERENCE_HOUSEHOLDS') could call this.
      # Create empty data frame with column names
      ages <- c(
        "a_0_9",
        "a_10_19",
        "a_20_29",
        "a_30_39",
        "a_40_49",
        "a_50_59",
        "a_60_69",
        "a_70_79",
        "a_80")

      # Add rows
      N <- self$c_params$N_REFERENCE_HOUSEHOLDS
      mat <- matrix(nrow = N, ncol = 9)
      for (i in 1:N) {
        offset <- i - 1
        mat[i,] <- c(
          get_household_value( self$c_params, offset, 0 ),
          get_household_value( self$c_params, offset, 1 ),
          get_household_value( self$c_params, offset, 2 ),
          get_household_value( self$c_params, offset, 3 ),
          get_household_value( self$c_params, offset, 4 ),
          get_household_value( self$c_params, offset, 5 ),
          get_household_value( self$c_params, offset, 6 ),
          get_household_value( self$c_params, offset, 7 ),
          get_household_value( self$c_params, offset, 8 ))
      }

      # Convert matrix into data.frame
      df <- as.data.frame(mat)
      names(df) <- ages
      return(df)
    }
  ),

  public = list(
    #' @field c_params SWIG pointer to C API struct.
    c_params = NA,

    #' @field household_df Household Data Frame.
    household_df = NA,

    #' @param input_param_file Input parameters CSV file path.
    #' @param param_line_number Which column of the input param file to read.
    #' @param output_file_dir Where to write output files to.
    #' @param input_households Household demographics file (required).
    #' @param hospital_input_param_file Hospital input parameters CSV file path.
    #' @param hospital_param_line_number Which column of the hospital input param file to read.
    #' @param read_param_file A boolean. If \code{TRUE}, read \code{input_param_file}. If \code{FALSE}, ignore \code{input_param_file}.
    #' @param read_hospital_param_file A boolean. If \code{TRUE}, read \code{hospital_input_param_file}. If \code{FALSE}, ignore \code{hospital_input_param_file}.
    initialize = function(
      input_param_file = NA_character_,
      param_line_number = NA_integer_,
      output_file_dir = "./",
      input_households = NA_character_,
      hospital_input_param_file = NA_character_,
      hospital_param_line_number = NA_integer_,
      read_param_file = TRUE,
      read_hospital_param_file = FALSE )
    {
      ## Convert doubles to integers
      param_line_number <- make_integer(param_line_number)
      hospital_param_line_number <- make_integer(hospital_param_line_number)

      ## Initialize the C struct
      self$c_params <- parameters()
      initialize_params( self$c_params )

      if (!is.na(input_param_file)) {
        self$c_params$input_param_file <- input_param_file
      } else if (is.na(input_param_file) && read_param_file ) {
        stop("input_param_file is NA and read_param_file set to TRUE")
      }

      if (!is.na(param_line_number)) {
        self$c_params$param_line_number <- param_line_number
      }

      if (is.character(input_households)) {
        self$c_params$input_household_file <- input_households
        self$household_df = NA
      } else if (is.data.frame(input_households)) {
        self$household_df <- input_households
      } else {
        stop("Household data must be supplied as a CSV")
      }

      if (!is.na(hospital_param_line_number)) {
        if (is.integer(hospital_param_line_number)) {
          self$c_params$hospital_param_line_number <- hospital_param_line_number
        } else {
          stop("hospital_param_line_number must be an integer or NA")
        }
      }

      if (!is.na(hospital_input_param_file) && read_hospital_param_file) {
        self$c_params$hospital_input_param_file <- hospital_input_param_file
      } else if (is.na(hospital_input_param_file) && read_hospital_param_file) {
        stop("hospital_input_param_file is NA and read_param_file is TRUE")
      }

      if (read_hospital_param_file) {
        private$read_hospital_param_file()
      }

      if (read_param_file) {
        private$read_and_check_from_file()
      }

      self$c_params$output_file_dir = output_file_dir
      if (!is.na(output_file_dir)) {
        self$c_params$sys_write_individual <- 1
      }
    },

    #' Get a C parameter by name.
    #' @param param A string representing the C parameter's name
    get_param = function(param)
    {
      getter <- get( paste("parameters_", param, "_get", sep = "") )
      result <- getter( self$c_params )
      return(result)
    },

    #' Set a C parameter by name.
    #' @param param A string representing the C parameter's name
    #' @param value The new value for the C parameter.
    set_param = function(param, value)
    {
      setter <- get( paste("parameters_", param, "_set", sep = "") )
      setter( self$c_params, value )
    }
  )
)



#' R6Class Model
#'
#' @description
#' Wrapper class for the \code{model} C struct (\emph{model.h}).
#'
Model <- R6Class(
  'Model',

  public = list(

    #' @param example PLACEHOLDER
    initialize = function(example)
    {
    }
  )
)



