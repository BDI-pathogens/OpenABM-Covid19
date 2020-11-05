

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



Parameters <- R6Class(
  'Parameters',

  private = list(
    read_and_check_from_file = function()
    {
      read_param_file( self$c_params )
    },

    read_hospital_param_file = function()
    {
      # TODO
    }
  ),

  public = list(
    c_params = NA,

    household_df = NA,

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
        stop("input_param_file is NA and read_param_file set to TRUE/")
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

      if (!is.na(hospital_input_param_file) && read_param_file) {
        self$c_params$hospital_input_param_file <- hospital_input_param_file
      } else if (is.na(hospital_input_param_file) && read_param_file) {
        stop("hospital_input_param_file is NA and read_param_file is TRUE")
      }

      if (read_hospital_param_file) {
        private$read_hospital_param_file()
      }

      if (read_param_file) {
        private$read_and_check_from_file()
      }

      if (!is.na(output_file_dir)) {
        self$c_params$sys_write_individual <- 1
      }
    }
  )
)



