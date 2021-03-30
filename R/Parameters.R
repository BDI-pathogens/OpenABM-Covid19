SWIG_set_occupation_network_table <- set_occupation_network_table

#' R6Class Parameters
#'
#' @description
#' Wrapper class for the \code{parameters} C struct (\emph{params.h}).
#'
#' @details
#' For a detailed explanation of the available parameters (including sources
#' and references), please read the
#' \href{https://github.com/BDI-pathogens/OpenABM-Covid19/blob/master/documentation/parameters/parameter_dictionary.md}{Online Documentation}.
#'
#' @seealso \code{\link{Model}}
#'
Parameters <- R6Class( classname = 'Parameters', cloneable = FALSE,

  private = list(
    update_lock = FALSE,

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
      if (self$get_param( "N_REFERENCE_HOUSEHOLDS" ) != 0) { return(); }

      c_params = self$c_params
      if (is.data.frame(self$household_df)) {
        # Move data from R dataframe to C memory
        N <- nrow(self$household_df)
        c_params$N_REFERENCE_HOUSEHOLDS <- N
        set_up_reference_household_memory( c_params )

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
        c_params$N_REFERENCE_HOUSEHOLDS <-
          nrow(read.csv(self$get_param( "input_household_file" ) ) )
        read_household_demographics_file(self$c_params)
      }
    },

    # Convert the C param REFERENCE_HOUSEHOLDS (2D int array) into into an R
    # data frame. Currently this function is used just for testing.
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
      N <- self$get_param( "N_REFERENCE_HOUSEHOLDS" )
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
    },

    .c_params   = NA,

    #' the C params R pointer object
    c_params_ptr = function() {
      return( private$.c_params()@ref )
    },

    #' check the C params still exists
    c_params_valid = function() {
      return( !is_null_xptr( private$.c_params@ref ))
    }
  ),

  active = list(
    #' @field c_params the C param sR pointer object (SWIG wrapped)
    c_params = function( val = NULL )
    {
      if( is.null( val ) )
      {
        if( private$c_params_valid() )
          return( private$.c_params )
        stop( "c_paramsis no longer valid - create a new set of Parameters")
      }
      else
        stop( "cannot set c_params" )
    }
  ),


  public = list(

    #' @field household_df Household Data Frame.
    household_df = NA,

    #' @param input_param_file Input parameters CSV file path.
    #' Optional, default:
    #' \code{system.file("default_params", "baseline_parameters.csv", package = "OpenABMCovid19")}
    #' @param param_line_number Which column of the input param file to read.
    #' @param output_file_dir Where to write output files to.
    #' @param input_households Household demographics file.
    #' Optional, default:
    #' \code{system.file("default_params", "baseline_household_demographics.csv", package = "OpenABMCovid19")}
    #' @param hospital_input_param_file Hospital input parameters CSV file path.
    #' Optional, default:
    #' \code{system.file("default_params", "hospital_baseline_parameters.csv", package = "OpenABMCovid19")}
    #' @param hospital_param_line_number Which column of the hospital input
    #' param file to read.
    #' @param read_param_file A boolean. If \code{TRUE}, read
    #' \code{input_param_file}. If \code{FALSE}, ignore
    #' \code{input_param_file}.
    #' @param read_hospital_param_file A boolean. If \code{TRUE}, read
    #' \code{hospital_input_param_file}. If \code{FALSE}, ignore
    #' \code{hospital_input_param_file}.
    initialize = function(
      input_param_file = NA_character_,
      param_line_number = 1,
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
      c_params = parameters()
      private$.c_params <- c_params
      initialize_params( self$c_params )

      # if no input_param file is given then use the default file
      if ( is.na(input_param_file)) {
        input_param_file <- system.file("default_params",
                        "baseline_parameters.csv", package = "OpenABMCovid19")
      }
      if (read_param_file) {
        if (!is.character(input_param_file) || !file.exists(input_param_file)) {
          stop("input_params_file must be a valid file name OR NA (default params)")
        }
        c_params$input_param_file <- input_param_file
      }

      if (!is.na(param_line_number)) {
        c_params$param_line_number <- param_line_number
      }

      if (is.data.frame(input_households)) {
        self$household_df <- input_households
      }
      else {
        if ( is.na(input_households)) {
          if(.Machine$sizeof.pointer < 8) {
            stop("Default household data unsupported on 32-bit systems (model requires more that 4 GiB).")
          }
          else {
            input_households <- system.file("default_params",
              "baseline_household_demographics.csv", package = "OpenABMCovid19")
          }
        }
        if (!is.character(input_households) || !file.exists(input_households)) {
          stop("input_households must be a data.frame of household OR a valid file OR left NA (default params)")
        }
        c_params$input_household_file <- input_households
        self$household_df <- NA
      }

      if (!is.na(hospital_param_line_number)) {
        if (is.integer(hospital_param_line_number)) {
          c_params$hospital_param_line_number <- hospital_param_line_number
        } else {
          stop("hospital_param_line_number must be an integer or NA")
        }
      }

      if (is.na(hospital_input_param_file)) {
        hospital_input_param_file <- system.file("default_params",
               "hospital_baseline_parameters.csv", package = "OpenABMCovid19")
      }

      if (read_hospital_param_file) {
        if (!is.character(hospital_input_param_file) || !file.exists(hospital_input_param_file )) {
          stop("if read_param_file is TRUE then hospital_input_param_file must be a valid file or NA (default params)")
        }
        c_params$hospital_input_param_file <- hospital_input_param_file
      }

      if (read_hospital_param_file) {
        private$read_hospital_param_file()
      }

      if (read_param_file) {
        private$read_and_check_from_file()
      }

      c_params$output_file_dir = output_file_dir
      if (!is.na(output_file_dir)) {
        c_params$sys_write_individual <- 1
      }
    },

    #' @description Get a C parameter by name.
    #' @param param A string representing the C parameter's name
    get_param = function(param)
    {
      enum <- get_base_param_from_enum(param)
      if (!is.null(enum)) {
        # multi-value parameter (C array)
        getter <- get(paste0("get_param_", enum$base_name))
        result <- getter( self$c_params, enum$index )
      } else {
        # single-value parameter
        getter <- get( paste0("parameters_", param, "_get") )
        result <- getter( self$c_params )
      }
      return(result)
    },

    #' @description Set a C parameter by name.
    #' @param param A string representing the C parameter's name
    #' @param value The new value for the C parameter.
    set_param = function(param, value)
    {
      if (private$update_lock) {
        stop("Parameters locked, please use Model$update_x functions")
      }

      enum <- get_base_param_from_enum(param)
      if (!is.null(enum)) {
        # multi-value parameter (C array)
        setter <- get(paste0("set_param_", enum$base_name))
        setter( self$c_params, value, enum$index )
      } else {
        # single-value parameter
        setter <- get( paste0("parameters_", param, "_set") )
        setter( self$c_params, value )
      }
    },

    #' @description Set C parameters from a list.
    #' @param params A named list with the name being the parameter
    set_param_list = function(params) {
      for( param in names(params) )
        self$set_param(param,params[[param]])
    },

    #' @description
    #' Set the \code{demographic_household_table} C struct (defined in
    #' \emph{demographics.h}). This function initializes the
    #' \code{self$c_params$demo_house} member.
    #' @param df_demo_house A data-frame representation of the
    #' \code{demographic_household_table} C struct. The data-frame must contain
    #' column names \code{c("ID", "age_group", "house_no")} and the number of
    #' rows must be equal to \code{self$c_params$n_total}.
    #' @return \code{TRUE} on success, \code{FALSE} on error.
    set_demographic_household_table = function(df_demo_house)
    {
      n_total <- nrow(df_demo_house)
      if (n_total != self$get_param( "n_total" ) ) {
        stop('df_demo_house must have n_total rows')
      }
      for (name in c('ID', 'age_group', 'house_no')) {
        if ( !hasName(df_demo_house, name)) {
          stop('df_demo_house must have column ', name)
        }
      }
      n_households <- max(df_demo_house[,'house_no']) + 1

      C_result <- set_demographic_house_table(
        self$c_params,
        n_total,
        n_households,
        df_demo_house[,'ID'],
        df_demo_house[,'age_group'],
        df_demo_house[,'house_no'])

      if (C_result == 0) {
        stop("C API set_demographic_house_table failed (returned FALSE)")
      }
    },

    #' @description
    #' Set the \code{demographic_occupation_network_table} C struct (defined in
    #' \emph{demographics.h}). This function initializes the
    #' \code{self$c_params$occupation_network_table} member.
    #' @param df_occupation_networks TODO(olegat)
    #' @param df_occupation_network_properties TODO(olegat)
    set_occupation_network_table = function(
      df_occupation_networks,
      df_occupation_network_properties)
    {
      n_total <- nrow(df_occupation_networks)
      if (n_total != self$get_param( "n_total" ) ) {
        stop('df_occupation_networks must have n_total rows')
      }

      # C memory alloc
      n_networks <- max(df_occupation_networks['network_no']) + 1
      SWIG_set_occupation_network_table( self$c_params, n_total, n_networks )

      # Write properties to C struct
      for (row in 1:nrow(df_occupation_network_properties)) {
        C_result <- set_indiv_occupation_network_property(
          self$c_params,
          df_occupation_network_properties[row,'network_no'],
          df_occupation_network_properties[row,'age_type'],
          df_occupation_network_properties[row,'mean_work_interaction'],
          df_occupation_network_properties[row,'lockdown_multiplier'],
          df_occupation_network_properties[row,'network_id'],
          df_occupation_network_properties[row,'network_name'])
        if (C_result == 0) {
          stop("C API set_indiv_occupation_network_property failed (returned FALSE)")
        }
      }

      # Write network assignment to C struct
      C_result <- set_indiv_occupation_network(
        self$c_params,
        n_total,
        df_occupation_networks[,'ID'],
        df_occupation_networks[,'network_no'])
      if (C_result == 0) {
        stop("C API set_indiv_occupation_network failed (returned FALSE)")
      }
    },

    #' @description
    #' Run a check on the parameters and return if the C code doesn't bail.
    #' This function locks the parameter value (i.e. make this class
    #' read-only))
    #' @return \code{self$c_params}
    return_param_object = function() {
      private$read_household_demographics()
      check_params(self$c_params)

      private$update_lock <- TRUE
      # TODO(olegat) find a way to lock write operations through '$<-'
      # e.g. self$c_params$n_total <- 10
      #setMethod('$<-', '_p_parameters', function(x, name, value) {
      #  stop("Parameters locked, please use Model$update_x functions")
      #})

      return(self$c_params)
    }
  )
)

############################################################################
#
# Wrapper functions for R users who are not used to using classes
#
############################################################################

#' Creates a new OpenABM Parameters object (wrapper for
#' \code{\link{Parameters}$new()})
#'
#' @param input_param_file Input parameters CSV file path.
#' Optional, default:
#' \code{system.file("default_params", "baseline_parameters.csv", package = "OpenABMCovid19")}
#' @param param_line_number Which column of the input param file to read.
#' @param output_file_dir Where to write output files to.
#' @param input_households Household demographics file.
#' Optional, default:
#' \code{system.file("default_params", "baseline_household_demographics.csv", package = "OpenABMCovid19")}
#' @param hospital_input_param_file Hospital input parameters CSV file path.
#' Optional, default:
#' \code{system.file("default_params", "hospital_baseline_parameters.csv", package = "OpenABMCovid19")}
#' @param hospital_param_line_number Which column of the hospital input
#' param file to read.
#' @param read_param_file A boolean. If \code{TRUE}, read
#' \code{input_param_file}. If \code{FALSE}, ignore
#' \code{input_param_file}.
#' @param read_hospital_param_file A boolean. If \code{TRUE}, read
#' \code{hospital_input_param_file}. If \code{FALSE}, ignore
#' \code{hospital_input_param_file}.
#' @seealso
#' \href{https://github.com/BDI-pathogens/OpenABM-Covid19/blob/master/documentation/parameters/parameter_dictionary.md}{Online Documentation}.
#' @return Parameters object (R6 Class)
Parameters.new = function(
  input_param_file = NA_character_,
  param_line_number = 1,
  output_file_dir = "./",
  input_households = NA_character_,
  hospital_input_param_file = NA_character_,
  hospital_param_line_number = NA_integer_,
  read_param_file = TRUE,
  read_hospital_param_file = FALSE
) {
  return(Parameters$new(input_param_file,param_line_number,output_file_dir,
    input_households, hospital_input_param_file, hospital_param_line_number,
    read_param_file, read_hospital_param_file))
}

#' Gets the value of a parameter (wrapper for
#' \code{\link{Parameters}$get_param(param)})
#' @param parameters A Parameters object
#' @param param The name of the parameter
#' @seealso
#' \href{https://github.com/BDI-pathogens/OpenABM-Covid19/blob/master/documentation/parameters/parameter_dictionary.md}{Online Documentation}.
Parameters.get_param = function(parameters,param) {
  return( parameters$get_param(param))
}

#' Sets the value of a parameter (wrapper for
#' \code{\link{Parameters}$set_param(param,value)})
#' @param parameters A Parameters object
#' @param param The name of the parameter
#' @param value The new value
#' @seealso
#' \href{https://github.com/BDI-pathogens/OpenABM-Covid19/blob/master/documentation/parameters/parameter_dictionary.md}{Online Documentation}.
Parameters.set_param = function(parameters,param,value) {
  return( parameters$set_param(param,value))
}

