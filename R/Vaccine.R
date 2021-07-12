#' R6Class Vaccine
#'
#' @description
#' Vaccine object has information about each new vaccine
#'
#' @examples
#' # Add new vaccine
#' vaccine = model$add_new_vaccine( )
#'
Vaccine <- R6Class( classname = 'Vaccine', cloneable = FALSE,

   private = list(
     #' the vaccine ID
     id        = NULL,

     #' .c_vaccine External pointer, reference to \code{vaccine} C struct.
     .c_vaccine = NULL,

     #' the C vaccine R pointer object
     c_vaccine_ptr = function() {
       return( self$c_vaccine@ref )
     },

     #' check the C vaccine still exists
     c_vaccine_valid = function() {
       return( !is_null_xptr( private$.c_vaccine@ref ))
     }
   ),

   active = list(
     #' @field c_vaccine the C vaccine R pointer object (SWIG wrapped)
     c_vaccine = function( val = NULL )
     {
       if( is.null( val ) )
       {
         if( private$c_vaccine_valid() )
           return( private$.c_vaccine )
         stop( "c_vaccine is no longer valid - create a new vaccine")
       }
       else
         stop( "cannot set c_vaccine" )
     }
   ),

   public = list(

     #' @param model R6 Model object
     #' @param stain_id The vaccine ID.
     initialize = function( model, vaccine_id )
     {
       if( !is.R6(model) || !inherits( model, "Model"))
         stop( "model must be a R6 class of type Model")

       private$.c_vaccine <- get_vaccine_by_id( model$c_model, vaccine_id )
       private$id         <- vaccine_id
     },

     #' @description Wrapper for C API \code{vaccine$idx()}.
     #' @return the index of the vaccine
     idx = function() {
       return(vaccine_idx( self$c_vaccine ))
     },

     #' @description Wrapper for C API \code{vaccine$n_strains()}.
     #' @return the number of strains the vaccine has efficacy for
     n_strain = function() {
       return(vaccine_n_strain( self$c_vaccine ))
     },

     #' @description Wrapper for C API \code{vaccine$time_to_protect()}.
     #' @return the time_to_protect of the vaccine
     time_to_protect = function() {
       return(vaccine_time_to_protect( self$c_vaccine ))
     },

     #' @description Wrapper for C API \code{vaccine$vaccine_protection_period()}.
     #' @return the vaccine_protection_period of the vaccine
     vaccine_protection_period = function() {
       return( vaccine_vaccine_protection_period( self$c_vaccine ))
     },

     #' @description Wrapper for C API \code{vaccine$full_efficacy()}.
     #' @return the full_efficacy of vaccine by strain
     full_efficacy = function() {
       return( .Call('R_vaccine_full_efficacy', private$c_vaccine_ptr(), PACKAGE='OpenABMCovid19') )
     },

     #' @description Wrapper for C API \code{vaccine$symptoms_efficacy()}.
     #' @return the symptoms_efficacy of vaccine by strain
     symptoms_efficacy = function() {
        return( .Call('R_vaccine_symptoms_efficacy', private$c_vaccine_ptr(), PACKAGE='OpenABMCovid19') )
     },

     #' @description Wrapper for C API \code{vaccine$severe_efficacy()}.
     #' @return the severe_efficacy of vaccine by strain
     severe_efficacy = function() {
        return( .Call('R_vaccine_severe_efficacy', private$c_vaccine_ptr(), PACKAGE='OpenABMCovid19') )
     }
   )
)
