#' R6Class Strain
#'
#' @description
#' Strain object has information about each new strain
#'
#' @examples
#' # Add new strain with increased transmissibility
#' strain = model$add_new_strain( transmission_multiplier = 1.3 )
#'
Strain <- R6Class( classname = 'Strain', cloneable = FALSE,

   private = list(
     #' the strain ID
     id        = NULL,

     #' .c_strain External pointer, reference to \code{strain} C struct.
     .c_strain = NULL,

     #' the C strain R pointer object
     c_strain_ptr = function() {
       return( self$c_strain@ref )
     },

     #' check the C strain still exists
     c_strain_valid = function() {
       return( !is_null_xptr( private$.c_strain@ref ))
     }
   ),

   active = list(
     #' @field c_strain the C strain R pointer object (SWIG wrapped)
     c_strain = function( val = NULL )
     {
       if( is.null( val ) )
       {
         if( private$c_strain_valid() )
           return( private$.c_strain )
         stop( "c_strain is no longer valid - create a new strain")
       }
       else
         stop( "cannot set c_strain" )
     }
   ),

   public = list(

      #' @param model R6 Model object
      #' @param stain_id The strain ID.
      initialize = function( model, strain_id )
      {
         if( !is.R6(model) || !inherits( model, "Model"))
            stop( "model must be a R6 class of type Model")

         private$.c_strain <- get_strain_by_id( model$c_model, strain_id )
         private$id         <- strain_id
     },

      #' @description Wrapper for C API \code{strain$idx()}.
      #' @return the index of the strain
      idx = function() {
         return(strain_idx( self$c_strain ))
      },

      #' @description Wrapper for C API \code{strain$transmission_multiplier()}.
      #' @return the transmission_multiplier of the strain
      transmission_multiplier = function() {
         return(strain_transmission_multiplier( self$c_strain ))
      },

      #' @description Wrapper for C API \code{strain$total_infected()}.
      #' @return the total number of people infected with the strain
      total_infected = function() {
         return(strain_total_infected( self$c_strain ))
      },

      #' @description Wrapper for C API \code{strain$hospitalised_fraction()}.
      #' @return the hospitalised fraction for the strain
      hospitalised_fraction = function()
      {
         c_strain_ptr = private$c_strain_ptr();
         return( .Call('R_strain_hospitalised_fraction',c_strain_ptr,PACKAGE='OpenABMCovid19') )
      }
   )

)
