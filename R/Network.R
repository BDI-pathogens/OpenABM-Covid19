SWIG_update_daily_fraction = update_daily_fraction

#' R6Class Network
#'
#' @description
#' Wrapper class for the \code{network} C struct (\emph{network.h}).
#'
#' Network object has information about a specific network.
#'
Network <- R6Class( classname = 'Network', cloneable = FALSE,

  private = list(
    #' the network ID
    id        = NULL,

    #' .c_network External pointer, reference to \code{network} C struct.
    .c_network = NULL,

    #' the C network R pointer object
    c_network_ptr = function() {
      return( private$c_network()@ref )
    },

    #' check the C network still exists
    c_network_valid = function() {
      return( !is_null_xptr( private$.c_network@ref ))
    }
  ),

  active = list(
    #' @field c_network the C network R pointer object (SWIG wrapped)
    c_network = function( val = NULL )
    {
      if( is.null( val ) )
      {
        if( private$c_network_valid() )
          return( private$.c_network )
        stop( "c_network is no longer valid - create a new netwrok")
      }
      else
        stop( "cannot set c_network" )
    }
  ),

  public = list(

    #' @param model R6 Model object
    #' @param network_id The network ID.
    initialize = function( model, network_id )
    {
      if( !is.R6(model) || !inherits( model, "Model"))
        stop( "model must be a R6 class of type Model")

      # check we're asking for a valid network
      n_ids = model$get_network_ids()
      if( !max( n_ids == network_id ) )
        stop( "network_id does not exist in this model")

      private$.c_network <- get_network_by_id( model$c_model, network_id )
      private$id         <- network_id
    },

    #' @description Wrapper for C API \code{network_n_edges}.
    #' @return Number of edges
    n_edges = function() {
      return(network_n_edges( self$c_network ))
    },

    #' @description Wrapper for C API \code{network_n_vertices}.
    #' @return Number of vertices
    n_vertices = function() {
      return(network_n_vertices( self$c_network ))
    },

    #' @description Wrapper for C API \code{network_name}.
    #' @return The network name
    name = function() {
      return(network_name( self$c_network ))
    },

    #' @description Wrapper for C API \code{get_network_by_id}
    #' @return The network ID
    network_id = function() {
      return(private$id)
    },

    #' @description Wrapper for C API \code{network_skip_hospitalised}.
    #' @return \code{TRUE} if interactions are skipped for hospitalised
    #' persons.
    skip_hospitalised = function() {
      return(network_skip_hospitalised( self$c_network ))
    },

    #' @description Wrapper for C API \code{network_skip_quarantined}.
    #' @return \code{TRUE} if interactions are skipped for quarantined
    #' persons.
    skip_quarantined = function() {
      return(network_skip_quarantined( self$c_network ))
    },

    #' @description Wrapper for C API \code{network_type}.
    #' @return Network type: 0 (household), 1 (occupation), or 2 (random)
    type = function() {
      return(network_type( self$c_network ))
    },

    #' @description Wrapper for C API \code{network_daily_fraction}.
    #' @return The fraction of edges on the network present each day (i.e.
    #' down-sampling the network). Value between 0 and 1.
    daily_fraction = function() {
      return(network_daily_fraction( self$c_network ))
    },

    #' @description Wrapper for C API \code{update_daily_fraction}.
    #' @param daily_fraction New fraction value; a value vetween 0 and 1.
    #' @return \code{TRUE} on success, \code{FALSE} on failure.
    update_daily_fraction = function(daily_fraction) {
      res <- SWIG_update_daily_fraction( self$c_network, daily_fraction )
      return(as.logical(res))
    },

    #' @description Print the object
    print = function() {
      pad <- function(x) { return(strtrim(paste(x,'                 '),17)) }
      keys <- c(
        'network_id', 'name','n_edges','n_vertices', 'skip_hospitalised',
        'skip_quarantined','type','daily_fraction'
      )
      for (k in keys)
      {
        v <- self[[k]]()
        cat(paste( pad(k), '=', v, '\n' ))
      }
    }
  )
)
