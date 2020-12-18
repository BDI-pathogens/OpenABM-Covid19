SWIG_update_daily_fraction = update_daily_fraction

#' R6Class Network
#'
#' @description
#' Wrapper class for the \code{network} C struct (\emph{network.h}).
#'
#' Network object has information about a specific network.
#'
Network <- R6Class( classname = 'Network', cloneable = FALSE,

  private = list( id = NULL ),

  public = list(
    #' @field External pointer, reference to \code{network} C struct.
    c_network = NULL,

    #' @param c_model External pointer, reference to \code{model} C struct.
    #' @param network_id The network ID.
    initialize = function(c_model, network_id)
    {
      # TODO(olegat) check for NULL c_network pointer here (R can crash)
      self$c_network <- get_network_by_id( c_model, network_id )
      private$id     <- network_id
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
    #' @return \code{TRUE} on success, \code{FALSE} on failure.
    update_daily_fraction = function(daily_fraction) {
      res <- SWIG_update_daily_fraction( self$c_network, daily_fraction )
      return(as.logical(res))
    },

    #' @description Print the object
    show = function() {
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
