
MetaModel <- R6Class( classname = 'MetaModel', cloneable = FALSE,

  private = list(
    .n_nodes     = NULL,
    .n_regions   = NULL,
    .node_list   = NULL,
    .clusterObj  = NULL,
    .base_params = NULL,

    .staticReturn = function( val, name )
    {
      if( !is.null( val ) )
        stop( sprintf( "cannot set %s", name ) )

      privateName <- sprintf( ".%s", name )
      return( private[[ privateName ]])
    },

    .cluster = function()
    {
      if( is.null( private$.clusterObj ) )
        stop( "cluster has been stopped" )
      return( private$.clusterObj )
    },

    .makeCluster = function()
    {
      n_nodes   <- self$n_nodes
      n_regions <- self$n_regions

      node_list <- vector(mode = "list", length = n_nodes )
      for( region in 1:n_regions )
      {
        node <- ( ( region -1 ) %% n_nodes ) + 1
        node_list[[ node ]] <- c( node_list[[ node ]], region )
      }
      private$.node_list  <- node_list
      private$.clusterObj <- makeCluster( n_nodes )

    },

    .stopCluster = function()
    {
      stopCluster( private$.cluster() )
      private$.clusterObj = NULL
    },

    .setBaseParams = function( base_params )
    {
      default_params = list(
        rng_seed = 0,
        days_of_interactions = 1,
        quarantine_days      = 1,
        rebuild_networks     = 0
      )

      for( name in names( base_params ) )
        default_params[[ name ]] <- base_params[[ name ]]

      private$.base_params <- default_params
    },

    .initializeSubModels = function()
    {
      base_params = self$base_params

      start_func = function( nodes  )
      {
        library( OpenABMCovid19)
        abms        <<- vector( mode = "list", length = length( nodes ))
        node_list   <<- nodes
        n_node_list <<- length( node_list )
        params      <<- vector( mode = "list", length = length( nodes ))

        for( nidx in 1:n_node_list )
        {
          ps <- base_params
          ps[[ "rng_seed"]] <- ps[[ "rng_seed"]] + node_list[ nidx ]

          params[[ nidx ]] <<- ps
          abms[[ nidx ]]   <<- Model.new(params = ps)
          params[[ nidx ]][[ "n_total" ]] <<- abms[[ nidx ]]$get_param( "n_total" )
        }
      }
      clusterApply( private$.cluster(), private$.node_list, start_func )
    }

  ),

  public = list(
    initialize = function(
      n_regions,
      n_nodes = parallel::detectCores(),
      base_params  = list()
    )
    {
      # clean destroyed models
      gc()

      n_nodes <- min( n_nodes, n_regions ) # no point making unused nodes
      private$.n_regions <- n_regions
      private$.n_nodes   <- n_nodes
      private$.makeCluster()

      private$.setBaseParams( base_params )
      private$.initializeSubModels()

    },

    finalize = function()
    {
      private$.stopCluster()
    },

    results = function()
    {
      results_func = function( data  )
      {
        results <- vector( mode = "list", length = n_node_list )
        for( nidx in 1:n_node_list )
          results[[ nidx ]] <- abms[[ nidx ]]$results()
        return( results )
      }

      t <- clusterApply( private$.cluster(), private$.node_list, results_func )
      max_time <- nrow( t[[ 1]][[1]] )
      t <- rbindlist(lapply( t, function( r ) rbindlist( lapply( r, as.data.table ), use.names = TRUE )), use.names = TRUE )

      if( t[,.N] != max_time * self$n_regions )
        stop( "the wrong number of results from sub=popualtions")

      t[ , n_region := rep( unlist( private$.node_list), each  = max_time ) ]
      return( t )
    },

    one_time_step_results = function()
    {
      results_func = function( data  )
      {
        results <- vector( mode = "list", length = n_node_list )
        for( nidx in 1:n_node_list ) {
          results[[ nidx ]] <- abms[[ nidx ]]$one_time_step_results()
          results[[ nidx ]][[ "n_region" ]] <- node_list[[ nidx ]]
        }
        return( results )
      }

      t <- clusterApply( private$.cluster(), private$.node_list, results_func )
      return( t )
    },

    migration_infect = function( n_infections )
    {
      infect_func = function( n_infections  )
      {
        for( nidx in 1:n_node_list )
        {

          if( n_infections[ nidx ] > 0 )
          {
            p_infect = floor( runif( n_infections[ nidx ] ) * params[[ nidx ]]$n_total )
             for( pdx in p_infect )
               abms[[ nidx ]]$seed_infect_by_idx( pdx )
          }
        }
        return()
      }

      if( length( n_infections ) == 1 )
        n_infections <- rep( n_infections, self$n_regions )

      infections_list <- lapply( private$.node_list, function( ndxs) n_infections[ ndxs ] )
      clusterApply( private$.cluster(), infections_list, infect_func )
      return()
    },

    update_running_params = function( param, values )
    {
      update_func = function( params  )
      {
        param  = params$param
        values = params$values

        for( ndx in 1:n_node_list )
        {
          if( !is.na( values[ ndx ] ) )
              abms[[ ndx ]]$update_running_params( param, values[ ndx ] )
        }
        return()
      }

      if( length( values ) == 1 )
        values <- rep( values, self$n_regions )

      values <- lapply( private$.node_list, function( ndxs) values[ ndxs ] )
      params <- lapply( values, function( v ) list( param = param, values = v ))

      clusterApply( private$.cluster(), params, update_func )
      return()
    },

    migration_infect_run = function( n_infections, n_steps )
    {
      infect_func = function( data )
      {
        results <- vector( mode = "numeric", length = n_node_list )

        n_infections = data$n_infect
        n_steps      = data$n_steps

        for( nidx in 1:n_node_list )
        {
          if( n_infections[ nidx ] > 0 )
          {
            p_infect = floor( runif( n_infections[ nidx ] ) * params[[ nidx ]]$n_total )
            for( pdx in p_infect )
              abms[[ nidx ]]$seed_infect_by_idx( pdx )
          }
          start <- abms[[ nidx ]]$one_time_step_results()[[ "total_infected" ]]
          abms[[ nidx ]]$run( n_steps, verbose = FALSE)
          results[ nidx ] <- abms[[ nidx ]]$one_time_step_results()[[ "total_infected" ]] - start

        }
        return( results )
      }

      # build input data
      if( length( n_infections ) == 1 )
        n_infections <- rep( n_infections, self$n_regions )

      infections_list <- lapply( private$.node_list, function( ndxs) n_infections[ ndxs ] )
      infections_list <- lapply( infections_list, function( v ) list( n_infect = v, n_steps = n_steps ) )

      res_list  <- clusterApply( private$.cluster(), infections_list, infect_func )
      results   <- vector( mode = "numeric", length = self$n_regions )
      node_list <- private$.node_list

      for( ndx in 1:length( node_list ) )
        results[ node_list[[ ndx ]] ] <- res_list[[ ndx ]]

      return(results )
    },

    time = function()
    {
      results_func = function( data  )
      {
        results <- vector( mode = "numeric", length = n_node_list )
        for( nidx in 1:n_node_list ) {
          results[ nidx ] <- abms[[ nidx ]]$c_model$time
        }
        return( results )
      }

      t <- clusterApply( private$.cluster(), private$.node_list, results_func )
      t <- unique( unlist( t ) )
      if( length( t ) != 1 )
        stop( "underlying ABMs are out of synch" )

      return( t )
    },

    run = function( n_steps = NULL )
    {
      run_func = function( n_steps  )
      {
        function( data  )
        {
          results <- vector( mode = "numeric", length = n_node_list )
          for( nidx in 1:n_node_list )
          {
            start <- abms[[ nidx ]]$one_time_step_results()[[ "total_infected" ]]
            abms[[ nidx ]]$run( n_steps, verbose = FALSE)
            results[ nidx ] <- abms[[ nidx ]]$one_time_step_results()[[ "total_infected" ]] - start
          }
          return( results )
        }
      }

      node_list <- private$.node_list
      res_list  <- clusterApply( private$.cluster(), node_list, run_func( n_steps ) )
      results   <- vector( mode = "numeric", length = self$n_regions )

      for( ndx in 1:length( node_list ) )
        results[ node_list[[ ndx ]] ] <- res_list[[ ndx ]]

      return(results )
    }
  ),

  active = list(
    n_nodes     = function( val = NULL ) private$.staticReturn( val, "n_nodes" ),
    n_regions   = function( val = NULL ) private$.staticReturn( val, "n_regions" ),
    base_params = function( val = NULL ) private$.staticReturn( val, "base_params" )
  )
)


