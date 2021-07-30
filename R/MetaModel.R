
MetaModel <- R6Class( classname = 'MetaModel', cloneable = FALSE,

  private = list(
    .n_nodes     = NULL,
    .n_regions   = NULL,
    .node_list   = NULL,
    .clusterObj  = NULL,
    .base_params = NULL,
    .n_strains   = NULL,

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
        rebuild_networks     = 0,
        max_n_strains        = 1
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
        strains     <<- vector( mode = "list", length = length( nodes ))
        strains     <<- lapply( strains, function(x) vector( mode = "list", length = base_params$max_n_strains ) )

        for( nidx in 1:n_node_list )
        {
          ps <- base_params
          ps[[ "rng_seed"]] <- ps[[ "rng_seed"]] + node_list[ nidx ]

          params[[ nidx ]] <<- ps
          abms[[ nidx ]]   <<- Model.new(params = ps)
          params[[ nidx ]][[ "n_total" ]] <<- abms[[ nidx ]]$get_param( "n_total" )
          strains[[ nidx ]][[ 1 ]] <<- Strain$new( abms[[ nidx]], 0 )
        }
      }
      clusterApply( private$.cluster(), private$.node_list, start_func )
    },

    .prepare_n_infections = function( n_infections )
    {
      if( length( n_infections ) == 1 )
        n_infections <- rep( n_infections, self$n_regions )

      if( is.matrix( n_infections ) )
      {
        if( ncol( n_infections) > self$n_strains )
          stop( "n_infections has more columns than strains in the model" )
        infections_list <- lapply( private$.node_list, function( ndxs) n_infections[ ndxs, ] )
      } else
        infections_list <- lapply( private$.node_list, function( ndxs) matrix( n_infections[ ndxs ], ncol = 1 ) )

      return( infections_list )
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
      private$.n_strains = 1

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
        n_strains = ncol( n_infections )

        for( nidx in 1:n_node_list )
        {
          for( strain_idx in 1:n_strains )
            if( n_infections[ nidx, strain_idx ] > 0 )
            {
              p_infect = floor( runif( n_infections[ nidx, strain_idx ] ) * params[[ nidx ]]$n_total )
               for( pdx in p_infect )
                 abms[[ nidx ]]$seed_infect_by_idx( pdx, strain_idx = strain_idx - 1 )
            }
        }
        return()
      }

      infections_list <- private$.prepare_n_infections( n_infections )
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

    combine_run = function( n_infections, n_steps )
    {
      infect_func = function( data )
      {
        n_infections <- data$n_infect
        n_steps      <- data$n_steps
        n_strains    <- ncol( n_infections )
        results      <- matrix( 0, ncol = n_strains, nrow = n_node_list )

        for( nidx in 1:n_node_list )
        {
          # migration infections
          for( strain_idx in 1:n_strains )
            if( n_infections[ nidx, strain_idx ] > 0 )
            {
              p_infect = floor( runif( n_infections[ nidx, strain_idx ] ) * params[[ nidx ]]$n_total )
              for( pdx in p_infect )
                abms[[ nidx ]]$seed_infect_by_idx( pdx, strain_idx = strain_idx - 1 )
            }

          # run steps
          for( strain_idx in 1:n_strains )
            results[ nidx, strain_idx ] <- strains[[ nidx ]][[ strain_idx ]]$total_infected() * -1
          abms[[ nidx ]]$run( n_steps, verbose = FALSE)
          for( strain_idx in 1:n_strains )
            results[ nidx, strain_idx ] <- results[ nidx, strain_idx ] + strains[[ nidx ]][[ strain_idx ]]$total_infected()

        }
        return( results )
      }

      # build input data

      infections_list <- private$.prepare_n_infections( n_infections )
      infections_list <- lapply( infections_list, function( v ) list( n_infect = v, n_steps = n_steps ) )

      res_list  <- clusterApply( private$.cluster(), infections_list, infect_func )
      node_list <- private$.node_list
      n_strains <- self$n_strains;
      results   <- matrix( 0, nrow = self$n_regions, ncol = n_strains )

      for( ndx in 1:length( node_list ) )
        results[ node_list[[ ndx ]], ] <- res_list[[ ndx ]]

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

    add_new_strain = function( transmission_multiplier = 1, hospitalised_fraction = NA, hospitalised_fraction_multiplier = 1 )
    {
        if( self$n_strains == self$base_params[[ "max_n_strains" ]] )
          stop( "max_n_strains strains have been added already" )

        add_new_strain_func = function( data  )
        {
          for( nidx in 1:n_node_list ) {
            strain = abms[[ nidx ]]$add_new_strain(
              transmission_multiplier = data$transmission_multiplier ,
              hospitalised_fraction   = data$hospitalised_fraction,
              hospitalised_fraction_multiplier = data$hospitalised_fraction_multiplier
            )
            strain_idx <- strain$idx()
            strains[[ nidx ]][[ strain_idx + 1 ]] <<- strain

          }
          return( strain_idx )
        }

        data <- list(
          transmission_multiplier = transmission_multiplier ,
          hospitalised_fraction   = hospitalised_fraction,
          hospitalised_fraction_multiplier = hospitalised_fraction_multiplier
        )
        data <- replicate( self$n_nodes, data, simplify = FALSE )

        t <- clusterApply( private$.cluster(), data, add_new_strain_func )
        private$.n_strains = t[[ 1 ]] + 1

        return( t[[ 1 ]]);
    },

    run = function( n_steps = NULL )
    {
      run_func = function( n_steps  )
      {
        function( data  )
        {
          n_strains <- length( strains[[ 1 ]] )
          results   <- matrix( 0, ncol = n_strains, nrow = n_node_list )
          for( nidx in 1:n_node_list )
          {
            for( strain_idx in 1:n_strains )
              results[ nidx, strain_idx ] <- strains[[ nidx ]][[ strain_idx ]]$total_infected() * -1
            abms[[ nidx ]]$run( n_steps, verbose = FALSE)
            for( strain_idx in 1:n_strains )
              results[ nidx, strain_idx ] <- results[ nidx, strain_idx ] + strains[[ nidx ]][[ strain_idx ]]$total_infected()
          }
          return( results )
        }
      }

      node_list <- private$.node_list
      res_list  <- clusterApply( private$.cluster(), node_list, run_func( n_steps ) )
      n_strains <- self$n_strains;
      results   <- matrix( 0, nrow = self$n_regions, ncol = n_strains )

      for( ndx in 1:length( node_list ) )
        results[ node_list[[ ndx ]], ] <- res_list[[ ndx ]]

      return(results )
    }
  ),

  active = list(
    n_nodes     = function( val = NULL ) private$.staticReturn( val, "n_nodes" ),
    n_regions   = function( val = NULL ) private$.staticReturn( val, "n_regions" ),
    base_params = function( val = NULL ) private$.staticReturn( val, "base_params" ),
    n_strains   = function( val = NULL ) private$.staticReturn( val, "n_strains" )
  )
)


