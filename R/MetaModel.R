
MetaModel <- R6Class( classname = 'MetaModel', cloneable = FALSE,

  private = list(
    .n_nodes     = NULL,
    .n_regions   = NULL,
    .node_list   = NULL,
    .clusterObj  = NULL,
    .base_params = NULL,
    .n_strains   = NULL,
    .network_names = NULL,
    .meta_data     = NULL,
    .map_data      = NULL,
    .xrange        = NULL,
    .yrange        = NULL,
    .migration_matrix = NULL,
    .migration_factor = NULL,
    .migration_frequency = NULL,
    .total_infected      = NULL,
    .time                = 0,

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
        n_total              = 1e4,
        days_of_interactions = 1,
        quarantine_days      = 1,
        rebuild_networks     = 0,
        max_n_strains        = 1
      )

      for( name in names( base_params ) )
        default_params[[ name ]] <- base_params[[ name ]]

      n_regions <- self$n_regions
      if( length( default_params[[ "rng_seed" ]] ) == 1 )
        default_params[[ "rng_seed" ]] <- default_params[[ "rng_seed" ]] * n_regions + seq( 1, n_regions )

      for( name in names( default_params ) )
      {
        n_p <- length( default_params[[ name ]] );

        if( !( n_p %in% c( 1, n_regions ) ) )
          stop( "parameters must be length 1 for global values are length of n_regions")

        if( n_p == 1 )
          default_params[[ name ]] <- rep(default_params[[ name ]], n_regions );
      }

      # convert to list of lists
      param_list = vector( mode = "list", length = n_regions )
      f_convert = function( name, vals )
      {
        for( idx in 1:n_regions )
          param_list[[ idx ]][[ name ]] <<- vals[ idx ]
      }
      mapply( f_convert, names( default_params ),default_params  )

      private$.base_params <- param_list
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
        strains     <<- lapply( strains, function(x) vector( mode = "list", length = base_params[[ 1 ]]$max_n_strains ) )
        networks    <<- vector( mode = "list", length = length( nodes ) )

        for( nidx in 1:n_node_list )
        {
          params[[ nidx ]] <<- base_params[[ nidx ]]
          abms[[ nidx ]]   <<- Model.new(params = params[[ nidx ]] )
          params[[ nidx ]][[ "n_total" ]] <<- abms[[ nidx ]]$get_param( "n_total" )
          strains[[ nidx ]][[ 1 ]] <<- Strain$new( abms[[ nidx]], 0 )

          networks[[ nidx ]] <<- list()
          network_ids <- abms[[ nidx]]$get_network_ids()
          for( network_id in network_ids )
          {
            net <- abms[[ nidx]]$get_network_by_id( network_id )
            networks[[ nidx ]][[ net$name() ]] <<- net
          }
        }
        return( list( network_names = names( networks[[ 1 ]] ) ) )
      }
      res = clusterApply( private$.cluster(), private$.node_list, start_func )

      private$.network_names <- unique( unlist( lapply( res, function( y ) y[[ "network_names"]]) ) )
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
    },

    .setMetaData = function( meta_data )
    {
      if( !is.null( meta_data ) )
      {
        reqCols = c( "n_region", "x", "y", "population", "name" )

        error_msg <- paste( "meta_data must be a data.table with a row for each region",
                            "and columns [", paste( reqCols, collapse = ", " ), "]" )

        if( !is.data.table( meta_data ) || length( setdiff( reqCols, names( meta_data)) ) )
          stop( error_msg )

        if( meta_data[ ,.N ] != self$n_regions )
          stop( error_msg )

        if( length( setdiff( seq( 1, self$n_regions ), meta_data[ , n_region ] ) ) )
          stop( "column n_region must have an entry for 1..n_region")

        meta_data = meta_data[ order( n_region ) ]
        meta_data[ , n_total := unlist( lapply( self$base_params, function( x ) x[[ "n_total" ]]) ) ]

        private$.meta_data <- meta_data

        private$.xrange = c( meta_data[ , min(x ) ], meta_data[ , max( x) ] )
        private$.yrange = c( meta_data[ , min(y ) ], meta_data[ , max( y) ] )
      }
    },

    .setMapData = function( map_data )
    {
      if( !is.null( map_data ) )
      {
        reqCols <- c( "n_region", "n_p", "xs", "ys" )

        error_msg <- paste( "map_data must be a data.table with columns [", paste( reqCols, collapse = ", " ), "]" )

        if( !is.data.table( map_data ) || length( setdiff( reqCols, names( map_data)) ) )
          stop( error_msg )

        # set the ranges for the graphs
        xrange <- c( map_data[ , min( xs) ],  map_data[ , max( xs) ] )
        yrange <- c( map_data[ , min( ys) ],  map_data[ , max( ys) ] )
        if( is.null( private$.xrange ) )
        {
          private$.xrange <= xrange
          private$.yrange <- yrange
        }
        else
        {
          private$.xrange <- c( min(private$.xrange[1] ,xrange[1]), max(private$.xrange[2] ,xrange[2]))
          private$.yrange <- c( min(private$.yrange[1] ,yrange[1]), max(private$.yrange[2] ,yrange[2]))
        }

        # build the map for plotly
        map_data <- map_data[ , lapply( .SD, function( x ) return( list(x) )), by = c( "n_region", "n_p" ), .SDcols = c("xs", "ys")]

        rect <- list(
          xref = "x",
          yref = "y",
          type = "path",
          line = list(width = 1, color = "black")
        )
        f_rect <- function( xs, ys)
        {
          l = rect;
          l[[ "path" ]] <- paste( c( sprintf( "%s %.2f %.2f ", c( "M", rep( "L", length( xs ) - 1) ), xs, ys), "Z"), collapse = " ")
          l
        }
        map_data[ , plotly_rect := mapply( f_rect, map_data[,xs], map_data[,ys], SIMPLIFY = FALSE) ]

        private$.map_data <- map_data
      }
    },

    .setMigrationMatrix = function( migration_matrix, migration_factor, migration_frequency )
    {
      if( !is.null( migration_matrix ) )
      {
        reqCols <- c( "n_region", "n_region_to", "transfer" )

        error_msg <- paste( "migration_matrix must be a data.table columns [", paste( reqCols, collapse = ", " ), "]" )

        if( !is.data.table( migration_matrix ) || length( setdiff( reqCols, names( migration_matrix)) ) )
          stop( error_msg )

        if( length( setdiff( seq( 1, self$n_regions ), migration_matrix[ , unique( n_region ) ] ) ) )
          stop( "column n_region of migration_matrix must have an entries for 1..n_region")

        if( length( setdiff( seq( 1, self$n_regions ), migration_matrix[ , unique( n_region_to ) ] ) ) )
          stop( "column n_region_to of migration_matrix must have an entries for 1..n_region")

        if( !is.numeric( migration_factor ) || migration_factor < 0 || migration_factor > 1 )
          stop( "migration_factor must be between 0 and 1" )

        migration_matrix[ , transfer_used := transfer * migration_factor ]

        private$.migration_matrix    <- migration_matrix
        private$.migration_factor    <- migration_factor
        private$.migration_frequency <- migration_frequency
      }
    },

    .get_range = function( range, pad )
    {
      diff <- range[ 2 ] - range[ 1 ]
      range[1] <- range[ 1 ] - diff * pad
      range[2] <- range[ 2 ] + diff * pad
      return( range )
    },

    xrange = function( pad = 0.05 ) private$.get_range( private$.xrange, pad ),
    yrange = function( pad = 0.05 ) private$.get_range( private$.yrange, pad )

  ),

  public = list(
    initialize = function(
      n_regions,
      n_nodes = parallel::detectCores(),
      base_params  = list(),
      meta_data = NULL,
      map_data  = NULL,
      migration_matrix = NULL,
      migration_factor = 0.1,
      migration_frequency = 1
    )
    {
      # clean destroyed models
      gc()

      n_nodes <- min( n_nodes, n_regions ) # no point making unused nodes
      private$.n_regions <- n_regions
      private$.n_nodes   <- n_nodes
      private$.makeCluster()

      private$.setBaseParams( base_params )
      private$.setMetaData( meta_data )
      private$.setMapData( map_data )
      private$.setMigrationMatrix( migration_matrix, migration_factor, migration_frequency )
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
            abms[[ nidx ]]$seed_infect_n_people(  n_infections[ nidx, strain_idx ], strain_idx = strain_idx - 1 )
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

    set_network_transmission_multiplier = function( multipliers )
    {
      update_func = function( mults  )
      {
        network_names = names( mults )
        for( mdx in 1:length( mults ) )
        {
          name = network_names[ mdx ]
          for( ndx in 1:n_node_list )
          {
            net = networks[[ ndx ]][[ name ]]
            if( is.R6( net ) )
              net$set_transmission_multiplier( mults[[ name ]][ ndx] )
          }
        }
        return()
      }

      # check to see if networks are known
      if( is.null( names( multipliers ) ) )
        stop( "Multipliers must be named list of networks")

      known_networks <- self$network_names
      if( length( setdiff( names( multipliers), known_networks ) ) > 0 )
        stop( "Not all networks names are recognised")

      # update the same for all
      n_regions   <- self$n_regions
      multipliers <- lapply( multipliers, function( x ) if( length( x ) == 1 ) rep( x, n_regions) else x )
      lapply( multipliers, function( x ) if( length( x ) != n_regions ) stop( "multipliers must be length 1 or n_regions") )

      # map on to node|_list
      n_nodes   <- self$n_nodes
      node_list <- private$.node_list
      mults     <- vector( mode = "list", length = n_nodes )
      for( ndx in 1:n_nodes )
        mults[[ ndx ]] <- lapply( multipliers, function( m ) m[ node_list[[ ndx ]] ] )

      clusterApply( private$.cluster(), mults, update_func )
      return()
    },

    combine_run = function( n_infections, n_steps )
    {
      infect_func = function( data )
      {
        n_infections   <- data$n_infect
        n_steps        <- data$n_steps
        n_strains      <- abms[[ 1 ]]$n_strains
        total_infected <- vector( mode = "list", length = n_node_list )
        start_time     <- abms[[ 1 ]]$time

        for( nidx in 1:n_node_list )
        {
          # migration infections
          if( !is.null( n_infections ) ) {
            for( strain_idx in 1:n_strains )
              abms[[ nidx ]]$seed_infect_n_people( n_infections[ nidx, strain_idx ], strain_idx = strain_idx - 1 )
          }

          # run steps( )
          abms[[ nidx ]]$run( n_steps, verbose = FALSE)
          end_time <- abms[[ nidx ]]$time
          total_infected[[ nidx ]] <- abms[[ nidx ]]$total_infected[ (start_time + 2 ):( end_time + 1 ), ]
        }

        return( list( total_infected = total_infected, time = end_time) )
      }

      # build input data
      node_list <- private$.node_list
      if( !is.null( n_infections ) ) {
        infections_list <- private$.prepare_n_infections( n_infections )
        data <- lapply( infections_list, function( v ) list( n_infect = v, n_steps = n_steps ) )
      } else {
        data <- lapply( node_list, function( ndxs ) list( n_infect = NULL, n_steps = n_steps ) )

      }

      # run results
      start_time    <- self$time
      results       <- clusterApply( private$.cluster(), data, infect_func )
      private$.time <- results[[ 1 ]]$time
      end_time      <- self$time

      # update total infected
      infected_cols  <- sprintf( "total_infected_strain_%d", 0:(self$base_params[[ 1 ]][[ "max_n_strains"]] - 1 ) )
      total_infected <- lapply( results, function( nd ) lapply( nd$total_infected, function( nnd ) as.data.table( nnd ) ) )
      total_infected <- rbindlist( unlist( total_infected, recursive = FALSE ), use.names = TRUE )
      setnames( total_infected, infected_cols )

      total_infected[ , time     := rep( (start_time + 1):end_time, n_regions )]
      total_infected[ , n_region := rep( unlist( private$.node_list), each = end_time - start_time ) ]

      private$.total_infected <- rbindlist( list( private$.total_infected, total_infected ), use.names = TRUE )

      total_infected <- self$total_infected
      new_infected   <- total_infected[ time == end_time][ order( n_region )][ , .SD, .SDcols = infected_cols ]
      if( start_time > 0 )
       new_infected <- new_infected - total_infected[ time == start_time ][ order( n_region )][, .SD, .SDcols = infected_cols ]

      return( new_infected )
    },


    add_new_strain = function( transmission_multiplier = 1, hospitalised_fraction = NA, hospitalised_fraction_multiplier = 1 )
    {
        if( self$n_strains == self$base_params[[ 1 ]][[ "max_n_strains" ]] )
          stop( "max_n_strains strains have been added already" )

        add_new_strain_func = function( data  )
        {
          for( nidx in 1:n_node_list ) {
            strain <- abms[[ nidx ]]$add_new_strain(
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

    set_cross_immunity_matrix = function( cross_immunity )
    {
      set_cross_immunity_matrix_func = function( cross_immunity  )
      {
        for( nidx in 1:n_node_list )
          abms[[ nidx ]]$set_cross_immunity_matrix( cross_immunity )
      }

      cross_immunity <- replicate( self$n_nodes, cross_immunity, simplify = FALSE )

      clusterApply( private$.cluster(), cross_immunity, set_cross_immunity_matrix_func )
      return()
    },

    run = function( n_steps = NULL )
    {
      n_infections <- NULL
    #  if( !is.null( self$migration_matrix ) )

     return( self$combine_run( n_infections, n_steps ) )
    },

    plot = function( time = NULL, height = 800 )
    {
      if( !is.null( time ) )
        stop( "single time not implemented" )

      if( is.null( self$meta_data ) )
        stop( "plot requires meta_data to be specified")

      results   <- self$results()
      meta_data <- self$meta_data

      results <- results[ ,.( time, n_region, total_infected ) ]
      results <- meta_data[ results, on = "n_region" ]
      results = results[ order( n_region, time ) ]
      results[ , new_infections := ifelse( n_region == shift( n_region, fill = 1 ),
                                           total_infected - shift( total_infected, fill = 0 ),
                                           total_infected ) ]
      results[ , percent_infections := new_infections / n_total * 100 ]


      xrange <- private$xrange()
      yrange <- private$yrange()

      width  = round( height * diff( xrange ) / diff( yrange ) )

      p = plot_ly(
        results,
        x = ~x,
        y = ~y,
        frame = ~time,
        color = ~percent_infections,
        text = ~name,
        type = "scatter",
        mode = "markers",
        height = height,
        width = width,
        marker = list(size = 10)
      ) %>%
        layout(
          shapes = self$map_data[ , plotly_rect ],
          xaxis  = list( range = xrange, title = "", visible = F),
          yaxis  = list( range = yrange, title = "", visible = F)
        )%>%
        animation_opts( 100, easing = "linear")

      return( p )
    }
  ),

  active = list(
    n_nodes     = function( val = NULL ) private$.staticReturn( val, "n_nodes" ),
    n_regions   = function( val = NULL ) private$.staticReturn( val, "n_regions" ),
    base_params = function( val = NULL ) private$.staticReturn( val, "base_params" ),
    n_strains   = function( val = NULL ) private$.staticReturn( val, "n_strains" ),
    time        = function( val = NULL ) private$.staticReturn( val, "time" ),
    total_infected = function( val = NULL ) private$.staticReturn( val, "total_infected" ),
    network_names  = function( val = NULL ) private$.staticReturn( val, "network_names" ),
    meta_data    = function( val = NULL ) private$.staticReturn( val, "meta_data" ),
    map_data     = function( val = NULL ) private$.staticReturn( val, "map_data" ),
    migration_matrix    = function( val = NULL ) private$.staticReturn( val, "migration_matrix" ),
    migration_factor    = function( val = NULL ) private$.staticReturn( val, "migration_factor" ),
    migration_frequency = function( val = NULL ) private$.staticReturn( val, "migration_frequency" )
  )
)


