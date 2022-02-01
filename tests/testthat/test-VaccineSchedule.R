library(R6)
library( data.table )

test_that("VaccineSchedule initialization", {

  abm = Model.new( params = list( n_total = 1e4 ) )
  vac = abm$add_vaccine( full_efficacy = 1, time_to_protect = 15, vaccine_protection_period = 365)

  v <- VaccineSchedule$new(
      frac_0_9        = 0.0,
      frac_10_19      = 0.1,
      frac_20_29      = 0.2,
      frac_30_39      = 0.3,
      frac_40_49      = 0.4,
      frac_50_59      = 0.5,
      frac_60_69      = 0.6,
      frac_70_79      = 0.7,
      frac_80         = 0.8,
      vaccine         = vac )

  expect_equal(v$fraction_to_vaccinate, seq(0, 0.8, 0.1))
  expect_equal(v$total_vaccinated, rep(0,9))
  expect_equal(v$vaccine$full_efficacy(), 1)
  expect_equal(v$vaccine$time_to_protect(), 15)
  expect_equal(v$vaccine$vaccine_protection_period(), 365 )

  rm( abm )
})


# check the correct number of people have been vaccinated
check_n_vaccinated <- function( abm, vac_frac, days)
{
  indiv  <- as.data.table( abm$get_individuals() )
  N_70   <- indiv[ age_group == AgeGroupEnum[ "_70_79"], .N ]
  V_70   <- indiv[ age_group == AgeGroupEnum[ "_70_79"] & vaccine_status > 0 , .N ]
  N_80   <- indiv[ age_group == AgeGroupEnum[ "_80"], .N ]
  V_80   <- indiv[ age_group == AgeGroupEnum[ "_80"] & vaccine_status > 0 , .N ]
  V_else <- indiv[ age_group != AgeGroupEnum[ "_70_79"] & age_group != AgeGroupEnum[ "_80"] & vaccine_status > 0 , .N ]

  expect_equal(V_else, 0, label = "<70 vaccinated when not scheduled" )
  expect_lt( abs( V_70 - N_70 * vac_frac * days ), days + 0.01, label = "incorrect 70-79 vaccinated" )
  expect_lt( abs( V_80 - N_80 * vac_frac * days ), days + 0.01, label = "incorrect 80+ vaccinated" )
}

test_that("VaccineSchedule - single day application", {

  n_total  <- 1e4
  vac_frac <- 0.1
  abm      <- Model.new( params = list( n_total = 1e4 ) )
  vac      <- abm$add_vaccine( full_efficacy = 1, time_to_protect = 15, vaccine_protection_period = 365)

  v <- VaccineSchedule$new(
    frac_0_9        = 0.0,
    frac_10_19      = 0.0,
    frac_20_29      = 0.0,
    frac_30_39      = 0.0,
    frac_40_49      = 0.0,
    frac_50_59      = 0.0,
    frac_60_69      = 0.0,
    frac_70_79      = vac_frac,
    frac_80         = vac_frac,
    vaccine         = vac )

  for( day in 1:5 ) {
    abm$run( 1, verbose = FALSE )
    abm$vaccinate_schedule( v )
    check_n_vaccinated( abm, vac_frac, day  )
  }

  rm( abm )
})

test_that("VaccineSchedule - repeating schedule", {

  n_total  <- 1e4
  vac_frac <- 0.1
  abm      <- Model.new( params = list( n_total = 1e4 ) )
  vac      <- abm$add_vaccine( full_efficacy = 1, time_to_protect = 15, vaccine_protection_period = 365)

  v <- VaccineSchedule$new(
    frac_0_9        = 0.0,
    frac_10_19      = 0.0,
    frac_20_29      = 0.0,
    frac_30_39      = 0.0,
    frac_40_49      = 0.0,
    frac_50_59      = 0.0,
    frac_60_69      = 0.0,
    frac_70_79      = vac_frac,
    frac_80         = vac_frac,
    vaccine         = vac )

  # set up a recurring schedule
  abm$run( 1, verbose = FALSE )
  abm$vaccinate_schedule( v, recurring = TRUE )

  # run for 5 days, check it is added each day
  abm$run( 5, verbose = FALSE )
  check_n_vaccinated( abm, vac_frac, 5 )

  # remove it, run for longer and check it is not still applied
  abm$vaccinate_schedule( NULL )
  abm$run( 5, verbose = FALSE )
  check_n_vaccinated( abm, vac_frac, 5 )


  rm( abm )
})

test_that("VaccineSchedule - list schedule", {

  n_total  <- 1e4
  vac_frac <- 0.1
  abm      <- Model.new( params = list( n_total = 1e4 ) )
  vac      <- abm$add_vaccine( full_efficacy = 1, time_to_protect = 15, vaccine_protection_period = 365)

  v <- VaccineSchedule$new(
    frac_0_9        = 0.0,
    frac_10_19      = 0.0,
    frac_20_29      = 0.0,
    frac_30_39      = 0.0,
    frac_40_49      = 0.0,
    frac_50_59      = 0.0,
    frac_60_69      = 0.0,
    frac_70_79      = vac_frac,
    frac_80         = vac_frac,
    vaccine         = vac )

  # set up a multi-day different schedule
  sched_list = vector( mode = "list", length = 10 )
  sched_list[[ 2 ]] <- v
  sched_list[[ 3 ]] <- v
  sched_list[[ 6 ]] <- v
  sched_list[[ 8 ]] <- v
  sched_list[[ 9 ]] <- v
  expected <- cumsum(unlist( lapply( sched_list, is.R6) ))
  abm$vaccinate_schedule( sched_list, recurring = TRUE )

  # run for 8 days and check the amount each day
  for( day in 1:8 )
  {
    abm$run( 1, verbose = FALSE )
    check_n_vaccinated( abm, vac_frac, expected[ day ] )
  }

  # run for 5 days, check the whole schedule has been applied
  abm$run( 5, verbose = FALSE )
  check_n_vaccinated( abm, vac_frac, max( expected ) )

  rm( abm )
})

