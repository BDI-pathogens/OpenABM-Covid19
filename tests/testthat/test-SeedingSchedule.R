library(R6)
library( OpenABMCovid19)
library( testthat )

test_that("SeedingSchedule single strain", {

  # set up a model with no seed_infections and no transmitted infections
  base_params <- list( n_total = 10000, infectious_rate = 0, n_seed_infection = 0)
  abm         <- Model.new( params = base_params)

  # add a seeding schedule and run model
  schedule <- matrix( c( 0, 3, 2,5,1,0,0,2 ), ncol = 1 )
  abm$set_seeding_schedule( schedule )
  abm$run( nrow( schedule ), verbose = FALSE )

  # check that the number of infections is equal to the number in the scheulde
  total <- abm$total_infected[ -1 ] # note results have t=0

  expect_equal( total, cumsum( schedule[, 1] ), label = "incorrect number of seeded infections" )
})


test_that("SeedingSchedule multiple strain", {

  # set up a model with no seed_infections and no transmitted infections
  base_params <- list( n_total = 10000, infectious_rate = 0, n_seed_infection = 0, max_n_strains = 3 )
  abm         <- Model.new( params = base_params)

  # add the extra strains
  s1 <- abm$add_new_strain()
  s2 <- abm$add_new_strain()

  # add a seeding schedule and run model
  schedule <- matrix( c( 0, 3, 2,1, 1,2,3,0,2,1,2,1), ncol = 3 )
  abm$set_seeding_schedule( schedule )
  abm$run( nrow( schedule ), verbose = FALSE )

  # check that the number of infections is equal to the number in the scheulde
  total <- abm$total_infected[ -1, ] # note results have t=0

  for( idx in 1:3)
    expect_equal( total[ , idx], cumsum( schedule[, idx] ), label = "incorrect number of seeded infections" )
})
