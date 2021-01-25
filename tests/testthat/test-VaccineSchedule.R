library(R6)

test_that("VaccineSchedule initialization", {
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
      vaccine_type    = 0,
      efficacy        = 1.0,
      time_to_protect = 15,
      vaccine_protection_period = 365 )

  expect_equal(v$fraction_to_vaccinate, seq(0, 0.8, 0.1))
  expect_equal(v$total_vaccinated, rep(0,9))
  expect_equal(v$vaccine_type, 0)
  expect_equal(v$efficacy, 1)
  expect_equal(v$time_to_protect, 15)
  expect_equal(v$vaccine_protection_period, 365)
})
