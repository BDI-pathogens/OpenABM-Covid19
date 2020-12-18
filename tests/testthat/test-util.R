test_that("get_base_param_from_enum", {
  expect_equal(get_base_param_from_enum("fatality_fraction"), NULL)
  expect_equal(get_base_param_from_enum("fatality_fraction_20_29")$base_name, 'fatality_fraction')
  expect_equal(get_base_param_from_enum("fatality_fraction_20_29")$index, 2)
})
