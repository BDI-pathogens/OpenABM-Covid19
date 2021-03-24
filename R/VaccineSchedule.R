SWIG_update_daily_fraction = update_daily_fraction

#' R6Class VaccineSchedule
#'
#' @description
#' VaccineSchedule object has information an age-group vaccination plan.
#'
#' @examples
#' # Vaccinate 15% of age group 70-79 and 85% of age group 80+
#' vaccine.schedule <- OpenABMCovid19::VaccineSchedule$new(
#'   frac_70_79 = 0.15,
#'   frac_80    = 0.85,
#'   vaccine_type = OpenABMCovid19::VACCINE_TYPES[['FULL']]
#' )
VaccineSchedule <- R6Class( classname = 'VaccineSchedule', cloneable = FALSE,

  public = list(
    #' @field fraction_to_vaccinate Get the vaccination fractions per age-group.
    fraction_to_vaccinate = NULL,

    #' @field total_vaccinated The total number of vaccinations per age-group.
    total_vaccinated = NULL,

    #' @field vaccine_type The type of vaccine, see \code{\link{VACCINE_TYPES}}.
    vaccine_type    = NULL,

    #' @field efficacy Probability that the person is successfully vaccinated
    #'   (must be \code{0 <= efficacy <= 1}).
    efficacy        = NULL,

    #' @field time_to_protect Delay before it takes effect (in days).
    time_to_protect = NULL,

    #' @field vaccine_protection_period The duration of the vaccine before it
    #'   wanes.
    vaccine_protection_period = NULL,

    #' @param frac_0_9 Fraction of age group 0-9.
    #' @param frac_10_19 Fraction of age group 10-19.
    #' @param frac_20_29 Fraction of age group 20-29.
    #' @param frac_30_39 Fraction of age group 30-39.
    #' @param frac_40_49 Fraction of age group 40-49.
    #' @param frac_50_59 Fraction of age group 50-59.
    #' @param frac_60_69 Fraction of age group 60-69.
    #' @param frac_70_79 Fraction of age group 70-79.
    #' @param frac_80 Fraction of age group >80.
    #' @param vaccine_type The type of vaccine, see \code{\link{VACCINE_TYPES}}.
    #' @param efficacy Probability that the person is successfully vaccinated
    #'   (must be \code{0 <= efficacy <= 1}).
    #' @param time_to_protect Delay before it takes effect (in days).
    #' @param vaccine_protection_period The duration of the vaccine before it
    #'   wanes.
    initialize = function(
      frac_0_9   = 0,
      frac_10_19 = 0,
      frac_20_29 = 0,
      frac_30_39 = 0,
      frac_40_49 = 0,
      frac_50_59 = 0,
      frac_60_69 = 0,
      frac_70_79 = 0,
      frac_80    = 0,
      vaccine_type    = 0,
      efficacy        = 1.0,
      time_to_protect = 15,
      vaccine_protection_period = 365 )
    {
      fractions <- c(frac_0_9, frac_10_19, frac_20_29, frac_30_39, frac_40_49,
                     frac_50_59, frac_60_69, frac_70_79, frac_80)
      n <- length(AgeGroupEnum)
      if (n != length(fractions)) {
        stop("length(AgeGroupEnum) doesn't match VaccineSchedule's age groups")
      }

      self$fraction_to_vaccinate     <- fractions
      self$total_vaccinated          <- as.integer(rep(0,n))
      self$vaccine_type              <- vaccine_type
      self$efficacy                  <- efficacy
      self$time_to_protect           <- time_to_protect
      self$vaccine_protection_period <- vaccine_protection_period
    }
  )
)
