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
#'   vaccine    = Vaccine
#' )
VaccineSchedule <- R6Class( classname = 'VaccineSchedule', cloneable = FALSE,

  public = list(
    #' @field fraction_to_vaccinate Get the vaccination fractions per age-group.
    fraction_to_vaccinate = NULL,

    #' @field total_vaccinated The total number of vaccinations per age-group.
    total_vaccinated = NULL,

    #' @field vaccinethe R vaccine object
    vaccine    = NULL,

    #' @param frac_0_9 Fraction of age group 0-9.
    #' @param frac_10_19 Fraction of age group 10-19.
    #' @param frac_20_29 Fraction of age group 20-29.
    #' @param frac_30_39 Fraction of age group 30-39.
    #' @param frac_40_49 Fraction of age group 40-49.
    #' @param frac_50_59 Fraction of age group 50-59.
    #' @param frac_60_69 Fraction of age group 60-69.
    #' @param frac_70_79 Fraction of age group 70-79.
    #' @param frac_80 Fraction of age group >80.
    #' @param vaccine A vaccine object
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
      vaccine    = -1,
      fractions  = NA
    )
    {
      if( is.na( fractions ) )
        fractions <- c(frac_0_9, frac_10_19, frac_20_29, frac_30_39, frac_40_49,
                       frac_50_59, frac_60_69, frac_70_79, frac_80)
      n <- length(AgeGroupEnum)
      if (n != length(fractions)) {
        stop("length(AgeGroupEnum) doesn't match VaccineSchedule's age groups")
      }

      if (!is.R6(vaccine) || !('Vaccine' %in% class(vaccine)))
        stop("argument vaccine must be an object of type Vaccine")

      self$fraction_to_vaccinate     <- fractions
      self$total_vaccinated          <- as.integer(rep(0,n))
      self$vaccine                   <- vaccine
    }
  )
)
