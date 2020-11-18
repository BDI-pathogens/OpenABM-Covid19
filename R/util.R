# check if a double is a whole number, and convert it into an integer.
# returns NA_integer_ if number is NA.
make_integer <- function(number) {
  if (is.na(number)) {
    return(NA_integer_)
  }
  if (!is.numeric(number)) {
    stop(paste(number, "is not numeric"))
  }
  whole_part <- as.integer(number)
  fraction_part <- number - whole_part
  if (fraction_part != 0) {
    stop(paste(number,"is not a whole number"))
  }
  return(whole_part)
}



#' Age Group Indices
#' @description
#' List of age group indices.
#' Wrapper for \code{enum AGE_GROUPS} (in \emph{constant.h}).
#' \itemize{
#' \item \code{ag0_9} Age group 0-9
#' \item \code{ag10_19} Age group 10-19
#' \item \code{ag20_29} Age group 20-29
#' \item \code{ag30_39} Age group 30-39
#' \item \code{ag40_49} Age group 40-49
#' \item \code{ag50_59} Age group 50-59
#' \item \code{ag60_69} Age group 60-69
#' \item \code{ag70_79} Age group 70-79
#' \item \code{ag80} Age group >80
#' }
#' @examples
#' # Add the age group constants to your workspace:
#' ag0_9   <- AgeGroups$ag0_9
#' ag10_19 <- AgeGroups$ag10_19
#' ag20_29 <- AgeGroups$ag20_29
#' ag30_39 <- AgeGroups$ag30_39
#' ag40_49 <- AgeGroups$ag40_49
#' ag50_59 <- AgeGroups$ag50_59
#' ag60_69 <- AgeGroups$ag60_69
#' ag70_79 <- AgeGroups$ag70_79
#' ag80    <- AgeGroups$ag80
AgeGroups <- list(
  'ag0_9'   = 0,
  'ag10_19' = 1,
  'ag20_29' = 2,
  'ag30_39' = 3,
  'ag40_49' = 4,
  'ag50_59' = 5,
  'ag60_69' = 6,
  'ag70_79' = 7,
  'ag80'    = 8)



# Suppress NOTEs from `R CMD check`, example;
# AGE_OCCUPATION_MAP: no visible global function definition for
#  'AGE_OCCUPATION_MAP_set'
`AGE_OCCUPATION_MAP_set` = function(s_AGE_OCCUPATION_MAP_set) {
  stop('AGE_OCCUPATION_MAP is read-only')
}
`AGE_TYPE_MAP_set` = function(s_AGE_TYPE_MAP_set) {
  stop('AGE_TYPE_MAP is read-only')
}
`EVENT_TYPE_TO_WARD_MAP_set` = function(s_EVENT_TYPE_TO_WARD_MAP_set) {
  stop('EVENT_TYPE_TO_WARD_MAP is read-only')
}
`NETWORK_TYPE_MAP_set` = function(s_NETWORK_TYPE_MAP_set) {
  stop('NETWORK_TYPE_MAP is read-only')
}
`OCCUPATION_DEFAULT_MAP_set` = function(s_OCCUPATION_DEFAULT_MAP_set) {
  stop('OCCUPATION_DEFAULT_MAP is read-only')
}
`NEWLY_INFECTED_STATES_set` = function(s_NEWLY_INFECTED_STATES_set) {
  stop('NEWLY_INFECTED_STATES is read-only')
}
