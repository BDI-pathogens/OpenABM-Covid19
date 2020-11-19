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
#' \item \code{_0_9} Age group 0-9
#' \item \code{_10_19} Age group 10-19
#' \item \code{_20_29} Age group 20-29
#' \item \code{_30_39} Age group 30-39
#' \item \code{_40_49} Age group 40-49
#' \item \code{_50_59} Age group 50-59
#' \item \code{_60_69} Age group 60-69
#' \item \code{_70_79} Age group 70-79
#' \item \code{_80} Age group >80
#' }
#' @examples
#' # Add the age group constants to your workspace:
#' ag0_9   <- AgeGroupEnum[['_0_9']]
#' ag10_19 <- AgeGroupEnum[['_10_19']]
#' ag20_29 <- AgeGroupEnum[['_20_29']]
#' ag30_39 <- AgeGroupEnum[['_30_39']]
#' ag40_49 <- AgeGroupEnum[['_40_49']]
#' ag50_59 <- AgeGroupEnum[['_50_59']]
#' ag60_69 <- AgeGroupEnum[['_60_69']]
#' ag70_79 <- AgeGroupEnum[['_70_79']]
#' ag80    <- AgeGroupEnum[['_80']]
AgeGroupEnum <- list(
  '_0_9'   = 0,
  '_10_19' = 1,
  '_20_29' = 2,
  '_30_39' = 3,
  '_40_49' = 4,
  '_50_59' = 5,
  '_60_69' = 6,
  '_70_79' = 7,
  '_80'    = 8)

OccupationNetworkEnum <- list(
  '_primary_network'   = 0,
  '_secondary_network' = 1,
  '_working_network'   = 2,
  '_retired_network'   = 3,
  '_elderly_network'   = 4
)

ChildAdultElderlyEnum <- list(
  '_child'   = 0,
  '_adult'   = 1,
  '_elderly' = 2
)

ListIndiciesEnum <- list(
  '_1' = 0,
  '_2' = 1,
  '_3' = 2,
  '_4' = 3,
  '_5' = 4,
  '_6' = 5
)

TransmissionTypeEnum <- list(
  '_household'  = 0,
  '_occupation' = 1,
  '_random'     = 2
)

get_base_param_from_enum <- function(param) {
  allEnums <- c(
    AgeGroupEnum, OccupationNetworkEnum, ChildAdultElderlyEnum,
    ListIndiciesEnum, TransmissionTypeEnum)
  for (i in 1:length(allEnums)) {
    e <- allEnums[i]
    eName <- names(e)[1]
    eValue <- e[[eName]]
    if (endsWith(param, eName)) {
      base_name <- substr(param, 0, nchar(param) - nchar(eName))
      return(list("base_name" = base_name, "index" = eValue))
    }
  }
  return(NULL)
}



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
