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



#' Update-able parameters
#' @description
#' This is the list of parameter names can be updated whilst the model is
#' running (in \code{Model$update_running_params}).
#' @seealso \code{\link{Model}}
#' @seealso \code{\link{COVID19IBM}}
SAFE_UPDATE_PARAMS <- c(
  "test_on_symptoms",
  "test_on_traced",
  "quarantine_on_traced",
  "traceable_interaction_fraction",
  "tracing_network_depth",
  "allow_clinical_diagnosis",
  "quarantine_household_on_positive",
  "quarantine_household_on_symptoms",
  "quarantine_household_on_traced_positive",
  "quarantine_household_on_traced_symptoms",
  "quarantine_household_contacts_on_positive",
  "quarantine_household_contacts_on_symptoms",
  "quarantine_days",
  "test_order_wait",
  "test_order_wait_priority",
  "test_result_wait",
  "test_result_wait_priority",
  "self_quarantine_fraction",
  "lockdown_on",
  "lockdown_elderly_on",
  "app_turned_on",
  "app_users_fraction",
  "trace_on_symptoms",
  "trace_on_positive",
  "lockdown_house_interaction_multiplier",
  "lockdown_random_network_multiplier",
  "lockdown_occupation_multiplier_primary_network",
  "lockdown_occupation_multiplier_secondary_network",
  "lockdown_occupation_multiplier_working_network",
  "lockdown_occupation_multiplier_retired_network",
  "lockdown_occupation_multiplier_elderly_network",
  "manual_trace_on",
  "manual_trace_on_hospitalization",
  "manual_trace_on_positive",
  "manual_trace_delay",
  "manual_trace_exclude_app_users",
  "manual_trace_n_workers",
  "manual_trace_interviews_per_worker_day",
  "manual_trace_notifications_per_worker_day",
  "manual_traceable_fraction_household",
  "manual_traceable_fraction_occupation",
  "manual_traceable_fraction_random",
  "relative_transmission_household",
  "relative_transmission_occupation",
  "relative_transmission_random",
  "priority_test_contacts_0_9",
  "priority_test_contacts_10_19",
  "priority_test_contacts_20_29",
  "priority_test_contacts_30_39",
  "priority_test_contacts_40_49",
  "priority_test_contacts_50_59",
  "priority_test_contacts_60_69",
  "priority_test_contacts_70_79",
  "priority_test_contacts_80",
  "test_release_on_negative",
  "fatality_fraction_0_9",
  "fatality_fraction_10_19",
  "fatality_fraction_20_29",
  "fatality_fraction_30_39",
  "fatality_fraction_40_49",
  "fatality_fraction_50_59",
  "fatality_fraction_60_69",
  "fatality_fraction_70_79",
  "fatality_fraction_80"
)



# TODO(olegat) perhaps the SWIG defineEnumeration() should be in NAMESPACE?
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
AgeGroupEnum <- c(
  '_0_9'   = 0,
  '_10_19' = 1,
  '_20_29' = 2,
  '_30_39' = 3,
  '_40_49' = 4,
  '_50_59' = 5,
  '_60_69' = 6,
  '_70_79' = 7,
  '_80'    = 8)

OccupationNetworkEnum <- c(
  '_primary_network'   = 0,
  '_secondary_network' = 1,
  '_working_network'   = 2,
  '_retired_network'   = 3,
  '_elderly_network'   = 4
)

ChildAdultElderlyEnum <- c(
  '_child'   = 0,
  '_adult'   = 1,
  '_elderly' = 2
)

ListIndiciesEnum <- c(
  '_1' = 0,
  '_2' = 1,
  '_3' = 2,
  '_4' = 3,
  '_5' = 4,
  '_6' = 5
)

TransmissionTypeEnum <- c(
  '_household'  = 0,
  '_occupation' = 1,
  '_random'     = 2
)

# TODO(olegat) perhaps the SWIG defineEnumeration() should be in NAMESPACE?
#' Network construction types
#' @description
#' List of network construction types.
#' Wrapper for \code{enum NETWORK_CONSTRUCTIONS} (in \emph{constant.h}).
#' \itemize{
#'   \item{BESPOKE} Wrapper for C enum
#'     \code{NETWORK_CONSTRUCTION_BESPOKE}
#'   \item{HOUSEHOLD} Wrapper for C enum
#'     \code{NETWORK_CONSTRUCTION_HOUSEHOLD}
#'   \item{WATTS_STROGATZ} Wrapper for C enum
#'     \code{NETWORK_CONSTRUCTION_WATTS_STROGATZ}
#'   \item{RANDOM_DEFAULT} Wrapper for C enum
#'     \code{NETWORK_CONSTRUCTION_RANDOM_DEFAULT}
#'   \item{RANDOM_DEFAULT} Wrapper for C enum
#'     \code{NETWORK_CONSTRUCTION_RANDOM}
#' }
NETWORK_CONSTRUCTIONS <- c(
  'BESPOKE'        = 0,
  'HOUSEHOLD'      = 1,
  'WATTS_STROGATZ' = 2,
  'RANDOM_DEFAULT' = 3,
  'RANDOM'         = 4
)

#' Vaccine types
#' @description
#' List of vaccine types.
#' Wrapper for \code{enum VACCINE_TYPES} (in \emph{constant.h}).
#' \itemize{
#'   \item{FULL} Wrapper for C enum \code{VACCINE_TYPE_FULL}
#'   \item{SYMPTOM} Wrapper for C enum \code{VACCINE_TYPE_SYMPTOM}
#' }
VACCINE_TYPES <- c(
  'FULL'    = 0,
  'SYMPTOM' = 1
)

#' Vaccine status
#' @description
#' List of vaccines statuses.
#' Wrapper for \code{enum VACCINE_STATUS} (in \emph{constant.h}).
#' \itemize{
#'   \item{NO_VACCINE} Wrapper for C enum
#'     \code{NO_VACCINE}
#'   \item{VACCINE_NO_PROTECTION} Wrapper for C enum
#'     \code{VACCINE_NO_PROTECTION}
#'   \item{VACCINE_PROTECTED_FULLY} Wrapper for C enum
#'     \code{VACCINE_PROTECTED_FULLY}
#'   \item{VACCINE_PROTECTED_SYMPTOMS} Wrapper for C enum
#'     \code{VACCINE_PROTECTED_SYMPTOMS}
#'   \item{VACCINE_WANED_FULLY} Wrapper for C enum
#'     \code{VACCINE_WANED_FULLY}
#'   \item{VACCINE_WANED_SYMPTOMS} Wrapper for C enum
#'     \code{VACCINE_WANED_SYMPTOMS}
#' }
VACCINE_STATUS <- c(
  'NO_VACCINE'                 = 0,
  'VACCINE_NO_PROTECTION'      = 1,
  'VACCINE_PROTECTED_FULLY'    = 2,
  'VACCINE_PROTECTED_SYMPTOMS' = 3,
  'VACCINE_WANED_FULLY'        = 4,
  'VACCINE_WANED_SYMPTOMS'     = 5,
  'VACCINE_WANED'              = 6
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
