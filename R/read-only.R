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
