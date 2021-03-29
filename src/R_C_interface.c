#include <R.h>
#include <Rinternals.h>

#pragma push_macro("ERROR")
#undef ERROR // mute GCC warning: "ERROR" redefined
#include "model.h"
#include "input.h"
#include "params.h"
#pragma pop_macro("ERROR")

SEXP R_get_app_users ( SEXP R_c_model, SEXP n_total )
{
  // get the point to the model from the R pointer object
  model *c_model = (model *) R_ExternalPtrAddr(R_c_model);
  int n_tot = asInteger(n_total );

  // allocate memory to for the function call
  long *users = calloc( n_tot, sizeof(long) );
  get_app_users(c_model,users);

  // convert to R object
  SEXP R_res = PROTECT(allocVector(INTSXP, n_tot));
  for( int i = 0; i < n_tot; i++ )
    INTEGER(R_res)[i] = users[i];

  // free the memory
  free(users);
  UNPROTECT(1);

  return R_res;
}

SEXP R_get_individuals ( SEXP R_c_model, SEXP n_total )
{
  // get the point to the model from the R pointer object
  model *c_model = (model *) R_ExternalPtrAddr(R_c_model);
  int n_tot = asInteger(n_total );

  // allocate memory to for the function call
  const char *names[7] = { "ID", "current_status", "age_group",
     "occupation_network", "house_no", "infection_count", "vaccine_status" };
  long *ids       = calloc( n_tot, sizeof(long) );
  int *statuses   = calloc( n_tot, sizeof(int) );
  int *age_groups = calloc( n_tot, sizeof(int) );
  long *house_ids = calloc( n_tot, sizeof(long) );
  int *infection_counts    = calloc( n_tot, sizeof(int) );
  short *vaccine_statuses  = calloc( n_tot, sizeof(short) );
  int *occupation_networks = calloc( n_tot, sizeof(int) );

  get_individuals(c_model,ids,statuses,age_groups, occupation_networks,
                  house_ids, infection_counts, vaccine_statuses);

  // convert to R object
  SEXP R_list       = PROTECT(allocVector(VECSXP, 7));
  SEXP R_Names      = PROTECT(allocVector(STRSXP, 7));
  SEXP R_ids        = PROTECT(allocVector(INTSXP, n_tot));
  SEXP R_statuses   = PROTECT(allocVector(INTSXP, n_tot));
  SEXP R_age_groups = PROTECT(allocVector(INTSXP, n_tot));
  SEXP R_house_ids  = PROTECT(allocVector(INTSXP, n_tot));
  SEXP R_infection_counts    = PROTECT(allocVector(INTSXP, n_tot));
  SEXP R_vaccine_statuses    = PROTECT(allocVector(INTSXP, n_tot));
  SEXP R_occupation_networks = PROTECT(allocVector(INTSXP, n_tot));

  for( int i = 0; i < n_tot; i++ ) {
    INTEGER(R_ids)[i]        = ids[i];
    INTEGER(R_statuses)[i]   = statuses[i];
    INTEGER(R_age_groups)[i] = age_groups[i];
    INTEGER(R_house_ids)[i]  = house_ids[i];
    INTEGER(R_infection_counts)[i]    = infection_counts[i];
    INTEGER(R_vaccine_statuses)[i]    = vaccine_statuses[i];
    INTEGER(R_occupation_networks)[i] = occupation_networks[i];
  }

  SET_VECTOR_ELT(R_list, 0, R_ids);
  SET_VECTOR_ELT(R_list, 1, R_statuses);
  SET_VECTOR_ELT(R_list, 2, R_age_groups);
  SET_VECTOR_ELT(R_list, 3, R_occupation_networks);
  SET_VECTOR_ELT(R_list, 4, R_house_ids);
  SET_VECTOR_ELT(R_list, 5, R_infection_counts);
  SET_VECTOR_ELT(R_list, 6, R_vaccine_statuses);

  for (int i = 0; i < 7; i++) SET_STRING_ELT(R_Names, i, mkChar(names[i]));
  setAttrib(R_list, R_NamesSymbol, R_Names);

  // free the memory
  free(ids);
  free(statuses);
  free(age_groups);
  free(occupation_networks);
  free(house_ids);
  free(infection_counts);
  free(vaccine_statuses);
  UNPROTECT(9);

  return R_list;
}

SEXP R_get_network_ids ( SEXP R_c_model, SEXP R_max_ids )
{
  // get the point to the model from the R pointer object
  model *c_model = (model *) R_ExternalPtrAddr(R_c_model);
  int max_ids = asInteger( R_max_ids );

  // allocate memory to for the function call
  int *ids = calloc( max_ids, sizeof(int) );
  int n_ids = get_network_ids(c_model,ids,max_ids);

  if( n_ids == -1 )
    return( ScalarInteger(-1));

  // convert to R object
  SEXP R_res = PROTECT(allocVector(INTSXP, n_ids));
  for( int i = 0; i < n_ids; i++ )
    INTEGER(R_res)[i] = ids[i];

  // free the memory
  free(ids);
  UNPROTECT(1);

  return R_res;
}

SEXP R_get_n_transmissions ( SEXP R_c_model )
{
  // get the point to the model from the R pointer object
  model *c_model = (model *) R_ExternalPtrAddr(R_c_model);

  int n_trans = get_n_transmissions(c_model);

  return ScalarInteger( n_trans );
}

SEXP R_get_transmissions ( SEXP R_c_model )
{
  // get the point to the model from the R pointer object
  model *c_model = (model *) R_ExternalPtrAddr(R_c_model);

  int n_trans = get_n_transmissions(c_model);

  // allocate memory to for the function call
  const int n_names = 33;
  const char *names[33] = { "ID_recipient", "age_group_recipient",
    "house_no_recipient","occupation_network_recipient","worker_type_recipient",
    "hospital_state_recipient","infector_network","infector_network_id",
    "generation_time","ID_source","age_group_source",
    "house_no_source","occupation_network_source","worker_type_source",
    "hospital_state_source","time_infected_source","status_source",
    "time_infected","time_presymptomatic","time_presymptomatic_mild",
    "time_presymptomatic_severe","time_symptomatic","time_symptomatic_mild",
    "time_symptomatic_severe","time_asymptomatic","time_hospitalised",
    "time_critical","time_hospitalised_recovering","time_death",
    "time_recovered","time_susceptible","is_case",
    "strain_multiplier" };
  long *ID_recipient = calloc( n_trans, sizeof(long) );
  int *age_group_recipient = calloc( n_trans, sizeof(int) );
  long *house_no_recipient = calloc( n_trans, sizeof(long) );
  int *occupation_network_recipient = calloc( n_trans, sizeof(int) );
  int *worker_type_recipient = calloc( n_trans, sizeof(int) );
  int *hospital_state_recipient = calloc( n_trans, sizeof(int) );
  int *infector_network = calloc( n_trans, sizeof(int) );
  int *infector_network_id = calloc( n_trans, sizeof(int) );
  int *generation_time = calloc( n_trans, sizeof(int) );
  long *ID_source = calloc( n_trans, sizeof(long) );
  int *age_group_source = calloc( n_trans, sizeof(int) );
  long *house_no_source = calloc( n_trans, sizeof(long) );
  int *occupation_network_source = calloc( n_trans, sizeof(int) );
  int *worker_type_source = calloc( n_trans, sizeof(int) );
  int *hospital_state_source = calloc( n_trans, sizeof(int) );
  int *time_infected_source = calloc( n_trans, sizeof(int) );
  int *status_source = calloc( n_trans, sizeof(int) );
  int *time_infected = calloc( n_trans, sizeof(int) );
  int *time_presymptomatic = calloc( n_trans, sizeof(int) );
  int *time_presymptomatic_mild = calloc( n_trans, sizeof(int) );
  int *time_presymptomatic_severe = calloc( n_trans, sizeof(int) );
  int *time_symptomatic = calloc( n_trans, sizeof(int) );
  int *time_symptomatic_mild = calloc( n_trans, sizeof(int) );
  int *time_symptomatic_severe = calloc( n_trans, sizeof(int) );
  int *time_asymptomatic = calloc( n_trans, sizeof(int) );
  int *time_hospitalised = calloc( n_trans, sizeof(int) );
  int *time_critical = calloc( n_trans, sizeof(int) );
  int *time_hospitalised_recovering = calloc( n_trans, sizeof(int) );
  int *time_death = calloc( n_trans, sizeof(int) );
  int *time_recovered = calloc( n_trans, sizeof(int) );
  int *time_susceptible = calloc( n_trans, sizeof(int) );
  int *is_case = calloc( n_trans, sizeof(int) );
  float *strain_multiplier = calloc( n_trans, sizeof(float) );

  get_transmissions( c_model, ID_recipient, age_group_recipient,
      house_no_recipient, occupation_network_recipient, worker_type_recipient,
      hospital_state_recipient, infector_network, infector_network_id,
      generation_time, ID_source, age_group_source, house_no_source,
      occupation_network_source, worker_type_source, hospital_state_source,
      time_infected_source, status_source, time_infected, time_presymptomatic,
      time_presymptomatic_mild, time_presymptomatic_severe, time_symptomatic,
      time_symptomatic_mild, time_symptomatic_severe, time_asymptomatic,
      time_hospitalised, time_critical, time_hospitalised_recovering,
      time_death, time_recovered, time_susceptible, is_case,strain_multiplier );

  // convert to R object
  SEXP R_list = PROTECT(allocVector(VECSXP, n_names));
  SEXP R_Names      = PROTECT(allocVector(STRSXP, n_names));
  SEXP R_ID_recipient = PROTECT(allocVector(INTSXP, n_trans));
  SEXP R_age_group_recipient = PROTECT(allocVector(INTSXP, n_trans));
  SEXP R_house_no_recipient = PROTECT(allocVector(INTSXP, n_trans));
  SEXP R_occupation_network_recipient = PROTECT(allocVector(INTSXP, n_trans));
  SEXP R_worker_type_recipient = PROTECT(allocVector(INTSXP, n_trans));
  SEXP R_hospital_state_recipient = PROTECT(allocVector(INTSXP, n_trans));
  SEXP R_infector_network = PROTECT(allocVector(INTSXP, n_trans));
  SEXP R_infector_network_id = PROTECT(allocVector(INTSXP, n_trans));
  SEXP R_generation_time = PROTECT(allocVector(INTSXP, n_trans));
  SEXP R_ID_source = PROTECT(allocVector(INTSXP, n_trans));
  SEXP R_age_group_source = PROTECT(allocVector(INTSXP, n_trans));
  SEXP R_house_no_source = PROTECT(allocVector(INTSXP, n_trans));
  SEXP R_occupation_network_source = PROTECT(allocVector(INTSXP, n_trans));
  SEXP R_worker_type_source = PROTECT(allocVector(INTSXP, n_trans));
  SEXP R_hospital_state_source = PROTECT(allocVector(INTSXP, n_trans));
  SEXP R_time_infected_source = PROTECT(allocVector(INTSXP, n_trans));
  SEXP R_status_source = PROTECT(allocVector(INTSXP, n_trans));
  SEXP R_time_infected = PROTECT(allocVector(INTSXP, n_trans));
  SEXP R_time_presymptomatic = PROTECT(allocVector(INTSXP, n_trans));
  SEXP R_time_presymptomatic_mild = PROTECT(allocVector(INTSXP, n_trans));
  SEXP R_time_presymptomatic_severe = PROTECT(allocVector(INTSXP, n_trans));
  SEXP R_time_symptomatic = PROTECT(allocVector(INTSXP, n_trans));
  SEXP R_time_symptomatic_mild = PROTECT(allocVector(INTSXP, n_trans));
  SEXP R_time_symptomatic_severe = PROTECT(allocVector(INTSXP, n_trans));
  SEXP R_time_asymptomatic = PROTECT(allocVector(INTSXP, n_trans));
  SEXP R_time_hospitalised = PROTECT(allocVector(INTSXP, n_trans));
  SEXP R_time_critical = PROTECT(allocVector(INTSXP, n_trans));
  SEXP R_time_hospitalised_recovering = PROTECT(allocVector(INTSXP, n_trans));
  SEXP R_time_death = PROTECT(allocVector(INTSXP, n_trans));
  SEXP R_time_recovered = PROTECT(allocVector(INTSXP, n_trans));
  SEXP R_time_susceptible = PROTECT(allocVector(INTSXP, n_trans));
  SEXP R_is_case = PROTECT(allocVector(INTSXP, n_trans));
  SEXP R_strain_multiplier = PROTECT(allocVector(REALSXP, n_trans));

  for( int i = 0; i < n_trans; i++ ) {
    INTEGER(R_ID_recipient)[i] = ID_recipient[i];
    INTEGER(R_age_group_recipient)[i] = age_group_recipient[i];
    INTEGER(R_house_no_recipient)[i] = house_no_recipient[i];
    INTEGER(R_occupation_network_recipient)[i] = occupation_network_recipient[i];
    INTEGER(R_worker_type_recipient)[i] = worker_type_recipient[i];
    INTEGER(R_hospital_state_recipient)[i] = hospital_state_recipient[i];
    INTEGER(R_infector_network)[i] = infector_network[i];
    INTEGER(R_infector_network_id)[i] = infector_network_id[i];
    INTEGER(R_generation_time)[i] = generation_time[i];
    INTEGER(R_ID_source)[i] = ID_source[i];
    INTEGER(R_age_group_source)[i] = age_group_source[i];
    INTEGER(R_house_no_source)[i] = house_no_source[i];
    INTEGER(R_occupation_network_source)[i] = occupation_network_source[i];
    INTEGER(R_worker_type_source)[i] = worker_type_source[i];
    INTEGER(R_hospital_state_source)[i] = hospital_state_source[i];
    INTEGER(R_time_infected_source)[i] = time_infected_source[i];
    INTEGER(R_status_source)[i] = status_source[i];
    INTEGER(R_time_infected)[i] = time_infected[i];
    INTEGER(R_time_presymptomatic)[i] = time_presymptomatic[i];
    INTEGER(R_time_presymptomatic_mild)[i] = time_presymptomatic_mild[i];
    INTEGER(R_time_presymptomatic_severe)[i] = time_presymptomatic_severe[i];
    INTEGER(R_time_symptomatic)[i] = time_symptomatic[i];
    INTEGER(R_time_symptomatic_mild)[i] = time_symptomatic_mild[i];
    INTEGER(R_time_symptomatic_severe)[i] = time_symptomatic_severe[i];
    INTEGER(R_time_asymptomatic)[i] = time_asymptomatic[i];
    INTEGER(R_time_hospitalised)[i] = time_hospitalised[i];
    INTEGER(R_time_critical)[i] = time_critical[i];
    INTEGER(R_time_hospitalised_recovering)[i] = time_hospitalised_recovering[i];
    INTEGER(R_time_death)[i] = time_death[i];
    INTEGER(R_time_recovered)[i] = time_recovered[i];
    INTEGER(R_time_susceptible)[i] = time_susceptible[i];
    INTEGER(R_is_case)[i] = is_case[i];
    REAL(R_strain_multiplier)[i] = strain_multiplier[i];
  }

  SET_VECTOR_ELT(R_list, 0, R_ID_recipient);
  SET_VECTOR_ELT(R_list, 1, R_age_group_recipient);
  SET_VECTOR_ELT(R_list, 2, R_house_no_recipient);
  SET_VECTOR_ELT(R_list, 3, R_occupation_network_recipient);
  SET_VECTOR_ELT(R_list, 4, R_worker_type_recipient);
  SET_VECTOR_ELT(R_list, 5, R_hospital_state_recipient);
  SET_VECTOR_ELT(R_list, 6, R_infector_network);
  SET_VECTOR_ELT(R_list, 7, R_infector_network_id);
  SET_VECTOR_ELT(R_list, 8, R_generation_time);
  SET_VECTOR_ELT(R_list, 9, R_ID_source);
  SET_VECTOR_ELT(R_list, 10, R_age_group_source);
  SET_VECTOR_ELT(R_list, 11, R_house_no_source);
  SET_VECTOR_ELT(R_list, 12, R_occupation_network_source);
  SET_VECTOR_ELT(R_list, 13, R_worker_type_source);
  SET_VECTOR_ELT(R_list, 14, R_hospital_state_source);
  SET_VECTOR_ELT(R_list, 15, R_time_infected_source);
  SET_VECTOR_ELT(R_list, 16, R_status_source);
  SET_VECTOR_ELT(R_list, 17, R_time_infected);
  SET_VECTOR_ELT(R_list, 18, R_time_presymptomatic);
  SET_VECTOR_ELT(R_list, 19, R_time_presymptomatic_mild);
  SET_VECTOR_ELT(R_list, 20, R_time_presymptomatic_severe);
  SET_VECTOR_ELT(R_list, 21, R_time_symptomatic);
  SET_VECTOR_ELT(R_list, 22, R_time_symptomatic_mild);
  SET_VECTOR_ELT(R_list, 23, R_time_symptomatic_severe);
  SET_VECTOR_ELT(R_list, 24, R_time_asymptomatic);
  SET_VECTOR_ELT(R_list, 25, R_time_hospitalised);
  SET_VECTOR_ELT(R_list, 26, R_time_critical);
  SET_VECTOR_ELT(R_list, 27, R_time_hospitalised_recovering);
  SET_VECTOR_ELT(R_list, 28, R_time_death);
  SET_VECTOR_ELT(R_list, 29, R_time_recovered);
  SET_VECTOR_ELT(R_list, 30, R_time_susceptible);
  SET_VECTOR_ELT(R_list, 31, R_is_case);
  SET_VECTOR_ELT(R_list, 32, R_strain_multiplier);

  for (int i = 0; i < n_names; i++)
    SET_STRING_ELT(R_Names, i, mkChar(names[i]));
  setAttrib(R_list, R_NamesSymbol, R_Names);

  // free the memory
  free( ID_recipient );
  free( age_group_recipient );
  free( house_no_recipient );
  free( occupation_network_recipient );
  free( worker_type_recipient );
  free( hospital_state_recipient );
  free( infector_network );
  free( infector_network_id );
  free( generation_time );
  free( ID_source );
  free( age_group_source );
  free( house_no_source );
  free( occupation_network_source );
  free( worker_type_source );
  free( hospital_state_source );
  free( time_infected_source );
  free( status_source );
  free( time_infected );
  free( time_presymptomatic );
  free( time_presymptomatic_mild );
  free( time_presymptomatic_severe );
  free( time_symptomatic );
  free( time_symptomatic_mild );
  free( time_symptomatic_severe );
  free( time_asymptomatic );
  free( time_hospitalised );
  free( time_critical );
  free( time_hospitalised_recovering );
  free( time_death );
  free( time_recovered );
  free( time_susceptible );
  free( is_case );
  free( strain_multiplier );
  UNPROTECT(n_names+2);

  return R_list;
}




