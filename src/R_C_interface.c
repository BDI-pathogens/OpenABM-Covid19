#include <R.h>
#include <Rinternals.h>
#include "params.h"

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




