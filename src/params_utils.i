%module params_utils

%inline %{
/*****************************************************************************************
*  Name:        get_param_quarantine_days
*  Description: Gets the value of an int parameter
******************************************************************************************/
int get_param_quarantine_days(parameters *params)
{
    return params->quarantine_days;
}

/*****************************************************************************************
*  Name: 		get_param_rng_seed
*  Description: Gets the value of a parameter
******************************************************************************************/
long get_param_rng_seed(parameters *params)
{
    return params->rng_seed;
}

/*****************************************************************************************
*  Name: 		get_param_param_id
*  Description: Gets the value of a parameter
******************************************************************************************/
long get_param_param_id(parameters *params)
{
    return params->param_id;
}

/*****************************************************************************************
*  Name: 		get_param_n_total
*  Description: Gets the value of a parameter
******************************************************************************************/
long get_param_n_total(parameters *params)
{
    return params->n_total;
}

/*****************************************************************************************
*  Name: 		get_param_days_of_interactions
*  Description: Gets the value of a parameter
******************************************************************************************/
int get_param_days_of_interactions(parameters *params)
{
    return params->days_of_interactions;
}

/*****************************************************************************************
*  Name: 		get_param_mean_random_interactions
*  Description: Gets the value of a parameter
******************************************************************************************/
double get_mean_random_interactions(parameters *params, int idx)
{
    if (idx >= N_AGE_TYPES) return -1;

    return params->mean_random_interactions[idx];
}

/*****************************************************************************************
*  Name: 		get_param_sd_random_interactions
*  Description: Gets the value of a parameter
******************************************************************************************/
double get_sd_random_interactions(parameters *params, int idx)
{
    if (idx >= N_AGE_TYPES) return -1;

    return params->sd_random_interactions[idx];
}

/*****************************************************************************************
*  Name: 		get_param_random_interaction_distribution
*  Description: Gets the value of a parameter
******************************************************************************************/
int get_param_random_interaction_distribution(parameters *params)
{
    return params->random_interaction_distribution;
}

/*****************************************************************************************
*  Name: 		get_param_mean_work_interactions
*  Description: Gets the value of a parameter
******************************************************************************************/
double get_param_mean_work_interactions(parameters *params, int idx)
{
    if (idx >= N_WORK_NETWORKS) return -1;

    return params->mean_work_interactions[idx];
}

/*****************************************************************************************
*  Name: 		get_param_daily_fraction_work
*  Description: Gets the value of a parameter
******************************************************************************************/
double get_param_daily_fraction_work(parameters *params)
{
    return params->daily_fraction_work;
}

/*****************************************************************************************
*  Name: 		get_param_daily_fraction_work_used
*  Description: Gets the value of a parameter
******************************************************************************************/
double get_param_daily_fraction_work_used(parameters *params, int idx)
{
    if (idx >= N_WORK_NETWORKS) return -1;

    return params->daily_fraction_work_used[idx];
}

/*****************************************************************************************
*  Name: 		get_param_child_network_adults
*  Description: Gets the value of a parameter
******************************************************************************************/
double get_param_child_network_adults(parameters *params)
{
    return params->child_network_adults;
}

/*****************************************************************************************
*  Name: 		get_param_elderly_network_adults
*  Description: Gets the value of a parameter
******************************************************************************************/
double get_param_elderly_network_adults(parameters *params)
{
    return params->elderly_network_adults;
}

/*****************************************************************************************
*  Name: 		get_param_mean_infectious_period
*  Description: Gets the value of a parameter
******************************************************************************************/
double get_param_mean_infectious_period(parameters *params)
{
    return params->mean_infectious_period;
}

/*****************************************************************************************
*  Name: 		get_param_sd_infectious_period
*  Description: Gets the value of a parameter
******************************************************************************************/
double get_param_sd_infectious_period(parameters *params)
{
    return params->sd_infectious_period;
}

/*****************************************************************************************
*  Name: 		get_param_infectious_rate
*  Description: Gets the value of a parameter
******************************************************************************************/
double get_param_infectious_rate(parameters *params)
{
    return params->infectious_rate;
}

/*****************************************************************************************
*  Name: 		get_param_relative_susceptibility
*  Description: Gets the value of a parameter
******************************************************************************************/
double get_param_relative_susceptibility(parameters *params, int idx)
{
    if (idx >= N_AGE_GROUPS) return -1;

    return params->relative_susceptibility[idx];
}

/*****************************************************************************************
*  Name: 		get_param_adjusted_susceptibility
*  Description: Gets the value of a parameter
******************************************************************************************/
double get_param_adjusted_susceptibility(parameters *params, int idx)
{
    if (idx >= N_AGE_GROUPS) return -1;

    return params->adjusted_susceptibility[idx];
}

/*****************************************************************************************
*  Name: 		get_param_relative_transmission_by_type
*  Description: Gets the value of a parameter
******************************************************************************************/
double get_param_relative_transmission_by_type(parameters *params, int idx)
{
    if (idx >= N_AGE_GROUPS) return -1;

    return params->relative_transmission_by_type[idx];
}

/*****************************************************************************************
*  Name: 		get_param_relative_transmission_by_type_used
*  Description: Gets the value of a parameter
******************************************************************************************/
double get_param_relative_transmission_by_type_used(parameters *params, int idx)
{
    if (idx >= N_AGE_GROUPS) return -1;

    return params->relative_transmission_by_type_used[idx];
}

/*****************************************************************************************
*  Name: 		get_param_mean_time_to_symptoms
*  Description: Gets the value of a parameter
******************************************************************************************/
double get_param_mean_time_to_symptoms(parameters *params)
{
    return params->mean_time_to_symptoms;
}

/*****************************************************************************************
*  Name: 		get_param_sd_time_to_symptoms
*  Description: Gets the value of a parameter
******************************************************************************************/
double get_param_sd_time_to_symptoms(parameters *params)
{
    return params->sd_time_to_symptoms;
}

/*****************************************************************************************
*  Name: 		get_param_hospitalised_fraction
*  Description: Gets the value of a parameter
******************************************************************************************/
double get_param_hospitalised_fraction(parameters *params, int idx)
{
    if (idx >= N_AGE_GROUPS) return -1;

    return params->hospitalised_fraction[idx];
}

/*****************************************************************************************
*  Name: 		get_param_critical_fraction
*  Description: Gets the value of a parameter
******************************************************************************************/
double get_param_critical_fraction(parameters *params, int idx)
{
    if (idx >= N_AGE_GROUPS) return -1;

    return params->critical_fraction[idx];
}

/*****************************************************************************************
*  Name: 		get_param_fatality_fraction
*  Description: Gets the value of a parameter
******************************************************************************************/
double get_param_fatality_fraction(parameters *params, int idx)
{
    if (idx >= N_AGE_GROUPS) return -1;

    return params->fatality_fraction[idx];
}

/*****************************************************************************************
*  Name: 		get_param_mean_time_to_hospital
*  Description: Gets the value of a parameter
******************************************************************************************/
double get_param_mean_time_to_hospital(parameters *params)
{
    return params->mean_time_to_hospital;
}

/*****************************************************************************************
*  Name: 		get_param_mean_time_to_critical
*  Description: Gets the value of a parameter
******************************************************************************************/
double get_param_mean_time_to_critical(parameters *params)
{
    return params->mean_time_to_critical;
}

/*****************************************************************************************
*  Name: 		get_param_mean_time_to_recover
*  Description: Gets the value of a parameter
******************************************************************************************/
double get_param_mean_time_to_recover(parameters *params)
{
    return params->mean_time_to_recover;
}

/*****************************************************************************************
*  Name: 		get_param_sd_time_to_recover
*  Description: Gets the value of a parameter
******************************************************************************************/
double get_param_mean_sd_to_recover(parameters *params)
{
    return params->sd_time_to_recover;
}

/*****************************************************************************************
*  Name: 		get_param_mean_time_to_death
*  Description: Gets the value of a parameter
******************************************************************************************/
double get_param_mean_time_to_death(parameters *params)
{
    return params->mean_time_to_death;
}

/*****************************************************************************************
*  Name: 		get_param_sd_time_to_death
*  Description: Gets the value of a parameter
******************************************************************************************/
double get_param_mean_sd_to_death(parameters *params)
{
    return params->sd_time_to_death;
}

/*****************************************************************************************
*  Name: 		get_param_household_size
*  Description: Gets the value of a parameter
******************************************************************************************/
double get_param_household_size(parameters *params, int idx)
{
    if (idx >= N_HOUSEHOLD_MAX) return -1;

    return params->household_size[idx];
}

/*****************************************************************************************
*  Name: 		get_param_population_group
*  Description: Gets the value of a parameter
******************************************************************************************/
double get_param_population_group(parameters *params, int idx)
{
    if (idx >= N_AGE_GROUPS) return -1;

    return params->population_group[idx];
}

/*****************************************************************************************
*  Name: 		get_param_population_type
*  Description: Gets the value of a parameter
******************************************************************************************/
double get_param_population_type(parameters *params, int idx)
{
    if (idx >= N_AGE_GROUPS) return -1;

    return params->population_type[idx];
}

/*****************************************************************************************
*  Name: 		get_param_fraction_asymptomatic
*  Description: Gets the value of a parameter
******************************************************************************************/
double get_param_fraction_asymptomatic(parameters *params, int idx)
{
    if (idx >= N_AGE_GROUPS) return -1;

    return params->fraction_asymptomatic[idx];
}

/*****************************************************************************************
*  Name: 		get_param_asymptomatic_infectious_factor
*  Description: Gets the value of a parameter
******************************************************************************************/
double get_param_asymptomatic_infectious_factor(parameters *params)
{
    return params->asymptomatic_infectious_factor;
}

/*****************************************************************************************
*  Name: 		get_param_mean_asymptomatic_to_recover
*  Description: Gets the value of a parameter
******************************************************************************************/
double get_param_mean_asymptomatic_to_recover(parameters *params)
{
    return params->mean_asymptomatic_to_recovery;
}

/*****************************************************************************************
*  Name: 		get_param_sd_asymptomatic_to_recover
*  Description: Gets the value of a parameter
******************************************************************************************/
double get_param_sd_asymptomatic_to_recover(parameters *params)
{
    return params->sd_asymptomatic_to_recovery;
}

/*****************************************************************************************
*  Name:		get_param_quarantine_daily_interactions
*  Description: Gets the value of an int parameter
******************************************************************************************/
int get_param_quarantine_daily_interactions(parameters *params)
{
    return params->quarantined_daily_interactions;
}

/*****************************************************************************************
*  Name:		get_param_hospitalised_daily_interactions
*  Description: Gets the value of an int parameter
******************************************************************************************/
int get_param_hospitalised_daily_interactions(parameters *params)
{
    return params->hospitalised_daily_interactions;
}

/*****************************************************************************************
*  Name:		get_param_self_quarantine_fraction
*  Description: Gets the value of an int parameter
******************************************************************************************/
double get_param_self_quarantine_fraction(parameters *params)
{
    return params->self_quarantine_fraction;
}

/*****************************************************************************************
*  Name:		get_param_trace_on_symptoms
*  Description: Gets the value of an int parameter
******************************************************************************************/
int get_param_trace_on_symptoms(parameters *params)
{
    return params->trace_on_symptoms;
}

/*****************************************************************************************
*  Name:		get_param_trace_on_positive
*  Description: Gets the value of an int parameter
******************************************************************************************/
int get_param_trace_on_positive(parameters *params)
{
    return params->trace_on_positive;
}

/*****************************************************************************************
*  Name:		get_param_quarantine_length_self
*  Description: Gets the value of an int parameter
******************************************************************************************/
int get_param_quarantine_length_self(parameters *params)
{
    return params->quarantine_length_self;
}

/*****************************************************************************************
*  Name:		get_param_quarantine_length_traced
*  Description: Gets the value of an int parameter
******************************************************************************************/
int get_param_quarantine_length_traced(parameters *params)
{
    return params->quarantine_length_traced;
}

/*****************************************************************************************
*  Name:		get_param_quarantine_length_positive
*  Description: Gets the value of an int parameter
******************************************************************************************/
int get_param_quarantine_length_positive(parameters *params)
{
    return params->quarantine_length_positive;
}

/*****************************************************************************************
*  Name:		get_param_quarantine_dropout_self
*  Description: Gets the value of an int parameter
******************************************************************************************/
double get_param_quarantine_dropout_self(parameters *params)
{
    return params->quarantine_dropout_self;
}

/*****************************************************************************************
*  Name:		get_param_quarantine_dropout_traced
*  Description: Gets the value of an int parameter
******************************************************************************************/
double get_param_quarantine_dropout_traced(parameters *params)
{
    return params->quarantine_dropout_traced;
}

/*****************************************************************************************
*  Name:		get_param_quarantine_dropout_positive
*  Description: Gets the value of an int parameter
******************************************************************************************/
double get_param_quarantine_dropout_positive(parameters *params)
{
    return params->quarantine_dropout_positive;
}

/*****************************************************************************************
*  Name:		get_param_quarantine_on_traced
*  Description: Gets the value of an int parameter
******************************************************************************************/
int get_param_quarantine_on_traced(parameters *params)
{
    return params->quarantine_on_traced;
}

/*****************************************************************************************
*  Name:		get_param_traceable_interaction_fraction
*  Description: Gets the value of an int parameter
******************************************************************************************/
double get_param_traceable_interaction_fraction(parameters *params)
{
    return params->traceable_interaction_fraction;
}

/*****************************************************************************************
*  Name:		get_param_tracing_network_depth
*  Description: Gets the value of an int parameter
******************************************************************************************/
int get_param_tracing_network_depth(parameters *params)
{
    return params->tracing_network_depth;
}

/*****************************************************************************************
*  Name:		get_param_allow_clinical_diagnosis
*  Description: Gets the value of an int parameter
******************************************************************************************/
int get_param_allow_clinical_diagnosis(parameters *params)
{
    return params->allow_clinical_diagnosis;
}

/*****************************************************************************************
*  Name:		get_param_quarantine_household_on_symptoms
*  Description: Gets the value of an int parameter
******************************************************************************************/
int get_param_quarantine_household_on_symptoms(parameters *params)
{
    return params->quarantine_household_on_symptoms;
}

/*****************************************************************************************
*  Name:		get_param_quarantine_household_on_positive
*  Description: Gets the value of an int parameter
******************************************************************************************/
int get_param_quarantine_household_on_positive(parameters *params)
{
    return params->quarantine_household_on_positive;
}

/*****************************************************************************************
*  Name:		get_param_quarantine_household_on_traced
*  Description: Gets the value of an int parameter
******************************************************************************************/
int get_param_quarantine_household_on_traced(parameters *params)
{
    return params->quarantine_household_on_traced;
}

/*****************************************************************************************
*  Name:		get_param_quarantine_household_contacts_on_positive
*  Description: Gets the value of an int parameter
******************************************************************************************/
int get_param_quarantine_household_contacts_on_positive(parameters *params)
{
    return params->quarantine_household_contacts_on_positive;
}

/*****************************************************************************************
*  Name:		get_param_test_on_symptoms
*  Description: Gets the value of an int parameter
******************************************************************************************/
int get_param_test_on_symptoms(parameters *params)
{
    return params->test_on_symptoms;
}

/*****************************************************************************************
*  Name:		get_param_test_on_traced
*  Description: Gets the value of an int parameter
******************************************************************************************/
int get_param_test_on_traced(parameters *params)
{
    return params->test_on_traced;
}

/*****************************************************************************************
*  Name:		get_param_test_insensitive_period
*  Description: Gets the value of an int parameter
******************************************************************************************/
int get_param_test_insensitive_period(parameters *params)
{
    return params->test_insensititve_period;
}

/*****************************************************************************************
*  Name:		get_param_test_result_wait
*  Description: Gets the value of an int parameter
******************************************************************************************/
int get_param_test_result_wait(parameters *params)
{
    return params->test_result_wait;
}

/*****************************************************************************************
*  Name:		get_param_test_order_wait
*  Description: Gets the value of an int parameter
******************************************************************************************/
int get_param_test_order_wait(parameters *params)
{
    return params->test_order_wait;
}

/*****************************************************************************************
*  Name:		get_param_app_users_fraction
*  Description: Gets the value of double parameter
******************************************************************************************/
double get_param_app_users_fraction(parameters *params)
{
    return params->app_users_fraction;
}

/*****************************************************************************************
*  Name:		get_param_app_turned_on
*  Description: Gets the value of an int parameter
******************************************************************************************/
int get_param_app_turned_on(parameters *params)
{
    return params->app_turned_on;
}

/*****************************************************************************************
*  Name:		get_param_app_turn_on_time
*  Description: Gets the value of an int parameter
******************************************************************************************/
int get_param_app_turn_on_time(parameters *params)
{
    return params->app_turn_on_time;
}

/*****************************************************************************************
*  Name:		get_param_seasonal_flu_rate
*  Description: Gets the value of double parameter
******************************************************************************************/
double get_param_seasonal_flu_rate(parameters *params)
{
    return params->seasonal_flu_rate;
}

/*****************************************************************************************
*  Name:		get_param_lockdown_work_network_multiplier
*  Description: Gets the value of double parameter
******************************************************************************************/
double get_param_lockdown_work_network_multiplier(parameters *params)
{
    return params->lockdown_work_network_multiplier;
}

/*****************************************************************************************
*  Name:		get_param_lockdown_random_network_multiplier
*  Description: Gets the value of double parameter
******************************************************************************************/
double get_param_lockdown_random_network_multiplier(parameters *params)
{
    return params->lockdown_random_network_multiplier;
}

/*****************************************************************************************
*  Name:		get_param_lockdown_house_interaction_multiplier
*  Description: Gets the value of double parameter
******************************************************************************************/
double get_param_lockdown_house_interaction_multiplier(parameters *params)
{
    return params->lockdown_house_interaction_multiplier;
}

/*****************************************************************************************
*  Name:		get_param_lockdown_time_on
*  Description: Gets the value of an int parameter
******************************************************************************************/
int get_param_lockdown_time_on(parameters *params)
{
    return params->lockdown_time_on;
}

/*****************************************************************************************
*  Name:		get_param_lockdown_time_off
*  Description: Gets the value of an int parameter
******************************************************************************************/
int get_param_lockdown_time_off(parameters *params)
{
    return params->lockdown_time_off;
}

/*****************************************************************************************
*  Name:		get_param_lockdown_on
*  Description: Gets the value of an int parameter
******************************************************************************************/
int get_param_lockdown_on(parameters *params)
{
    return params->lockdown_on;
}

/*****************************************************************************************
*  Name:		get_param_testing_symptoms_time_on
*  Description: Gets the value of an int parameter
******************************************************************************************/
int get_param_symptoms_time_on(parameters *params)
{
    return params->testing_symptoms_time_on;
}

/*****************************************************************************************
*  Name:		get_param_testing_symptoms_time_off
*  Description: Gets the value of an int parameter
******************************************************************************************/
int get_param_testing_symptoms_time_off(parameters *params)
{
    return params->lockdown_time_off;
}


/*****************************************************************************************
*  Name:        set_param_rng_seed
*  Description: Sets the value of parameter
******************************************************************************************/
int set_param_rng_seed(parameters *params, double value)
{
    params->rng_seed = value;
    return TRUE;
}

/*****************************************************************************************
*  Name:        set_param_param_id
*  Description: Sets the value of parameter
******************************************************************************************/
int set_param_param_id(parameters *params, double value)
{
    params->param_id = value;
    return TRUE;
}

/*****************************************************************************************
*  Name:        set_param_n_total
*  Description: Sets the value of parameter
******************************************************************************************/
int set_param_n_total(parameters *params, double value)
{
    params->n_total = value;
    return TRUE;
}

/*****************************************************************************************
*  Name:        set_param_days_of_interactions
*  Description: Sets the value of parameter
******************************************************************************************/
int set_param_days_of_interactions(parameters *params, double value)
{
    params->days_of_interactions = value;
    return TRUE;
}

/*****************************************************************************************
*  Name:        set_param_mean_random_interactions
*  Description: Sets the value of parameter in array
******************************************************************************************/
int set_param_mean_random_interactions(parameters *params, double value, int idx)
{
    if (idx >= N_AGE_TYPES) return FALSE;
    params->mean_random_interactions[idx] = value;
    return TRUE;
}

/*****************************************************************************************
*  Name:        set_param_sd_random_interactions
*  Description: Sets the value of parameter in array
******************************************************************************************/
int set_param_sd_random_interactions(parameters *params, double value, int idx)
{
    if (idx >= N_AGE_TYPES) return FALSE;
    params->sd_random_interactions[idx] = value;
    return TRUE;
}

/*****************************************************************************************
*  Name:        set_param_random_interaction_distribution
*  Description: Sets the value of parameter
******************************************************************************************/
int set_param_random_interaction_distribution(parameters *params, int value)
{
    params->random_interaction_distribution = value;
    return TRUE;
}

/*****************************************************************************************
*  Name:        set_param_mean_work_interactions
*  Description: Sets the value of parameter in array
******************************************************************************************/
int set_param_mean_work_interactions(parameters *params, double value, int idx)
{
    if (idx >= N_WORK_NETWORKS) return FALSE;
    params->mean_work_interactions[idx] = value;
    return TRUE;
}

/*****************************************************************************************
*  Name:        set_param_daily_fraction_work
*  Description: Sets the value of parameter
******************************************************************************************/
int set_param_daily_fraction_work(parameters *params, double value)
{
    params->daily_fraction_work = value;
    return TRUE;
}

/*****************************************************************************************
*  Name:        set_param_daily_fraction_work_used
*  Description: Sets the value of parameter
******************************************************************************************/
int set_param_daily_fraction_work_used(parameters *params, double value, int idx)
{
    if (idx >= N_WORK_NETWORKS) return FALSE;
    params->daily_fraction_work_used[idx] = value;
    return TRUE;
}

/*****************************************************************************************
*  Name:        set_param_child_network_adults
*  Description: Sets the value of parameter
******************************************************************************************/
int set_param_child_network_adults(parameters *params, double value)
{
    params->child_network_adults = value;
    return TRUE;
}

/*****************************************************************************************
*  Name:        set_param_elderly_network_adults
*  Description: Sets the value of parameter
******************************************************************************************/
int set_param_elderly_network_adults(parameters *params, double value)
{
    params->elderly_network_adults = value;
    return TRUE;
}

/*****************************************************************************************
*  Name:        set_param_mean_infectious_period
*  Description: Sets the value of parameter
******************************************************************************************/
int set_param_mean_infectious_period(parameters *params, double value)
{
    params->mean_infectious_period = value;
    return TRUE;
}

/*****************************************************************************************
*  Name:        set_param_sd_infectious_period
*  Description: Sets the value of parameter
******************************************************************************************/
int set_param_sd_infectious_period(parameters *params, double value)
{
    params->sd_infectious_period = value;
    return TRUE;
}

/*****************************************************************************************
*  Name:        set_param_infectious_rate
*  Description: Sets the value of parameter
******************************************************************************************/
int set_param_infectious_rate(parameters *params, int value) {
   params->infectious_rate = value;
   return TRUE;
}

/*****************************************************************************************
*  Name:        set_param_relative_susceptibility
*  Description: Sets the value of parameter in array
******************************************************************************************/
int set_param_relative_susceptibility(parameters *params, double value, int idx)
{
    if (idx >= N_AGE_GROUPS) return FALSE;
    params->relative_susceptibility[idx] = value;
    return TRUE;
}

/*****************************************************************************************
*  Name:        set_param_adjusted_susceptibility
*  Description: Sets the value of parameter in array
******************************************************************************************/
int set_param_adjusted_susceptibility(parameters *params, double value, int idx)
{
    if (idx >= N_AGE_GROUPS) return FALSE;
    params->adjusted_susceptibility[idx] = value;
    return TRUE;
}

/*****************************************************************************************
*  Name:        set_param_relative_transmission_by_type
*  Description: Sets the value of parameter in array
******************************************************************************************/
int set_param_relative_transmission_by_type(parameters *params, double value, int idx)
{
    if (idx >= N_INTERACTION_TYPES) return FALSE;
    params->relative_transmission_by_type[idx] = value;
    return TRUE;
}

/*****************************************************************************************
*  Name:        set_param_relative_transmission_by_type_used
*  Description: Sets the value of parameter in array
******************************************************************************************/
int set_param_relative_transmission_by_type_used(parameters *params, double value, int idx)
{
    if (idx >= N_INTERACTION_TYPES) return FALSE;
    params->relative_transmission_by_type_used[idx] = value;
    return TRUE;
}

/*****************************************************************************************
*  Name:		set_param_mean_time_to_symptoms
*  Description: Sets the value of parameter
******************************************************************************************/
int set_param_mean_time_to_symptoms( parameters *params, double value )
{
    params->mean_time_to_symptoms = value;
    return TRUE;
}

/*****************************************************************************************
*  Name:		set_param_sd_time_to_symptoms
*  Description: Sets the value of parameter
******************************************************************************************/
int set_param_sd_time_to_symptoms( parameters *params, double value )
{
    params->sd_time_to_symptoms = value;
    return TRUE;
}

/*****************************************************************************************
*  Name:        set_param_hospitalised_fraction
*  Description: Sets the value of parameter in array
******************************************************************************************/
int set_param_hospitalised_fraction(parameters *params, double value, int idx)
{
    if (idx >= N_AGE_GROUPS) return FALSE;
    params->hospitalised_fraction[idx] = value;
    return TRUE;
}

/*****************************************************************************************
*  Name:        set_param_critical_fraction
*  Description: Sets the value of parameter in array
******************************************************************************************/
int set_param_critical_fraction(parameters *params, double value, int idx)
{
    if (idx >= N_AGE_GROUPS) return FALSE;
    params->critical_fraction[idx] = value;
    return TRUE;
}

/*****************************************************************************************
*  Name:        set_param_fatality_fraction
*  Description: Sets the value of parameter in array
******************************************************************************************/
int set_param_fatality_fraction(parameters *params, double value, int idx)
{
    if (idx >= N_AGE_GROUPS) return FALSE;
    params->fatality_fraction[idx] = value;
    return TRUE;
}

/*****************************************************************************************
*  Name:        set_param_mean_time_to_hospital
*  Description: Sets the value of parameter in array
******************************************************************************************/
int set_param_mean_time_to_hospital(parameters *params, double value, int idx)
{
    params->mean_time_to_hospital = value;
    return TRUE;
}

/*****************************************************************************************
*  Name:        set_param_mean_time_to_critical
*  Description: Sets the value of parameter in array
******************************************************************************************/
int set_param_mean_time_to_critical(parameters *params, double value, int idx)
{
    params->mean_time_to_critical = value;
    return TRUE;
}

/*****************************************************************************************
*  Name:        set_param_mean_time_to_recover
*  Description: Sets the value of parameter in array
******************************************************************************************/
int set_param_mean_time_to_recover(parameters *params, double value, int idx)
{
    params->mean_time_to_recover = value;
    return TRUE;
}

/*****************************************************************************************
*  Name:        set_param_sd_time_to_recover
*  Description: Sets the value of parameter in array
******************************************************************************************/
int set_param_sd_time_to_recover(parameters *params, double value, int idx)
{
    params->sd_time_to_recover = value;
    return TRUE;
}

/*****************************************************************************************
*  Name:        set_param_mean_time_to_death
*  Description: Sets the value of parameter in array
******************************************************************************************/
int set_param_mean_time_to_death(parameters *params, double value, int idx)
{
    params->mean_time_to_death = value;
    return TRUE;
}

/*****************************************************************************************
*  Name:        set_param_household_size
*  Description: Sets the value of parameter in array
******************************************************************************************/
int set_param_household_size(parameters *params, double value, int idx)
{
    if (idx >= N_HOUSEHOLD_MAX) return FALSE;
    params->household_size[idx] = value;
    return TRUE;
}

/*****************************************************************************************
*  Name:        set_param_population_group
*  Description: Sets the value of parameter in array
******************************************************************************************/
int set_param_population_group(parameters *params, double value, int idx)
{
    if (idx >= N_AGE_GROUPS) return FALSE;
    params->population_group[idx] = value;
    return TRUE;
}

/*****************************************************************************************
*  Name:        set_param_population_type
*  Description: Sets the value of parameter in array
******************************************************************************************/
int set_param_population_type(parameters *params, double value, int idx)
{
    if (idx >= N_AGE_TYPES) return FALSE;
    params->population_type[idx] = value;
    return TRUE;
}

/*****************************************************************************************
*  Name:        set_param_fraction_asymptomatic
*  Description: Sets the value of parameter in array
******************************************************************************************/
int set_param_fraction_asymptomatic(parameters *params, double value, int idx)
{
    if (idx >= N_AGE_GROUPS) return FALSE;
    params->fraction_asymptomatic[idx] = value;
    return TRUE;
}

/*****************************************************************************************
*  Name:        set_param_asymptomatic_infectious_factor
*  Description: Sets the value of parameter in array
******************************************************************************************/
int set_param_asymptomatic_infectious_factor(parameters *params, double value, int idx)
{
    params->asymptomatic_infectious_factor = value;
    return TRUE;
}

/*****************************************************************************************
*  Name:        set_param_mean_asymptomatic_infectious_factor
*  Description: Sets the value of parameter in array
******************************************************************************************/
int set_param_mean_asymptomatic_to_recovery(parameters *params, double value, int idx)
{
    params->mean_asymptomatic_to_recovery = value;
    return TRUE;
}

/*****************************************************************************************
*  Name:        set_param_sd_asymptomatic_infectious_factor
*  Description: Sets the value of parameter in array
******************************************************************************************/
int set_param_sd_asymptomatic_to_recovery(parameters *params, double value, int idx)
{
    params->sd_asymptomatic_to_recovery = value;
    return TRUE;
}

/*****************************************************************************************
*  Name:        set_param_quarantined_daily_interactions
*  Description: Sets the value of parameter in array
******************************************************************************************/
int set_param_quarantined_daily_interactions(parameters *params, int value, int idx)
{
    params->quarantined_daily_interactions = value;
    return TRUE;
}

/*****************************************************************************************
*  Name:        set_param_hospitalised_daily_interactions
*  Description: Sets the value of parameter in array
******************************************************************************************/
int set_param_hospitalised_daily_interactions(parameters *params, int value, int idx)
{
    params->hospitalised_daily_interactions = value;
    return TRUE;
}

/*****************************************************************************************
*  Name:        set_param_quarantine_days
*  Description: Sets the value of parameter
******************************************************************************************/
int set_param_quarantine_days(parameters *params, int value )
{
    params->quarantine_days = value;
    return TRUE;
}

/*****************************************************************************************
*  Name:        set_param_self_quarantine_fraction
*  Description: Sets the value of parameter
******************************************************************************************/
int set_param_self_quarantine_fraction(parameters *params, double value)
{
    params->self_quarantine_fraction = value;
    return TRUE;
}

/*****************************************************************************************
*  Name:        set_param_trace_on_symptoms
*  Description: Sets the value of parameter
******************************************************************************************/
int set_param_trace_on_symptoms(parameters *params, int value) {
   params->trace_on_symptoms = value;
   return TRUE;
}

/*****************************************************************************************
*  Name:        set_param_trace_on_positive
*  Description: Sets the value of parameter
******************************************************************************************/
int set_param_trace_on_positive(parameters *params, int value) {
   params->trace_on_positive = value;
   return TRUE;
}

/*****************************************************************************************
*  Name:        set_param_quarantine_length_self
*  Description: Sets the value of parameter
******************************************************************************************/
int set_param_quarantine_length_self(parameters *params, int value) {
   params->quarantine_length_self = value;
   return TRUE;
}

/*****************************************************************************************
*  Name:        set_param_quarantine_length_traced
*  Description: Sets the value of parameter
******************************************************************************************/
int set_param_quarantine_length_traced(parameters *params, int value) {
   params->quarantine_length_traced = value;
   return TRUE;
}

/*****************************************************************************************
*  Name:        set_param_quarantine_length_positive
*  Description: Sets the value of parameter
******************************************************************************************/
int set_param_quarantine_length_positive(parameters *params, int value) {
   params->quarantine_length_positive = value;
   return TRUE;
}

/*****************************************************************************************
*  Name:        set_param_quarantine_dropout_self
*  Description: Sets the value of parameter
******************************************************************************************/
int set_param_quarantine_dropout_self(parameters *params, double value) {
   params->quarantine_dropout_self = value;
   return TRUE;
}

/*****************************************************************************************
*  Name:        set_param_quarantine_dropout_traced
*  Description: Sets the value of parameter
******************************************************************************************/
int set_param_quarantine_dropout_traced(parameters *params, double value) {
   params->quarantine_dropout_traced = value;
   return TRUE;
}

/*****************************************************************************************
*  Name:        set_param_quarantine_dropout_positive
*  Description: Sets the value of parameter
******************************************************************************************/
int set_param_quarantine_dropout_positive(parameters *params, double value) {
   params->quarantine_dropout_positive = value;
   return TRUE;
}

/*****************************************************************************************
*  Name: 		set_param_quarantine_on_traced
*  Description: Sets the value of parameter
******************************************************************************************/
int set_param_quarantine_on_traced( parameters *params, int value )
{
    params->quarantine_on_traced = value;
    return TRUE;
}

/*****************************************************************************************
*  Name:		set_param_traceable_interaction_fractio
*  Description: Sets the value of parameter
******************************************************************************************/
int set_param_traceable_interaction_fraction( parameters *params, double value )
{
    params->traceable_interaction_fraction = value;
    return TRUE;
}

/*****************************************************************************************
*  Name:		set_param_tracing_network_depth
*  Description: Sets the value of parameter
******************************************************************************************/
int set_param_tracing_network_depth( parameters *params, int value )
{
    params->tracing_network_depth = value;
    return TRUE;
}

/*****************************************************************************************
*  Name:		set_param_allow_clinical_diagnosis
*  Description: Sets the value of parameter
******************************************************************************************/
int set_param_allow_clinical_diagnosis( parameters *params, int value )
{
    params->allow_clinical_diagnosis = value;
    return TRUE;
}

/*****************************************************************************************
*  Name:		set_param_quarantine_household_on_symptoms
*  Description: Sets the value of parameter
******************************************************************************************/
int set_param_quarantine_household_on_symptoms( parameters *params, int value )
{
    params->quarantine_household_on_symptoms = value;
    return TRUE;
}

/*****************************************************************************************
*  Name:		set_param_quarantine_household_on_positive
*  Description: Sets the value of parameter
******************************************************************************************/
int set_param_quarantine_household_on_positive( parameters *params, int value )
{
    params->quarantine_household_on_positive = value;
    return TRUE;
}

/*****************************************************************************************
*  Name:		set_param_quarantine_household_on_traced
*  Description: Sets the value of parameter
******************************************************************************************/
int set_param_quarantine_household_on_traced( parameters *params, int value )
{
    params->quarantine_household_on_traced = value;
    return TRUE;
}

/*****************************************************************************************
*  Name:		set_param_quarantine_household_contacts_on_positive
*  Description: Sets the value of parameter
******************************************************************************************/
int set_param_quarantine_household_contacts_on_positive( parameters *params, int value )
{
    params->quarantine_household_contacts_on_positive = value;
    return TRUE;
}

/*****************************************************************************************
*  Name:		set_param_test_on_symptoms
*  Description: Sets the value of parameter
******************************************************************************************/
int set_param_test_on_symptoms(parameters *params, int value) {
   params->test_on_symptoms = value;
   return TRUE;
}

/*****************************************************************************************
*  Name:		set_param_test_on_traced
*  Description: Sets the value of parameter
******************************************************************************************/
int set_param_test_on_traced( parameters *params, int value )
{
    params->test_on_traced = value;
    return TRUE;
}

/*****************************************************************************************
*  Name:		set_param_test_result_wait
*  Description: Sets the value of parameter
******************************************************************************************/
int set_param_test_result_wait( parameters *params, int value )
{
    params->test_result_wait = value;
    return TRUE;
}

/*****************************************************************************************
*  Name:		set_param_test_order_wait
*  Description: Sets the value of parameter
******************************************************************************************/
int set_param_test_order_wait( parameters *params, int value )
{
    params->test_order_wait = value;
    return TRUE;
}

/*****************************************************************************************
*  Name:		set_param_app_users_fraction
*  Description: Sets the value of parameter
******************************************************************************************/
int set_param_app_users_fraction( parameters *params, double value )
{
    params->app_users_fraction = value;
    return TRUE;
}

/*****************************************************************************************
*  Name:		set_param_app_turn_on_time
*  Description: Sets the value of parameter
******************************************************************************************/
int set_param_app_turn_on_time( parameters *params, int value )
{
    params->app_turn_on_time = value;
    return TRUE;
}

/*****************************************************************************************
*  Name:		set_param_app_turned_on
*  Description: Sets the value of parameter
******************************************************************************************/
int set_param_app_turned_on( parameters *params, int value )
{
    params->app_turned_on = value;
    return TRUE;
}

/*****************************************************************************************
*  Name:		set_param_seasonal_flu_rate
*  Description: Sets the value of parameter
******************************************************************************************/
int set_param_seasonal_flu_rate( parameters *params, double value )
{
    params->seasonal_flu_rate = value;
    return TRUE;
}

/*****************************************************************************************
*  Name:		set_param_lockdown_work_network_multiplier
*  Description: Sets the value of parameter
******************************************************************************************/
int set_param_lockdown_work_network_multiplier( parameters *params, double value )
{
    params->lockdown_work_network_multiplier = value;
    return TRUE;
}
/*****************************************************************************************
*  Name:		set_param_lockdown_random_network_multiplier
*  Description: Sets the value of parameter
******************************************************************************************/
int set_param_random_network_multiplier( parameters *params, double value )
{
    params->lockdown_random_network_multiplier = value;
    return TRUE;
}

/*****************************************************************************************
*  Name:		set_param_lockdown_house_interaction_multiplier
*  Description: Sets the value of parameter
******************************************************************************************/
int set_param_lockdown_house_interaction_multiplier( parameters *params, double value )
{
    params->lockdown_house_interaction_multiplier = value;
    return TRUE;
}

/*****************************************************************************************
*  Name:		set_param_lockdown_time_on
*  Description: Sets the value of parameter
******************************************************************************************/
int set_param_lockdown_time_on( parameters *params, int value )
{
    params->lockdown_time_on = value;
    return TRUE;
}

/*****************************************************************************************
*  Name:		set_param_lockdown_time_off
*  Description: Sets the value of parameter
******************************************************************************************/
int set_param_lockdown_time_off( parameters *params, int value )
{
    params->lockdown_time_off = value;
    return TRUE;
}

/*****************************************************************************************
*  Name:		set_param_lockdown_on
*  Description: Carries out checks on the input parameters
******************************************************************************************/
int set_param_lockdown_on( parameters *params, int value )
{
    params->lockdown_on = TRUE;
    return TRUE;
}

/*****************************************************************************************
*  Name:		set_param_testing_symptoms_time_on
*  Description: Sets the value of parameter
******************************************************************************************/
int set_param_testing_symptoms_time_on( parameters *params, int value )
{
    params->testing_symptoms_time_on = value;
    return TRUE;
}

/*****************************************************************************************
*  Name:		set_param_testing_symptoms_time_off
*  Description: Sets the value of parameter
******************************************************************************************/
int set_param_testing_symptoms_time_off( parameters *params, int value )
{
    params->testing_symptoms_time_off = value;
    return TRUE;
}
%}
