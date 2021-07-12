/*
 * hospital.c
 *
 *  Created on: 30 Mar 2020
 *      Author: vuurenk
 */

#include "hospital.h"
#include "constant.h"
#include "utilities.h"
#include "network.h"
#include "model.h"
#include "disease.h"
#include "interventions.h"
#include "list.h"

/*****************************************************************************************
*  Name:		initialise_hospital
*  Description: initialises and individual at the start of the simulation, note can
*  				only be called once per individual
*  Returns:		void
******************************************************************************************/
void initialise_hospital(
	hospital *hospital,
	parameters *params,
	int hdx
)
{
	int ward_type, ward_idx, hcw_type;

	if( hospital->hospital_idx != 0 )
		print_exit( "Attempted to initialise a hospital more than once.");

	hospital->hospital_idx  = hdx;

	hospital->wards = calloc( N_HOSPITAL_WARD_TYPES, sizeof(ward*) );

	//Initialise wards based on the size of the wards and their assigned healthcare workers.
	for( ward_type = 0; ward_type < N_HOSPITAL_WARD_TYPES; ward_type++ )
	{
		hospital->n_wards[ward_type] = params->n_wards[ward_type];
		hospital->wards[ward_type] = calloc( params->n_wards[ward_type], sizeof(ward) );

		//Initialise ward waiting list for each ward type - i.e. general and ICU.
		hospital->waiting_list[ward_type] = malloc( sizeof (list) );
		initialise_list( hospital->waiting_list[ward_type] );

		for( hcw_type = 0; hcw_type < N_WORKER_TYPES; hcw_type++ )
			hospital->n_workers[hcw_type] += params->n_wards[ward_type] * params->n_hcw_per_ward[ward_type][hcw_type];
		
		for( ward_idx = 0; ward_idx < params->n_wards[ward_type]; ward_idx++ )
			initialise_ward( &(hospital->wards[ward_type][ward_idx]),
							 ward_idx, ward_type,
							 params->n_ward_beds[ward_type],
							 params->n_hcw_per_ward[ward_type][DOCTOR],
							 params->n_hcw_per_ward[ward_type][NURSE] );
	}
}

/*****************************************************************************************
*  Name:		set_up_hospital_networks
*  Description: for all hospitals in simulation calls setup functions for related networks
*  Returns:		void
******************************************************************************************/
void set_up_hospital_networks( model *model )
{
	int hospital_idx, hcw_idx, n_healthcare_workers, ward_idx, ward_type;
	long *healthcare_workers;
	hospital *hospital;

	for( hospital_idx = 0; hospital_idx < model->params->n_hospitals; hospital_idx++ )
	{
		hospital = &model->hospitals[hospital_idx];
		//Setup hospital workplace network.
		n_healthcare_workers = 0;
		healthcare_workers = calloc( hospital->n_workers[DOCTOR] + hospital->n_workers[NURSE], sizeof(long) );

		//Setup HCW-patient networks for all wards.
		for ( ward_type = 0; ward_type < N_HOSPITAL_WARD_TYPES; ward_type++ )
		{
			for( ward_idx = 0; ward_idx < hospital->n_wards[ward_type]; ward_idx++ )
			{
				for( hcw_idx = 0; hcw_idx < hospital->wards[ward_type][ward_idx].n_worker[DOCTOR]; hcw_idx++ )
					healthcare_workers[n_healthcare_workers++] = hospital->wards[ward_type][ward_idx].doctors[hcw_idx].pdx;

				for( hcw_idx = 0; hcw_idx < hospital->wards[ward_type][ward_idx].n_worker[NURSE]; hcw_idx++ )
					healthcare_workers[n_healthcare_workers++] = hospital->wards[ward_type][ward_idx].nurses[hcw_idx].pdx;

				set_up_ward_networks( model, &(hospital->wards[ward_type][ward_idx]), model->params->max_hcw_daily_interactions );
			}
		}

		//Setup HCW workplace network.
		hospital->hospital_workplace_network = calloc( 1, sizeof( network ));
		hospital->hospital_workplace_network = add_new_network( model, n_healthcare_workers, HOSPITAL_WORK );
		hospital->hospital_workplace_network->skip_hospitalised = TRUE;
		hospital->hospital_workplace_network->skip_quarantined  = TRUE;
		hospital->hospital_workplace_network->construction      = NETWORK_CONSTRUCTION_WATTS_STROGATZ;
		hospital->hospital_workplace_network->daily_fraction    = 1.0;

		build_watts_strogatz_network( hospital->hospital_workplace_network, n_healthcare_workers, model->params->hcw_mean_work_interactions, model->params->work_network_rewire, FALSE );
		relabel_network( hospital->hospital_workplace_network, healthcare_workers );

		free( healthcare_workers );
	}
}

/*****************************************************************************************
*  Name:		rebuild_healthcare_worker_patient_networks
*  Description: rebuilds hcw->patient networks for each ward in a hospital
*  Returns:     void
******************************************************************************************/
void rebuild_healthcare_worker_patient_networks( model *model, hospital *hospital )
{
	int ward_type, ward_idx;

	for( ward_type = 0; ward_type < N_HOSPITAL_WARD_TYPES; ward_type++ )
		for( ward_idx = 0; ward_idx < hospital->n_wards[ward_type]; ward_idx++ )
			build_ward_networks( model, &(hospital->wards[ward_type][ward_idx]) );
}

/*****************************************************************************************
*  Name:		add_hospital_network_interactions
*  Description: Adds the daily interactions to all individual from hospital networks
*  Returns:     void
******************************************************************************************/
void add_hospital_network_interactions(model *model, hospital *hospital)
{
	int ward_type, ward_idx;
	add_interactions_from_network( model, hospital->hospital_workplace_network );
	for( ward_type = 0; ward_type < N_HOSPITAL_WARD_TYPES; ward_type++ )
	{
		for( ward_idx = 0; ward_idx < hospital->n_wards[ward_type]; ward_idx++ )
		{
			if( hospital->wards[ward_type][ward_idx].doctor_patient_network->n_edges > 0 )
				add_interactions_from_network( model, hospital->wards[ward_type][ward_idx].doctor_patient_network );
			if( hospital->wards[ward_type][ward_idx].nurse_patient_network->n_edges > 0 )
				add_interactions_from_network( model, hospital->wards[ward_type][ward_idx].nurse_patient_network );
		}
	}
}

/*****************************************************************************************
*  Name:		add_healthcare_worker_to_hopsital
*  Description: Adds healthcare worker ids to hospital wards
*  Returns:     void
******************************************************************************************/
void add_healthcare_worker_to_hospital( hospital *hospital, individual* indiv, int hcw_type )
{
	int ward_type, ward_idx;
	int hcw_allocated;

	hcw_allocated = FALSE;

	for( ward_type = 0; ward_type < N_HOSPITAL_WARD_TYPES; ward_type++ )
	{
		for( ward_idx = 0; ward_idx < hospital->n_wards[ward_type]; ward_idx++ )
		{
			//Breaks if the number of assigned healthcare workers is greater than the max.
			if( hospital->wards[ward_type][ward_idx].n_worker[hcw_type] < hospital->wards[ward_type][ward_idx].n_max_hcw[hcw_type] )
			{
				hcw_allocated = TRUE;
				break;
			}
		}
		if( hcw_allocated == TRUE)
			break;
	}

	if( hcw_allocated == FALSE)
		print_exit( "Attempted to allocated more healthcare workers than max number of doctors to hospital." );

	if( hcw_type == DOCTOR )
		initialise_doctor( &(hospital->wards[ward_type][ward_idx].doctors[hospital->wards[ward_type][ward_idx].n_worker[DOCTOR]++]),
				indiv, hospital->hospital_idx, ward_idx, ward_type);
	else
		initialise_nurse( &(hospital->wards[ward_type][ward_idx].nurses[hospital->wards[ward_type][ward_idx].n_worker[NURSE]++]),
				indiv, hospital->hospital_idx, ward_idx, ward_type);
}

/*****************************************************************************************
*  Name:		healthcare_worker_working
*  Description: Checks that a healthcare worker is capable of working in a hospital.
*  Returns:     int
******************************************************************************************/
int healthcare_worker_working( individual *indiv )
{
	if( indiv->status == DEATH || is_in_hospital( indiv ) || indiv->quarantined == TRUE )
		return FALSE;

	return TRUE;
}

int individual_eligible_to_become_healthcare_worker( individual *indiv )
{
	return indiv->worker_type == NOT_HEALTHCARE_WORKER &&
			indiv->age_group > AGE_10_19 &&
			indiv->age_group < AGE_60_69 &&
			indiv->occupation_network == WORKING_NETWORK
			;
}

/*****************************************************************************************
*  Name:		transition_one_hospital_event
*  Description: Generic function for updating an individual with their next hospital
*				event and adding the applicable events
*  Returns:		void
******************************************************************************************/
void transition_one_hospital_event(
		model *model,
		individual *indiv,
		int from,
		int to,
		int edge
)
{
	indiv->hospital_state = from;
	if( from != NO_EVENT )
		indiv->infection_events->times[from] = model->time;
	if( indiv->current_hospital_event != NULL )
		remove_event_from_event_list( model, indiv->current_hospital_event );
	if( indiv->next_hospital_event != NULL )
		indiv->current_hospital_event = indiv->next_hospital_event;

	if( to != NO_EVENT )
	{
		indiv->infection_events->times[to]     = model->time + ifelse( edge == NO_EDGE, 0, sample_transition_time( model, edge ) );
		indiv->next_hospital_event = add_individual_to_event_list( model, to, indiv, indiv->infection_events->times[to], NULL );
	}
}

/*****************************************************************************************
*  Name:		transition_to_waiting
*  Description: Transitions a severely symptomatic (HOSPITATLISED) individual from the
*               general populace to the admissions list for the hospital. Also applies
*               interventions to the patient.
*  Returns:		void
******************************************************************************************/
void transition_to_waiting( model *model, individual *indiv )
{ 
	set_waiting( indiv, model->params, 1);
}

/*****************************************************************************************
*  Name:		transition_to_general
*  Description: Transitions a severely symptomatic / critical recovering individual from the admissions list to a
*               general ward.
*               This only occurs if there is enough space in the general wards. However, the scheduler
*               should have only scheduled this transition if there is space for it to happen. So if it
*               fails then an error is thrown to show the scheduler was incorrect.
*  Returns:		void
******************************************************************************************/
void transition_to_general( model *model, individual *indiv )
{   
	if( indiv->status == HOSPITALISED_RECOVERING && indiv->hospital_state == ICU )
		remove_patient_from_ward( &model->hospitals[indiv->hospital_idx].wards[COVID_ICU][indiv->ward_idx], indiv );
	
	if( add_patient_to_hospital( model, indiv, COVID_GENERAL) )
		set_general_admission( indiv, model->params, 1);
	else
		print_exit("Scheduled a transition to the general hospital location, but failed to add to a general ward.");
}

/*****************************************************************************************
*  Name:		transition_to_ICU
*  Description: Transitions a critically ill individual to the ICU from either a general
*               ward or the admissions list. This only occurs if there is enough space in
*               the ICU. However, the scheduler should have only scheduled this transition
*               if there is space for it to happen. So if it fails then an error is thrown
*               to show the scheduler was incorrect.
*  Returns:		void
******************************************************************************************/
void transition_to_icu( model *model, individual *indiv )
{
	if( indiv->hospital_state == GENERAL )
		remove_patient_from_ward( &model->hospitals[indiv->hospital_idx].wards[COVID_GENERAL][indiv->ward_idx], indiv );

	if( add_patient_to_hospital( model, indiv, COVID_ICU) )        
		set_icu_admission( indiv, model->params, 1);
	else
		print_exit( "Scheduled a transition to the ICU hospital location, but failed to add to an ICU ward." );    
}
/*****************************************************************************************
*  Name:		transition_to_mortuary
*  Description: Transitions a newly deceased person from either the general ward or ICU
*               into a "mortuary" to free space for new patients. This always succeeds in
*               assigning the patient location.
*  Returns:		void
******************************************************************************************/
void transition_to_mortuary( model *model, individual *indiv )
{
	set_mortuary_admission( indiv, model->params, 1);
	release_patient_from_hospital( indiv, &(model->hospitals[indiv->hospital_idx]) );
	transition_one_hospital_event( model, indiv, MORTUARY, NO_EVENT, NO_EDGE );
}

/*****************************************************************************************
*  Name:		transition_to_discharged
*  Description: Transitions a recovered individual from either the general wards or ICU
*               back into the general population. This always succeeds in assigning the patient
*               location.
*  Returns:		void
******************************************************************************************/
void transition_to_discharged( model *model, individual *indiv )
{
	set_discharged( indiv, model->params, 1);
	release_patient_from_hospital( indiv, &(model->hospitals[indiv->hospital_idx]) );
	transition_one_hospital_event( model, indiv, DISCHARGED, NO_EVENT, NO_EDGE );
}

/*****************************************************************************************
*  Name:		add_patient_to_hospital
*  Description: Checks to see if there is an available ward in a patient's hospital. Returns
*               true if yes, false if no.
*  Returns:		int
******************************************************************************************/
int add_patient_to_hospital( model* model, individual *indiv, int required_ward )
{
	int ward_idx, assigned_ward_idx;
	hospital *assigned_hospital;

	assigned_hospital = &(model->hospitals[indiv->hospital_idx]); 
	assigned_ward_idx = 0;

	//Check for ward with the smallest number of patients.
	for( ward_idx = 1; ward_idx < assigned_hospital->n_wards[required_ward]; ward_idx++ )
		if( assigned_hospital->wards[required_ward][assigned_ward_idx].patients->size > assigned_hospital->wards[required_ward][ward_idx].patients->size )
			assigned_ward_idx = ward_idx;

	// If the patient can be added to the ward, do so.
	if( add_patient_to_ward( &(assigned_hospital->wards[required_ward][assigned_ward_idx]), indiv ) )
	{  
		indiv->ward_idx  = assigned_ward_idx;
		indiv->ward_type = required_ward;

		if( indiv->idx != list_pop( assigned_hospital->waiting_list[required_ward] ) )
			print_exit( "ERROR: Attempted to add patient to hospital ward who is not at the top of the waiting list.");

		return TRUE;
	}
	return FALSE;
}

/*****************************************************************************************
*  Name:		hospital_waiting_list_transition_scheduler
*  Description: Checks ward availability and attempts to schedule transitions for patients.
*               At this point, we know whether a person would get worse or better depending
*               on the availability of care, so the predict_patient_progression() function is
*               also called here.
*  Returns:		void
******************************************************************************************/
void hospital_waiting_list_transition_scheduler( model *model, int hospital_state )
{
	int idx, hospital_idx, ward_type;
	double patient_waiting_modifier;
	individual *indiv;
	hospital* hospital;

	ward_type = EVENT_TYPE_TO_WARD_MAP[ hospital_state ];

	for( hospital_idx = 0; hospital_idx < model->params->n_hospitals; hospital_idx++ )
	{
		hospital = &( model->hospitals[hospital_idx] );
		//Iterate backwards over waiting list. This is because the last event added to a event list is the first
		//event that is iterated over in the transition_events function. So by iterating over this waiting list backwards
		//ensures that the first patient in the list will have their transition scheduled to happen first in the
		//transition_events function.
		for( idx = hospital->waiting_list[ ward_type ]->size - 1; idx >= 0; idx-- )
		{
			indiv = &model->population[ list_element_at( hospital->waiting_list[ ward_type ], idx ) ];
			//If beds are available, add the patient to the required ward and schedule the relevant transition.
			//Otherwise, apply a waiting modifier to signify a worsening progression.
			if ( hospital_available_beds( hospital, ward_type ) - idx > 0 )
			{
				transition_one_hospital_event( model, indiv, indiv->hospital_state, hospital_state, NO_EDGE );
				patient_waiting_modifier = 1;
			} else
			{
				if( not_in_hospital(indiv) )
					transition_one_hospital_event( model, indiv, indiv->hospital_state, WAITING, NO_EDGE );

				if ( indiv->status == HOSPITALISED )
					patient_waiting_modifier = model->params->hospitalised_waiting_mod;
				else
					patient_waiting_modifier = model->params->critical_waiting_mod;
			}
			//Predict the effect of care availability on future disease state.
			predict_patient_disease_progression( model, indiv, patient_waiting_modifier, ward_type );
		}
	}
}

/*****************************************************************************************
*  Name:		swap_waiting_general_and_icu_patients
*  Description: Ensures that patients in the ICU who need to be in the general ward and vice versa
*               swap beds if possible. This is done by remvoving the same number of patients who need
*               to swap from their current ward and then pushing them to the top of the waiting list
*               for the ward they need to move into.
*  Returns:		void
******************************************************************************************/
void swap_waiting_general_and_icu_patients( model *model )
{
	hospital *hospital;
	list *patient_general_list, *patient_icu_list;
	int hospital_idx, ward_idx, patient_idx;

	patient_general_list = malloc(sizeof(list));
	patient_icu_list     = malloc(sizeof(list));

	//For each hospital in the hospital`
	for( hospital_idx = 0; hospital_idx < model->params->n_hospitals; hospital_idx++ )
	{
		hospital = &model->hospitals[hospital_idx];

		initialise_list( patient_general_list );
		initialise_list( patient_icu_list );

		//Create list of all patients in a general ward who need to transition to an ICU ward.
		for( ward_idx = 0; ward_idx < hospital->n_wards[COVID_GENERAL]; ward_idx++ )
			for( patient_idx = 0; patient_idx < hospital->wards[COVID_GENERAL][ward_idx].patients->size; patient_idx++ )
				if( model->population[ list_element_at(hospital->wards[COVID_GENERAL][ward_idx].patients, patient_idx) ].status == CRITICAL )
					list_push_back( list_element_at(hospital->wards[COVID_GENERAL][ward_idx].patients, patient_idx), patient_general_list );

		//Create list of all patients in an ICU ward who need to transition to a general ward..
		for( ward_idx = 0; ward_idx < hospital->n_wards[COVID_ICU]; ward_idx++ )
			for( patient_idx = 0; patient_idx < hospital->wards[COVID_ICU][ward_idx].patients->size; patient_idx++ )
				if( model->population[ list_element_at(hospital->wards[COVID_ICU][ward_idx].patients, patient_idx) ].status == HOSPITALISED_RECOVERING )
					list_push_back( list_element_at(hospital->wards[COVID_ICU][ward_idx].patients, patient_idx), patient_icu_list );

		//Looop through the two lists until reaching the end of one. Each loop, remove patients from
		//their current ward and push to the top of the waiting list for their new ward.
		patient_idx = 0;
		while( patient_idx < patient_general_list->size && patient_idx < patient_icu_list->size ) 
		{
			individual *indiv_general = &model->population[ list_element_at( patient_general_list, patient_idx )];
			individual *indiv_icu   = &model->population[ list_element_at( patient_icu_list, patient_idx )];

            //Remove patients from their current ward.
			remove_patient_from_ward( &(hospital->wards[COVID_GENERAL][indiv_general->ward_idx]), indiv_general);
			remove_patient_from_ward( &(hospital->wards[COVID_ICU][indiv_icu->ward_idx]), indiv_icu);

			//Remove them from their current position in the waiting list for their required ward.
			list_remove_element( indiv_general->idx, hospital->waiting_list[COVID_ICU] );
			list_remove_element( indiv_icu->idx, hospital->waiting_list[COVID_GENERAL] );

			//Push them to the front of the waiting list for their required ward.
			list_push_front( indiv_general->idx, hospital->waiting_list[COVID_ICU] ) ;
			list_push_front( indiv_icu->idx, hospital->waiting_list[COVID_GENERAL] );

			patient_idx++;
		}

		destroy_list( patient_general_list );
		destroy_list( patient_icu_list );
	}

	free( patient_general_list );
	free( patient_icu_list );
}

/*****************************************************************************************
*  Name:		predict_patient_disease_progression
*  Description: Given the availability of care, predict whether a person gets better or worse
*               once they've transitioned to either a severe (HOSPITALISED) or critical disease
*               state. Then schedule the relevant event transition. This is only done once
*               for each patient when they first enter a HOSPITALISED or CRITICAL state hence the
*               disease_progression_predicted flag.
*  Returns:		void
******************************************************************************************/
void predict_patient_disease_progression( model *model, individual *indiv, double patient_waiting_modifier, int type )
{
	//Check if progression for this particular disease state has already been predicted.
	if( indiv->disease_progression_predicted[type] == FALSE )
	{
		if( type == COVID_GENERAL )
		{
			if( gsl_ran_bernoulli( rng, min(model->params->critical_fraction[ indiv->age_group ] * patient_waiting_modifier, 1) ) )
			{
				if( gsl_ran_bernoulli( rng, model->params->location_death_icu[ indiv->age_group ] ) )
					transition_one_disese_event( model, indiv, HOSPITALISED, CRITICAL, HOSPITALISED_CRITICAL );
				else
					transition_one_disese_event( model, indiv, HOSPITALISED, DEATH, HOSPITALISED_CRITICAL );
			}
			else
				transition_one_disese_event( model, indiv, HOSPITALISED, RECOVERED, HOSPITALISED_RECOVERED);
		} 
		else if( type == COVID_ICU )
		{
			if( gsl_ran_bernoulli( rng, min(model->params->fatality_fraction[ indiv->age_group ] * patient_waiting_modifier, 1) ) )
				transition_one_disese_event(model, indiv, CRITICAL, DEATH, CRITICAL_DEATH );
			else
				transition_one_disese_event( model, indiv, CRITICAL, HOSPITALISED_RECOVERING, CRITICAL_HOSPITALISED_RECOVERING );
		}
		indiv->disease_progression_predicted[type] = TRUE;
	}
}

/*****************************************************************************************
*  Name:		find_least_full_hospital
*  Description: Iterates over hospitals to see which wards have the most beds available.
*               Returns the index of the hospital which has most space.
*  Returns:		int
******************************************************************************************/
int find_least_full_hospital(model* model, int required_ward)
{
	int hospital_idx, available_beds, assigned_hospital_idx;

	assigned_hospital_idx = 0;
	available_beds = hospital_available_beds( &(model->hospitals[assigned_hospital_idx] ), required_ward )
			- model->hospitals[assigned_hospital_idx].waiting_list[required_ward]->size;

	for( hospital_idx = 1; hospital_idx < model->params->n_hospitals; hospital_idx++ )
	{
		int next_available_beds = hospital_available_beds( &(model->hospitals[hospital_idx]), required_ward )
				- model->hospitals[hospital_idx].waiting_list[required_ward]->size;
		
		if( next_available_beds > available_beds )
		{
			assigned_hospital_idx = hospital_idx;
			available_beds = next_available_beds;
		}
	}
	return assigned_hospital_idx;
}

/*****************************************************************************************
*  Name:		hospital_available_beds
*  Description: Determines the number of available beds in all the wards of specified type.
*  Returns:		int
******************************************************************************************/
int hospital_available_beds( hospital* hospital, int ward_type )
{
	int ward_idx;
	int available_beds = 0;

	for( ward_idx = 0; ward_idx < hospital->n_wards[ward_type]; ward_idx++ )
		available_beds += ( ward_available_beds( &(hospital->wards[ward_type][ward_idx]) ));

	return available_beds;
}

/*****************************************************************************************
*  Name:		release_patient_from_hospital
*  Description: Removes an individual from a hospital and sets their ward information to
*               null.
*  Returns:		void
******************************************************************************************/
void release_patient_from_hospital( individual *indiv, hospital *hospital )
{
	remove_if_in_waiting_list( indiv, hospital );

	if( indiv->ward_type != NO_WARD )
		remove_patient_from_ward( &(hospital->wards[indiv->ward_type][indiv->ward_idx]), indiv );

	indiv->ward_type = NO_WARD;
	indiv->ward_idx  = NO_WARD;
}

/*****************************************************************************************
*  Name:		add_patient_to_waiting_list
*  Description: Adds a patient to their required ward and sets their hospital index to the
*               hospital that owns that ward.
*  Returns:		int
******************************************************************************************/

void add_patient_to_waiting_list( individual *indiv, hospital *hospital, int ward_type)
{
	list_push_back( indiv->idx, hospital->waiting_list[ward_type] );
	indiv->hospital_idx = hospital->hospital_idx;
}

/*****************************************************************************************
*  Name:		remove_if_in_waiting_list
*  Description: Checks if a patient is in a waiting list and removes them if so.
*  Returns:		int
******************************************************************************************/
void remove_if_in_waiting_list( individual *indiv, hospital *hospital )
{
	if( list_elem_exists( indiv->idx, hospital->waiting_list[COVID_GENERAL] ))
		list_remove_element( indiv->idx, hospital->waiting_list[COVID_GENERAL] );
	else if( list_elem_exists( indiv->idx, hospital->waiting_list[COVID_ICU] ) )
		list_remove_element( indiv->idx, hospital->waiting_list[COVID_ICU] );
}


/*****************************************************************************************
*  Name:		destroy_hospital
*  Description: Destroys the model structure and releases its memory
******************************************************************************************/
void destroy_hospital( hospital *hospital)
{
	free( hospital->hospital_workplace_network );
	for( int ward_type = 0; ward_type < N_HOSPITAL_WARD_TYPES; ward_type++ )
	{
		for(int ward_idx = 0; ward_idx < hospital->n_wards[ward_type]; ward_idx++ )
			destroy_ward( &(hospital->wards[ward_type][ward_idx]) );

		free( hospital->wards[ward_type] );
	}

	free( hospital->wards );
}
