/*
 * hsopital.c
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
#include "waiting_list.h"

/*****************************************************************************************
*  Name:		initialize_hospital
*  Description: initializes and individual at the start of the simulation, note can
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
        print_exit( "a hospital can only be initialised once!");

    hospital->hospital_idx  = hdx;

    hospital->wards = calloc( N_HOSPITAL_WARD_TYPES, sizeof(ward*) );
    hospital->waiting_list = calloc( N_HOSPITAL_WARD_TYPES, sizeof(waiting_list) );

    for( ward_type = 0; ward_type < N_HOSPITAL_WARD_TYPES; ward_type++ )
    {
        hospital->n_wards[ward_type] = params->n_wards[ward_type];
        hospital->wards[ward_type] = calloc( params->n_wards[ward_type], sizeof(ward) );
        //hospital->waiting_list[ward_type] = initialise_waiting_list();
        initialise_waiting_list( &(hospital->waiting_list[ward_type]) );

        for( hcw_type = 0; hcw_type < N_WORKER_TYPES; hcw_type++ )
            hospital->n_workers[hcw_type] += params->n_wards[ward_type] * params->n_hcw_per_ward[ward_type][hcw_type];
        
        for( ward_idx = 0; ward_idx < params->n_wards[ward_type]; ward_idx++ )
            initialise_ward( &(hospital->wards[ward_type][ward_idx]), ward_idx, ward_type, params->n_ward_beds[ward_type], params->n_hcw_per_ward[ward_type][DOCTOR], params->n_hcw_per_ward[ward_type][NURSE] );
    }
}

/*****************************************************************************************
*  Name:		set_up_hospital_networks
*  Description: calls setup functions for all networks related to the hospital instance
*  Returns:		void
******************************************************************************************/
void set_up_hospital_networks( hospital* hospital, int max_hcw_daily_interactions )
{
    int idx, n_healthcare_workers;
    int ward_idx, ward_type;
    long *healthcare_workers;

    //setup hospital workplace network
    n_healthcare_workers = 0;
    healthcare_workers = calloc( hospital->n_workers[DOCTOR] + hospital->n_workers[NURSE], sizeof(long) );

    //setup hcw -> patient networks for all wards
    for ( ward_type = 0; ward_type < N_HOSPITAL_WARD_TYPES; ward_type++ )
    {
        for( ward_idx = 0; ward_idx < hospital->n_wards[ward_type]; ward_idx++ )
        {
            for( idx = 0; idx < hospital->wards[ward_type][ward_idx].n_worker[DOCTOR]; idx++ )
                healthcare_workers[n_healthcare_workers++] = hospital->wards[ward_type][ward_idx].doctors[idx].pdx;

            for( idx = 0; idx < hospital->wards[ward_type][ward_idx].n_worker[NURSE]; idx++ )
                healthcare_workers[n_healthcare_workers++] = hospital->wards[ward_type][ward_idx].nurses[idx].pdx;

            set_up_ward_networks( &(hospital->wards[ward_type][ward_idx]), max_hcw_daily_interactions );
        }
    }

    //setup hcw workplace network
    hospital->hospital_workplace_network = calloc( 1, sizeof( network ));
    hospital->hospital_workplace_network = new_network( n_healthcare_workers, HOSPITAL_WORK );
    int n_interactions = 20;//TODO: maybe make this number of interactions set via param... and should nurses have more??
    build_watts_strogatz_network( hospital->hospital_workplace_network, n_healthcare_workers, n_interactions, 0.1, TRUE ); //TODO: p_rewire probability higher??
    relabel_network( hospital->hospital_workplace_network, healthcare_workers );

    free( healthcare_workers );
}

void build_hospital_networks( model *model, hospital *hospital )
{
    int ward_type, ward_idx;

    for( ward_type = 0; ward_type < N_HOSPITAL_WARD_TYPES; ward_type++ )
        for( ward_idx = 0; ward_idx < hospital->n_wards[ward_type]; ward_idx++ )
            build_ward_networks( model, &(hospital->wards[ward_type][ward_idx]) );
}

/*****************************************************************************************
*  Name:		add_nurse
*  Description: adds population id of a doctor / nurse to hospital's
*               doctor / nurse population id list
******************************************************************************************/
void add_healthcare_worker_to_hospital(hospital *hospital, long pdx, int hcw_type)
{
    int ward_type, ward_idx;
    int hcw_allocated;

    hcw_allocated = FALSE;

    for( ward_type = 0; ward_type < N_HOSPITAL_WARD_TYPES; ward_type++ )
    {
        for( ward_idx = 0; ward_idx < hospital->n_wards[ward_type]; ward_idx++ )
        {
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
        print_exit( "attempted to allocated more than max number of doctors to hospital!!" );

    //TODO: if statement below not necessary when separate doctor / nurse structs are moved into a hcw struct
    if( hcw_type == DOCTOR )
        initialise_doctor( &(hospital->wards[ward_type][ward_idx].doctors[hospital->wards[ward_type][ward_idx].n_worker[DOCTOR]++]) , pdx, hospital->hospital_idx, ward_idx, ward_type);
    else
        initialise_nurse( &(hospital->wards[ward_type][ward_idx].nurses[hospital->wards[ward_type][ward_idx].n_worker[NURSE]++]) , pdx, hospital->hospital_idx, ward_idx, ward_type);
}

int healthcare_worker_working( individual* indiv )
{
    if( indiv->status == DEATH || is_in_hospital(indiv) || indiv->quarantined == TRUE )
        return FALSE;

    return TRUE;
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

/*****************************************************************************************
*  Name:		transition_one_hospital_event
*  Description: Generic function for updating an individual with their next
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
        indiv->time_event[from] = model->time;
    if( indiv->current_hospital_event != NULL )
        remove_event_from_event_list( model, indiv->current_hospital_event );
    if( indiv->next_hospital_event != NULL )
        indiv->current_hospital_event = indiv->next_hospital_event;

    if( to != NO_EVENT )
    {
        indiv->time_event[to]     = model->time + ifelse( edge == NO_EDGE, 0, sample_transition_time( model, edge ) );  // TOM: PROBABLY NEEDS SOME PARAMETERISATION HERE?
        indiv->next_hospital_event = add_individual_to_event_list( model, to, indiv, indiv->time_event[to] );
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
    if( indiv->hospital_state == WAITING && indiv->status == CRITICAL)
        remove_patient_from_waiting_list(indiv, &(model->hospitals[indiv->hospital_idx]), COVID_GENERAL);
    
    intervention_on_hospitalised( model, indiv );
    if( indiv->quarantined )
        intervention_quarantine_release( model, indiv );

    set_waiting( indiv, model->params, 1);
}

/*****************************************************************************************
*  Name:		transition_to_general
*  Description: Transitions a severely symptomatic individual from the admissions list to a
*               general ward.
*               This only occurs if there is enough space in the general wards.
*  Returns:		void
******************************************************************************************/
void transition_to_general( model *model, individual *indiv )
{
    if( add_patient_to_hospital( model, indiv) )
        set_general_admission( indiv, model->params, 1);
    else 
        transition_one_hospital_event(model, indiv, indiv->hospital_state, GENERAL, HOSPITAL_TRANSITION); // schedule attempt to add to general next timestep

    //TODO: at some point will need to add transition from icu back to general
}

/*****************************************************************************************
*  Name:		transition_to_ICU
*  Description: Transitions a critically ill individual to the ICU from either a general
*               ward or the admissions list. This only occurs if there is enough space in
*               the ICU.
*  Returns:		void
******************************************************************************************/
void transition_to_icu( model *model, individual *indiv )
{
    int hospital_idx = indiv->hospital_idx;
    hospital* assigned_hospital = &(model->hospitals[hospital_idx]);

    if ( indiv->hospital_state == WAITING )
    {
        if ( assign_to_ward( indiv, assigned_hospital, COVID_ICU ) == TRUE )
        {
            set_icu_admission( indiv, model->params, 1);

            if( assigned_hospital->n_patients_waiting > 0)
                assigned_hospital->n_patients_waiting--;
        }
    } else if ( indiv->hospital_state == GENERAL )
    {
        if ( assign_to_ward( indiv, assigned_hospital, COVID_ICU ) == TRUE )
            remove_patient_from_ward(&(model->hospitals[indiv->hospital_idx].wards[indiv->ward_type][indiv->ward_idx]), indiv->idx);
            set_icu_admission( indiv, model->params, 1);
    }
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
    release_patient_from_hospital( indiv, &(model->hospitals[indiv->hospital_idx]) );
    set_mortuary_admission( indiv, model->params, 1);
    transition_one_hospital_event( model, indiv, MORTUARY, NO_EVENT, NO_EDGE );
}

/*****************************************************************************************
*  Name:		transition_to_populace
*  Description: Transitions a recovered individual from either the general wards or ICU
*               back into the general population. This always succeeds in assigning the patient
*               location.
*  Returns:		void
******************************************************************************************/
void transition_to_discharged( model *model, individual *indiv )
{
    release_patient_from_hospital( indiv, &(model->hospitals[indiv->hospital_idx]) );
    transition_one_hospital_event( model, indiv, DISCHARGED, NO_EVENT, NO_EDGE );
    set_discharged( indiv, model->params, 1);
}

/*****************************************************************************************
*  Name:		release_patient_from_hospital
*  Description:
*  Returns:		int
******************************************************************************************/
int add_patient_to_hospital( model* model, individual *indiv )
{
    int  ward_idx, required_ward, assigned_ward_idx;
    hospital *assigned_hospital;

    required_ward = NO_WARD;
    assigned_hospital = &(model->hospitals[indiv->hospital_idx]); 
    if( assigned_hospital < 0 )
        print_exit("ERROR: calling add_patient_to_hospital who has not previously been assigned to a hospital!!");
    
    required_ward = EVENT_TYPE_TO_WARD_MAP[indiv->status];
    if( required_ward == NOT_IN_HOSPITAL )
        print_exit("ERROR: Adding individual to hospital who is not in HOSPITALISED/HOSPITALISED_RECOVERING/CRIITICAL state!");
        
    assigned_ward_idx = 0;
    for( ward_idx = 1; ward_idx < assigned_hospital->n_wards[required_ward]; ward_idx++ )
        if( assigned_hospital->wards[required_ward][assigned_ward_idx].n_patients > assigned_hospital->wards[required_ward][ward_idx].n_patients)
            assigned_ward_idx = ward_idx;

    if( add_patient_to_ward( &(assigned_hospital->wards[required_ward][assigned_ward_idx]), indiv->idx ) )
    {   
        if( assigned_hospital->waiting_list[required_ward].size > 0 )
            if( indiv->idx != remove_first_patient_from_waiting_list( assigned_hospital, required_ward ) )
                print_exit( "ERROR: adding patient to hospital who is not at the top of the waiting list!!");

        return TRUE;
    }
    return FALSE;
}


/*****************************************************************************************
*  Name:		assign_patient_to_hospital
*  Description: Search a for hospital with bed space, then assign the hospital to that individual based on
*               which ward type they currently need. If there is no space available, assign the patient to the
*               hospital with the shortest waiting list.
*  Returns:		void
******************************************************************************************/
int assign_patient_to_hospital( model* model, individual *indiv )
{
    int hospital_idx, required_ward, added_to_hospital;
    added_to_hospital = FALSE;
    
    if( indiv->status == HOSPITALISED || indiv->status == HOSPITALISED_RECOVERING )
        required_ward = COVID_GENERAL;
    else if( indiv->status == CRITICAL )
        required_ward = COVID_ICU;
    else
        print_exit("ERROR: Adding individual to hospital who is not in HOSPITALISED/HOSPITALISED_RECOVERING/CRIITICAL state!");

    int assigned_hospital_idx = 0;
    int available_beds = hospital_available_beds( &(model->hospitals[assigned_hospital_idx]), required_ward) - model->hospitals[assigned_hospital_idx].waiting_list[required_ward].size;

    for( hospital_idx = 1; hospital_idx < model->params->n_hospitals; hospital_idx++ )
    {
        int next_available_beds = hospital_available_beds( &(model->hospitals[hospital_idx]), required_ward) - model->hospitals[hospital_idx].waiting_list[required_ward].size;
        
        if( next_available_beds > available_beds)
        {
            assigned_hospital_idx = hospital_idx;
            available_beds = next_available_beds;
        }
    }

    indiv->hospital_idx = assigned_hospital_idx;
    if( available_beds < 0 )
    {
        add_patient_to_waiting_list( indiv, &(model->hospitals[assigned_hospital_idx]), required_ward );    
        return FALSE;
    }
    return TRUE;
}

/*****************************************************************************************
*  Name:		assign_to_ward
*  Description:
*  Returns:		int
******************************************************************************************/

int assign_to_ward(individual *indiv, hospital *hospital, int ward_type ) {
    int ward_idx;

    for( ward_idx = 0; ward_idx < hospital->n_wards[ward_type]; ward_idx++ )
    {
        if( add_patient_to_ward( &(hospital->wards[ward_type][ward_idx]), indiv->idx ) == TRUE )
        {
            indiv->ward_type = ward_type;
            indiv->ward_idx  = ward_idx;

            return TRUE;
        }
    }

    return FALSE;
}

/*****************************************************************************************
*  Name:		release_patient_from_hospital
*  Description:
*  Returns:		int
******************************************************************************************/

int hospital_available_beds( hospital* hospital, int ward_type )
{
    int ward_idx;
    int available_beds = 0;

    for( ward_idx = 0; ward_idx < hospital->n_wards[ward_type]; ward_idx++ )
    {
        available_beds += ( ward_available_beds( &(hospital->wards[ward_type][ward_idx]) ));
    }

    return available_beds;
}

/*****************************************************************************************
*  Name:		release_patient_from_hospital
*  Description:
*  Returns:		int
******************************************************************************************/
void release_patient_from_hospital( individual *indiv, hospital *hospital )
{
    if( indiv->hospital_state == WAITING )
        hospital->n_patients_waiting--;
    else
        remove_patient_from_ward( &(hospital->wards[indiv->ward_type][indiv->ward_idx]), indiv->idx );

    indiv->ward_type = NO_WARD;
    indiv->ward_idx  = NO_WARD;
}

/*****************************************************************************************
*  Name:		release_patient_from_hospital
*  Description:
*  Returns:		int
******************************************************************************************/

void add_patient_to_waiting_list( individual *indiv, hospital *hospital, int ward_type)
{
    push( indiv->idx, &(hospital->waiting_list[ward_type]) );
}

/*****************************************************************************************
*  Name:		release_patient_from_hospital
*  Description:
*  Returns:		int
******************************************************************************************/
long remove_first_patient_from_waiting_list( hospital *hospital, int ward_type )
{
    return pop( &(hospital->waiting_list[ward_type]) );
}

/*****************************************************************************************
*  Name:		release_patient_from_hospital
*  Description:
*  Returns:		int
******************************************************************************************/
void remove_patient_from_waiting_list( individual *indiv, hospital *hospital, int ward_type )
{
    remove_patient( indiv->idx, &(hospital->waiting_list[ward_type]));
}
