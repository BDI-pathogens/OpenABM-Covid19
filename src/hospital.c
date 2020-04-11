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

    for( ward_type = 0; ward_type < N_HOSPITAL_WARD_TYPES; ward_type++ )
    {
        hospital->n_wards[ward_type] = params->n_wards[ward_type];

        for( hcw_type = 0; hcw_type < N_WORKER_TYPES; hcw_type++ )
            hospital->n_workers[hcw_type] += params->n_wards[ward_type] * params->n_hcw_per_ward[ward_type][hcw_type];

        hospital->wards[ward_type] = calloc( params->n_wards[ward_type], sizeof(ward) );
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
//    indiv->status           = from;
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
    assign_patient_to_hospital( model, indiv );

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
    int hospital_idx = indiv->hospital_idx;
    hospital* assigned_hospital = &(model->hospitals[hospital_idx]);

    if ( indiv->hospital_state == WAITING )
    {
        if ( assign_to_ward( indiv, assigned_hospital, COVID_GENERAL ) == TRUE )
        {
            set_general_admission( indiv, model->params, 1);

            if (assigned_hospital->n_patients_waiting > 0 )
                assigned_hospital->n_patients_waiting--;
        }
    }
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
    if (indiv->idx == 87333)
        printf("break");

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
    //set_discharged( indiv, model->params, 1);
    set_discharged( indiv, model->params, 1);
}

/*****************************************************************************************
*  Name:		assign_patient_to_hospital
*  Description: Search a for hospital with bed space, then assign the hospital to that individual based on
*               which ward type they currently need. If there is no space available, assign the patient to the
*               hospital with the shortest waiting list.
*  Returns:		void
******************************************************************************************/
void assign_patient_to_hospital( model* model, individual *indiv )
{
    int hospital_idx, ward_idx;
    int required_ward;

    int added_to_hospital = FALSE;

    if( indiv->status == HOSPITALISED )
        required_ward = COVID_GENERAL;

    if( indiv->status == CRITICAL )
        required_ward = COVID_ICU;

    hospital* assigned_hospital;
    for( hospital_idx = 0; hospital_idx < model->params->n_hospitals; hospital_idx++ )
    {
        assigned_hospital = &(model->hospitals[hospital_idx]);
        for( ward_idx = 0; ward_idx < assigned_hospital->n_wards[required_ward]; ward_idx++ )
        {
            //Check if number of patients in a ward is less than the max no. of beds.
            if( assigned_hospital->wards[required_ward][ward_idx].n_patients
                    <  assigned_hospital->wards[required_ward][ward_idx].n_beds )
            {
                 indiv->hospital_idx = hospital_idx;
                 assigned_hospital->n_patients_waiting++;
                 added_to_hospital = TRUE;
            }
        }
    }

    if( added_to_hospital == FALSE )
    {
        hospital* assigned_hospital = &(model->hospitals[0]);

        for( hospital_idx = 1; hospital_idx < model->params->n_hospitals; hospital_idx++ )
        {
            if( model->hospitals[hospital_idx].n_patients_waiting < assigned_hospital->n_patients_waiting)
                assigned_hospital = &(model->hospitals[hospital_idx]);
        }
        assigned_hospital->n_patients_waiting++;
        indiv->hospital_idx = assigned_hospital->hospital_idx;
    }
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
void release_patient_from_hospital( individual *indiv, hospital *hospital )
{
    if( indiv->hospital_state == WAITING )
        hospital->n_patients_waiting--;
    else
        remove_patient_from_ward( &(hospital->wards[indiv->ward_type][indiv->ward_idx]), indiv->idx );

    indiv->ward_type = NO_WARD;
    indiv->ward_idx  = NO_WARD;
}
