/*
 * individual.c
 *
 *  Created on: 30 Mar 2020
 *      Author: vuurenk
 */

#include "hospital.h"
#include "constant.h"
#include "utilities.h"
#include "network.h"
#include "model.h"

/*****************************************************************************************
*  Name:		initialize_hospital
*  Description: initializes and individual at the start of the simulation, note can
*  				only be called once per individual
*  Returns:		void
******************************************************************************************/
void initialise_hospital(
    hospital *hospital,
    parameters *params,
//    network *hospital_network,
    int hdx
)
{
    if( hospital->hospital_idx != 0 )
        print_exit( "a hospital can only be initialised once!");

    hospital->hospital_idx   = hdx;
//    hospital->hospital_network = hospital_network;

    hospital->available_beds = params->hospital_n_beds;
    hospital->available_icus = params->hospital_n_icus;

    //TODO: Change this to store the actual list of workers.
    hospital->doctor_pdxs  = calloc(params->n_total_doctors, sizeof(long) );
    hospital->nurse_pdxs   = calloc(params->n_total_nurses, sizeof(long) );
    hospital->patient_pdxs = calloc(params->n_total, sizeof(long) ); //TODO: should memory allocated be size of beds + icus??

    hospital->n_total_doctors = 0;
    hospital->n_total_nurses  = 0;
}

/*****************************************************************************************
*  Name:		add_nurse
*  Description: adds population id of a doctor / nurse to hospital's
*               doctor / nurse population id list
******************************************************************************************/
void add_healthcare_worker_to_hospital(hospital *hospital, long pdx, int type)
{
    if( type == DOCTOR )
        hospital->doctor_pdxs[hospital->n_total_doctors++] = pdx;
    else if( type == NURSE )
        hospital->nurse_pdxs[hospital->n_total_nurses++] = pdx;
}

/*****************************************************************************************
*  Name:		add_nurse
*  Description: adds population id of a doctor / nurse to hospital's
*               doctor / nurse population id list
******************************************************************************************/
void add_patient_to_hospital(hospital *hospital, long pdx, int type)
{
    if( type == HOSPITALISED )
    {
        hospital->patient_pdxs[hospital->n_total_patients++] = pdx;
        hospital->available_beds--;
    }
    else if( type == HOSPITALISED_CRITICAL)
    {
        hospital->patient_pdxs[hospital->n_total_patients++] = pdx;
        hospital->available_icus--;
    }
}


int healthcare_worker_working(individual* indiv)
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
    free( hospital->doctor_pdxs );
    free( hospital->nurse_pdxs );
};

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
    indiv->status           = from;

    if( from != NO_EVENT )
        indiv->time_event[from] = model->time;
    if( indiv->current_disease_event != NULL )
        remove_event_from_event_list( model, indiv->current_disease_event );
    if( indiv->next_disease_event != NULL )
        indiv->current_disease_event = indiv->next_disease_event;

    if( to != NO_EVENT )
    {
        indiv->time_event[to]     = model->time + ifelse( edge == NO_EDGE, 0,
                sample_transition_time( model, edge ) );
        indiv->next_disease_event = add_individual_to_event_list( model, to, indiv, indiv->time_event[to] );
    }
}

/*****************************************************************************************
*  Name:		transition_to_waiting
*  Description: Transitions a severely symptomatic individual from the general populace
*               to the admissions list for the hospital.
*               At the moment - severely symptomatic refers to HOSPITALISED individuals.
*  Returns:		void
******************************************************************************************/
void transition_to_waiting( model *model, individual *indiv ) {

}
/*****************************************************************************************
*  Name:		transition_to_general
*  Description: Transitions a severely symptomatic individual from the admissions list to a
*               general ward TODO: or a recovering patient from the ICU to a general ward.
*               This only occurs if there is enough space in the general wards.
*  Returns:		void
******************************************************************************************/
void transition_to_general( model *model, individual *indiv ) {

}
/*****************************************************************************************
*  Name:		transition_to_ICU
*  Description: Transitions a critically ill individual to the ICU from either a general
*               ward or the admissions list. This only occurs if there is enough space in
*               the ICU.
*  Returns:		void
******************************************************************************************/
void transition_to_icu( model *model, individual *indiv ) {

}
/*****************************************************************************************
*  Name:		transition_to_mortuary
*  Description: Transitions a newly deceased person from either the general ward or ICU
*               into a "morutary" to free space for new patients.
*  Returns:		void
******************************************************************************************/
void transition_to_mortuary( model *model, individual *indiv ) {

}
/*****************************************************************************************
*  Name:		transition_to_populace
*  Description: Transitions a recovered individual from either the general wards or ICU
*               back into the general population.
*  Returns:		void
******************************************************************************************/
void transition_to_populace( model *model, individual *indiv ) {

}

