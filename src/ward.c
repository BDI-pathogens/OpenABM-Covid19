/*
 * doctor.c
 *
 *  Created on: 03 Apr 2020
 *      Author: vuurenk
 */

#include "ward.h"
#include "constant.h"
#include "model.h"

/*****************************************************************************************
*  Name:		initialize_hospital
*  Description: initializes and individual at the start of the simulation, note can
*  				only be called once per individual
*  Returns:		void
******************************************************************************************/
void initialise_ward(
    ward *ward,
    int ward_idx,
    int type,
    int n_beds
)
{
    ward->ward_idx = ward_idx;
    ward->type = type;

    ward->patient_pdxs = calloc( ward->n_beds, sizeof(long) );
    for (int i = 0; i < ward->n_beds; i++)
        ward->patient_pdxs[i] = NO_PATIENT;

    ward->doctors = calloc( ward->n_max_doctors, sizeof(doctor) );
    ward->nurses  = calloc( ward->n_max_nurses, sizeof(nurse) );

    ward->n_beds = n_beds;
}

void set_up_ward_networks( ward* ward )
{
    int interaction_type;

    //TODO: there must be a better way of getting these interactiont type enums... should there be some kind of enum map?
    interaction_type = ( ward->type == COVID_GENERAL ) ? HOSPITAL_DOCTOR_PATIENT_GENERAL : HOSPITAL_DOCTOR_PATIENT_ICU;
    ward->doctor_patient_network = new_network( ward->n_doctors, interaction_type );
    ward->doctor_patient_network->edges = NULL;
    interaction_type = ( ward->type == COVID_GENERAL ) ? HOSPITAL_DOCTOR_PATIENT_GENERAL : HOSPITAL_DOCTOR_PATIENT_ICU;
    ward->nurse_patient_network = new_network( ward->n_nurses, interaction_type );
    ward->nurse_patient_network->edges = NULL;
}

void build_ward_networks( model *model, ward* ward )
{
    int idx, n_hcw_working;
    long *hc_workers;
    hc_workers = calloc( ward->n_doctors + ward->n_nurses, sizeof(long) );
    n_hcw_working = 0;

    //get list of ward's working doctor pdxs
    for( idx = 0; idx < ward->n_doctors; idx++ )
        if( healthcare_worker_working( &(model->population[ ward->doctors[idx].pdx ]) ))
            hc_workers[n_hcw_working++] = ward->doctors[idx].pdx;

    //rebuild doctor -> patient network
    build_hcw_patient_network( ward, ward->doctor_patient_network,  hc_workers, n_hcw_working, model->params->n_patient_required_interactions[ward->type][DOCTOR], model->params->max_hcw_daily_interactions );

    //get list of ward's working nurse's pdxs
    n_hcw_working = 0;
    for( idx = 0; idx < ward->n_nurses; idx++ )
        if( healthcare_worker_working( &(model->population[ ward->nurses[idx].pdx ]) ))
            hc_workers[n_hcw_working++] = ward->nurses[idx].pdx;

    //rebuild nurse -> patient network
    build_hcw_patient_network( ward, ward->nurse_patient_network,  hc_workers, n_hcw_working, model->params->n_patient_required_interactions[ward->type][NURSE], model->params->max_hcw_daily_interactions );
}

void build_hcw_patient_network( ward* ward, network *network, long *hc_workers, int n_hcw_working, int n_patient_required_interactions, int max_hcw_daily_interactions )
{
    int idx, hdx, patient_interactions_per_hcw, n_total_interactions, patient, n_pos;
    long *all_required_interactions, *capped_hcw_interactions;

    patient_interactions_per_hcw = round( (n_patient_required_interactions * ward->n_patients) / n_hcw_working );
    //TODO: should there be different max interactions for doctors / nurses?
    patient_interactions_per_hcw = (patient_interactions_per_hcw > max_hcw_daily_interactions) ? max_hcw_daily_interactions : patient_interactions_per_hcw;
    n_total_interactions = patient_interactions_per_hcw * n_hcw_working;

    all_required_interactions = calloc( n_patient_required_interactions * ward->n_patients, sizeof(long) );
    capped_hcw_interactions   = calloc( n_total_interactions, sizeof(long) );
    free( network->edges );
    network->edges            = calloc( n_total_interactions, sizeof(edge) );
    network->n_vertices       = n_hcw_working + ward->n_patients;

    n_pos = 0;
    for( patient = 0; patient < ward->n_patients; patient++ )
        for (idx = 0; idx < n_patient_required_interactions; idx++)
            all_required_interactions[n_pos++] = ward->patient_pdxs[patient];

    //shuffle list of all interactions
    gsl_ran_shuffle( rng, all_required_interactions, n_pos, sizeof(long) );

    //pick the capped (max) amount of interactions randomly from shuffled list;
    gsl_ran_choose( rng, capped_hcw_interactions, n_total_interactions, all_required_interactions, n_pos, sizeof(long) );

    idx = 0;
    hdx = 0;
    n_total_interactions--;
    //assign network edges between hcw and randomly picked patient interactions
    //TODO: shift patterns eg day off
    while( idx < n_total_interactions )
    {
        network->edges[network->n_edges].id1 = hc_workers[ hdx++ ];
        network->edges[network->n_edges].id2 = capped_hcw_interactions[ idx++ ];
        network->n_edges++;
        hdx = ( hdx++ < n_hcw_working ) ? hdx : 0;
    }

    free( all_required_interactions );
    free( capped_hcw_interactions );
    free( hc_workers );
}

int add_patient_to_ward( ward *ward, long pdx )
{
    if( ward->n_patients < ward->n_beds )
    {
        for( int idx = 0; idx < ward->n_beds; idx++ )
        {
            if( ward->patient_pdxs[idx] == NO_PATIENT )
            {
                ward->patient_pdxs[idx] = pdx;
                ward->n_patients++;
                return TRUE;
            }
        }
    }
    return FALSE;
}

void remove_patient_from_ward( ward* ward, long pdx)
{
    int idx;

    for( idx = 0; idx < ward->n_beds; idx++ )
    {
        if( ward->patient_pdxs[idx] == pdx )
        {
            ward->patient_pdxs[idx] = NO_PATIENT;
            ward->n_patients--;
            break;
        }
    }
}

void destroy_ward( ward* ward )
{
    free( ward->doctor_patient_network );
    free( ward->nurse_patient_network );
    free( ward->doctors );
    free( ward->nurses );
}
