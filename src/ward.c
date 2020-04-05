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
    int type
)
{
    ward->ward_idx = ward_idx;
    ward->type = type;

    ward->doctors = calloc( ward->n_max_doctors, sizeof(doctor) );
    ward->nurses  = calloc( ward->n_max_nurses, sizeof(nurse) );
}

void set_up_ward_networks( ward* ward )
{
    int interaction_type = ( ward->type == COVID_GENERAL ) ? HOSPITAL_DOCTOR_PATIENT_GENERAL : HOSPITAL_DOCTOR_PATIENT_ICU;
    ward->doctor_patient_network = new_network( ward->n_doctors, interaction_type );
    ward->doctor_patient_network->edges = NULL;
    //TODO: there must be a better way of getting these interactiont type enums
    interaction_type = ( ward->type == COVID_GENERAL ) ? HOSPITAL_DOCTOR_PATIENT_GENERAL : HOSPITAL_DOCTOR_PATIENT_ICU;
    ward->nurse_patient_network = new_network( ward->n_nurses, interaction_type );
    ward->nurse_patient_network->edges = NULL;
}

void build_ward_networks( model *model, ward* ward )
{
    int idx, n_hcw_working, required_interaction_doctors, required_interaction_nurse;
    long *hc_workers;
    hc_workers = calloc( ward->n_doctors + ward->n_nurses, sizeof(long) );
    n_hcw_working = 0;

    //get list of ward's working hcw pdxs
    for( idx = 0; idx < ward->n_doctors; idx++ )
        if( healthcare_worker_working( &(model->population[ ward->doctors[idx].pdx ]) ))
            hc_workers[n_hcw_working++] = ward->doctors[idx].pdx;

    for( idx = 0; idx < ward->n_nurses; idx++ )
        if( healthcare_worker_working( &(model->population[ ward->nurses[idx].pdx ]) ))
            hc_workers[n_hcw_working++] = ward->nurses[idx].pdx;

    if( ward->type == COVID_ICU ) //TODO: make interactions required variables a list accessible with ward types
    {
        required_interaction_doctors = model->params->icu_patient_doctor_required_interactions;
        required_interaction_nurse = model->params->icu_patient_nurse_required_interactions;
    }

    build_hcw_patient_network( model, ward->doctor_patient_network, ward->patient_pdxs, hc_workers, ward)
}

void build_hcw_patient_network( model *model, ward* ward, network *network, long *hc_workers, int n_hcw_working, int patient_required_interactions )
{
    int idx, hdx, patient_interactions_per_hcw, patient_interaction_per_hcw, n_total_interactions, patient, n_pos;
    long *all_required_interactions, *capped_hcw_interactions;

    patient_interactions_per_hcw = round( (patient_required_interactions * ward->n_patients) / n_hcw_working );
    //TODO: should there be different max interactions for doctors / nurses?
    patient_interaction_per_hcw = (patient_interactions_per_hcw > model->params->max_hcw_daily_interactions) ? model->params->max_hcw_daily_interactions : patient_interaction_per_hcw;
    n_total_interactions = patient_interaction_per_hcw * n_hcw_working;

    all_required_interactions = calloc( patient_required_interactions * ward->n_patients, sizeof(long) );
    capped_hcw_interactions   = calloc( n_total_interactions, sizeof(long) );
    network->edges            = calloc( n_total_interactions, sizeof(edge) );
    network->n_vertices       = n_hcw_working + ward->n_patients;

    n_pos = 0;
    for( patient = 0; patient < ward->n_patients; patient++ )
        for (idx = 0; idx < patient_required_interactions; idx++)
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

void build_hcw_patient_network( model *model, network *network, long *patient_pdxs, doctor *doctors, nurse *nurses, int n_patients, int total_workers, int patient_required_interactions)
{
    long n_pos;
    int idx, hdx, patient, n_hcw_working, patient_interaction_per_hcw, n_total_interactions;
    long *hc_workers, *all_required_interactions, *capped_hcw_interactions;

    free( network->edges );
    network->n_edges = 0;
    n_pos            = 0;

    //get array of hcw who are actually working (ie not quarantined / hospitalised / critical / dead)
    hc_workers = calloc( total_workers, sizeof(long) );
    n_hcw_working = 0;
    for( idx = 0; idx < total_workers; idx++ )
        if( healthcare_worker_working( &(model->population[ worker_pdxs[idx] ]) ))
            hc_workers[n_hcw_working++] = worker_pdxs[idx];

    patient_interaction_per_hcw = round( (patient_required_interactions * n_patients) / n_hcw_working );

    //TODO: should there be different max interactions for doctors / nurses?
    patient_interaction_per_hcw = (patient_interaction_per_hcw > model->params->max_hcw_daily_interactions) ? model->params->max_hcw_daily_interactions : patient_interaction_per_hcw;

    n_total_interactions = patient_interaction_per_hcw * n_hcw_working;

    all_required_interactions = calloc( patient_required_interactions * n_patients, sizeof(long) );
    capped_hcw_interactions   = calloc( n_total_interactions, sizeof(long) );
    network->edges            = calloc( n_total_interactions, sizeof(edge) );
    network->n_vertices       = n_hcw_working + n_patients;

    //get array of all required interactions
    n_pos = 0;
    for( patient = 0; patient < n_patients; patient++ )
        for (idx = 0; idx < patient_required_interactions; idx++)
            all_required_interactions[n_pos++] = patient_pdxs[patient];

    //shuffle list of all interactions
    gsl_ran_shuffle( rng, all_required_interactions, n_pos, sizeof(long) );

    //pick the capped (max) amount of interactions randomly from shuffled list;
    gsl_ran_choose( rng, capped_hcw_interactions, n_total_interactions, all_required_interactions, n_pos, sizeof(long) );

    //TODO: ask rob - should we also randomly shuffle the list of healthcare workers before assigning interactions
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

void destroy_ward( ward* ward )
{
    free( ward->doctor_patient_network );
    free( ward->nurse_patient_network );
}

