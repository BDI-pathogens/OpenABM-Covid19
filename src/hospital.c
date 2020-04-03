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
    if( hospital->hospital_idx != 0 )
        print_exit( "a hospital can only be initialised once!");

    hospital->hospital_idx   = hdx;

    hospital->n_total_beds = params->hospital_n_beds;
    hospital->n_total_icus = params->hospital_n_icus;

    hospital->doctors = calloc( params->n_total_doctors, sizeof(doctor) );
    hospital->nurses  = calloc( params->n_total_doctors, sizeof(doctor) );

    hospital->general_patient_pdxs = calloc( hospital->n_total_beds, sizeof(long) ); //TODO: should memory allocated be size of beds + icus??
    hospital->icu_patient_pdxs     = calloc( hospital->n_total_icus, sizeof(long) );

    //setup wards
    hospital->n_wards = calloc( N_HOSPITAL_WARD_TYPES, sizeof(int*) );
    hospital->wards = calloc( N_HOSPITAL_WARD_TYPES, sizeof(ward*) );

    for( int w_type = 0; w_type < N_HOSPITAL_WARD_TYPES; w_type++ )
    {
        hospital->wards[w_type] = calloc( hospital->n_wards[w_type], sizeof(ward) );
        for( int n_ward = 0; n_ward < hospital->n_wards[w_type]; n_ward++ )
            initialise_ward( &(hospital->wards[w_type][n_ward]), w_type, n_ward);
    }
}

/*****************************************************************************************
*  Name:		set_up_hospital_networks
*  Description: calls setup functions for all networks related to the hospital instance
*  Returns:		void
******************************************************************************************/
void set_up_hospital_networks( hospital* hospital )
{
    int idx, n_healthcare_workers;
    int ward_idx, ward_type;
    long *healthcare_workers;

    //setup hospital workplace network
    n_healthcare_workers = 0;
    healthcare_workers = calloc( hospital->n_total_doctors + hospital->n_total_nurses, sizeof(long) );

    for ( idx = 0; idx < hospital->n_total_doctors; idx++ )
        healthcare_workers[n_healthcare_workers++] = hospital->doctors[idx].pdx;

    for ( idx = 0; idx < hospital->n_total_nurses; idx++ )
        healthcare_workers[n_healthcare_workers++] = hospital->nurses[idx].pdx;

    hospital->hospital_workplace_network = calloc( 1, sizeof( network ));
    hospital->hospital_workplace_network = new_network( n_healthcare_workers, HOSPITAL_WORK );
    int n_interactions = 20;//TODO: maybe make this number of interactions set via param... and should nurses have more??

    build_watts_strogatz_network( hospital->hospital_workplace_network, n_healthcare_workers, n_interactions, 0.1, TRUE ); //TODO: p_rewire probability higher??
    relabel_network( hospital->hospital_workplace_network, healthcare_workers );

    //setup hcw -> patient networks for all wards
    hospital->wards_networks = calloc( N_HOSPITAL_WARD_TYPES, sizeof(network*) );

    for ( ward_type = 0; ward_type < N_HOSPITAL_WARD_TYPES; ward_type++ )
        hospital->wards_networks[N_HOSPITAL_WARD_TYPES] = calloc( hospital->n_wards[ward_type], sizeof(network*) );
        for( ward_idx = 0; ward_idx < hospital->n_wards[N_HOSPITAL_WARD_TYPES]; ward_idx++ )
            set_up_ward_networks( &(hospital->wards[ward_type][ward_idx]) );

    free( healthcare_workers );
}

/*****************************************************************************************
*  Name:		add_nurse
*  Description: adds population id of a doctor / nurse to hospital's
*               doctor / nurse population id list
******************************************************************************************/
void add_healthcare_worker_to_hospital(hospital *hospital, long pdx, int type)
{
    int ward_type, ward_idx;
    int ward_found = FALSE;

    if( type == DOCTOR )
    {
        for( ward_type = 0; ward_type < N_HOSPITAL_WARD_TYPES && ward_found != TRUE; ward_type++ )
            for( ward_idx = 0; ward_idx < hospital->n_wards[ward_type] && ward_found != TRUE; ward_idx++ )
                if( hospital->wards[ward_type][ward_idx].n_doctors < hospital->wards[ward_type][ward_idx].max_doctors )
                    ward_found = TRUE;

        if( ward_found == FALSE)
            print_exit( "attempted to allocated more than max number of doctors to hospital!!" );

        initialise_doctor( &(hospital->doctors[hospital->n_total_doctors++]) , pdx, hospital->hospital_idx, ward_idx, ward_type);
    }
    else if( type == NURSE )
    {
        for( ward_type = 0; ward_type < N_HOSPITAL_WARD_TYPES && ward_found != TRUE; ward_type++ )
            for( ward_idx = 0; ward_idx < hospital->n_wards[ward_type] && ward_found != TRUE; ward_idx++ )
                if( hospital->wards[ward_type][ward_idx].n_nurses < hospital->wards[ward_type][ward_idx].max_nurses )
                    ward_found = TRUE;

        if( ward_found == FALSE)
            print_exit( "attempted to allocated more than max number of nurses to hospital!!" );

        initialise_nurse( &(hospital->nurses[hospital->n_total_nurses++]) , pdx, hospital->hospital_idx, ward_idx, ward_type);
    }
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
        hospital->general_patient_pdxs[hospital->n_total_general_patients++] = pdx;
        hospital->n_total_general_patients++;
    }
    else if( type == HOSPITALISED_CRITICAL)
    {
        hospital->icu_patient_pdxs[hospital->n_total_icu_patients++] = pdx;
        hospital->n_total_icu_patients++;
    }
}


int healthcare_worker_working( individual* indiv )
{
    if( indiv->status == DEATH || is_in_hospital(indiv) || indiv->quarantined == TRUE )
        return FALSE;

    return TRUE;
}

void build_hcw_patient_network( model *model, network *network, long *patient_pdxs, long *worker_pdxs, int n_patients, int total_workers, int patient_required_interactions)
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

/*****************************************************************************************
*  Name:		destroy_hospital
*  Description: Destroys the model structure and releases its memory
******************************************************************************************/
void destroy_hospital( hospital *hospital)
{
    free( hospital->nurses );
    free( hospital->doctors );
    free( hospital->hospital_workplace_network );

};


