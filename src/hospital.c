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
    int hdx
)
{
    if( hospital->hospital_idx != 0 )
        print_exit( "a hospital can only be initialised once!");

    hospital->hospital_idx   = hdx;

    hospital->n_total_beds = params->hospital_n_beds;
    hospital->n_total_icus = params->hospital_n_icus;

    //TODO: Change this to store the actual list of workers.
    hospital->doctor_pdxs  = calloc(params->n_total_doctors, sizeof(long) );
    hospital->nurse_pdxs   = calloc(params->n_total_nurses, sizeof(long) );

    hospital->general_patient_pdxs = calloc( hospital->n_total_beds, sizeof(long) ); //TODO: should memory allocated be size of beds + icus??
    hospital->icu_patient_pdxs     = calloc( hospital->n_total_icus, sizeof(long) );

    hospital->n_total_doctors = params->n_total_doctors;
    hospital->n_total_nurses  = params->n_total_nurses;
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
    free( hospital->doctor_pdxs );
    free( hospital->nurse_pdxs );
};


