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
*  Name:		initialise_ward
*  Description: initialises a ward at the start of the simulation, assigns .
*  Returns:		void
******************************************************************************************/
void initialise_ward(
    ward *ward,
    int ward_idx,
    int type,
    int n_beds,
    int n_max_doctors,
    int n_max_nurses
)
{
    ward->ward_idx          = ward_idx;
    ward->type              = type;
    ward->n_beds            = n_beds;
    ward->n_max_hcw[DOCTOR] = n_max_doctors;
    ward->n_max_hcw[NURSE]  = n_max_nurses;

    ward->n_worker[NURSE]   = 0;
    ward->n_worker[DOCTOR]  = 0;
    ward->n_patients        = 0;

    ward->doctors = calloc( n_max_doctors, sizeof(doctor) );
    ward->nurses  = calloc( n_max_nurses, sizeof(nurse) );

    ward->patient_pdxs = calloc( n_beds, sizeof(long) );
    for (int i = 0; i < n_beds; i++)
        ward->patient_pdxs[i] = NO_PATIENT;
}

void set_up_ward_networks( ward* ward, int max_hcw_daily_interactions )
{
    int interaction_type;

    //TODO: there must be a better way of getting these interactiont type enums... should there be some kind of enum map?
    interaction_type = ( ward->type == COVID_GENERAL ) ? HOSPITAL_DOCTOR_PATIENT_GENERAL : HOSPITAL_DOCTOR_PATIENT_ICU;
    ward->doctor_patient_network = new_network( ward->n_worker[DOCTOR], interaction_type );
    ward->doctor_patient_network->edges = calloc( max_hcw_daily_interactions * ward->n_worker[DOCTOR], sizeof(edge) );
    interaction_type = ( ward->type == COVID_GENERAL ) ? HOSPITAL_DOCTOR_PATIENT_GENERAL : HOSPITAL_DOCTOR_PATIENT_ICU;
    ward->nurse_patient_network = new_network( ward->n_worker[NURSE], interaction_type );
    ward->nurse_patient_network->edges = calloc( max_hcw_daily_interactions * ward->n_worker[NURSE], sizeof(edge) );
}

void build_ward_networks( model *model, ward* ward )
{
    ward->doctor_patient_network->n_edges = 0;
    ward->nurse_patient_network->n_edges  = 0;

    if (ward->n_patients > 0 )
    {
        int idx, n_hcw_working;
        long *hc_workers;
        long *hc_workers_nurse;

        hc_workers = calloc(ward->n_worker[DOCTOR], sizeof (long) );
        n_hcw_working = 0;

        for( idx = 0; idx < ward->n_worker[DOCTOR]; idx++ )
            if( healthcare_worker_working( &(model->population[ ward->doctors[idx].pdx ]) ))
                hc_workers[n_hcw_working++] = ward->doctors[idx].pdx;

        //rebuild doctor -> patient network
        build_hcw_patient_network( ward, ward->doctor_patient_network,  hc_workers, n_hcw_working, model->params->n_patient_required_interactions[ward->type][DOCTOR], model->params->max_hcw_daily_interactions );

        //free( hc_workers );
        hc_workers_nurse = calloc( ward->n_worker[NURSE], sizeof (long));

        //get list of ward's working nurse's pdxs
        n_hcw_working = 0;
        for( idx = 0; idx < ward->n_worker[NURSE]; idx++ )
            if( healthcare_worker_working( &(model->population[ ward->nurses[idx].pdx ]) ))
                hc_workers_nurse[n_hcw_working++] = ward->nurses[idx].pdx;

        //rebuild nurse -> patient network
        build_hcw_patient_network( ward, ward->nurse_patient_network,  hc_workers_nurse, n_hcw_working, model->params->n_patient_required_interactions[ward->type][NURSE], model->params->max_hcw_daily_interactions );

        free( hc_workers );
        free( hc_workers_nurse );
    }
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

    network->n_edges = 0;
    network->n_vertices       = n_hcw_working + ward->n_patients;

    patient = 0;
    n_pos = 0;

    int npatients = 0;
    for(int idx = 0; idx < ward->n_beds; idx++ )
    {
        if( ward->patient_pdxs[idx] != NO_PATIENT )
        {
            for( int i = 0; i < n_patient_required_interactions; i++)
                all_required_interactions[n_pos++] = ward->patient_pdxs[idx];
            npatients++;
        }
    }

    if( npatients != ward->n_patients )
        print_exit("number of patients in patient pdx list is not equal to ward's n_patients!!");

    //shuffle list of all interactions
    gsl_ran_shuffle( rng, all_required_interactions, n_pos, sizeof(long) );

    idx = 0;
    hdx = 0;
    n_total_interactions--;
    //assign network edges between hcw and randomly picked patient interactions
    //TODO: shift patterns eg day off
    while( idx < n_total_interactions )
    {
        network->edges[network->n_edges].id1 = hc_workers[ hdx ];
        network->edges[network->n_edges].id2 = all_required_interactions[ idx++ ];

        if( network->edges[network->n_edges].id2 == 0 )
        {
            printf("edge2 pdx ewual to null!!");
        }

        if(  network->edges[network->n_edges].id1 == 0 )
        {
            printf("edge1 pdx ewual to null!!");
        }

        network->n_edges++;
        hdx++;
        if( hdx >= n_hcw_working )
            hdx = 0;
    }

    free( all_required_interactions );
    free( capped_hcw_interactions );
}


int patient_pdx_at( ward* ward, int idx )
{
    return ward->patient_pdxs[idx];
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
                //ward->n_patients++;
                increment_patients(ward);
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
            //ward->n_patients--;
            decrement_patients(ward);
            //break;
        }
    }
}

int ward_available_beds( ward* ward)
{
    return ( ward->n_beds - ward->n_patients );
}

void destroy_ward( ward* ward )
{
    free( ward->doctor_patient_network );
    free( ward->nurse_patient_network );
    free( ward->patient_pdxs );
    free( ward->doctors );
    free( ward->nurses );
}

void increment_patients( ward* ward )
{
    ward->n_patients++;
    //printf("INCREMENTED: ward type%i idx%i beds: %i\n", ward->type, ward->ward_idx, ward_available_beds( ward ));
}

void decrement_patients( ward* ward )
{
    ward->n_patients--;
    //printf("DECREMENTED: ward type%i idx%i beds: %i\n", ward->type, ward->ward_idx, ward_available_beds( ward ));
}
    
