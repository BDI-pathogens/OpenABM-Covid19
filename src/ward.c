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

    ward->doctors = calloc( n_max_doctors, sizeof(doctor) );
    ward->nurses  = calloc( n_max_nurses, sizeof(nurse) );

    ward->patients = calloc( 1, sizeof (list) );
    initialise_list( ward->patients );
}

/*****************************************************************************************
*  Name:		set_up_ward_networks
*  Returns:		void
******************************************************************************************/
void set_up_ward_networks( model* model, ward* ward, int max_hcw_daily_interactions )
{
    int interaction_type;

    interaction_type = ( ward->type == COVID_GENERAL ) ? HOSPITAL_DOCTOR_PATIENT_GENERAL : HOSPITAL_DOCTOR_PATIENT_ICU;
    ward->doctor_patient_network = add_new_network( model, ward->n_worker[DOCTOR], interaction_type );
    ward->doctor_patient_network->edges = calloc( max_hcw_daily_interactions * ward->n_worker[DOCTOR], sizeof(edge) );
    ward->doctor_patient_network->skip_hospitalised = FALSE;
    ward->doctor_patient_network->skip_quarantined  = TRUE;
    ward->doctor_patient_network->construction      = NETWORK_CONSTRUCTION_WATTS_STROGATZ;
    ward->doctor_patient_network->daily_fraction    = 1.0;

    interaction_type = ( ward->type == COVID_GENERAL ) ? HOSPITAL_NURSE_PATIENT_GENERAL : HOSPITAL_NURSE_PATIENT_ICU;
    ward->nurse_patient_network = add_new_network( model, ward->n_worker[NURSE], interaction_type );
    ward->nurse_patient_network->edges = calloc( max_hcw_daily_interactions * ward->n_worker[NURSE], sizeof(edge) );
    ward->nurse_patient_network->skip_hospitalised = FALSE;
    ward->nurse_patient_network->skip_quarantined  = TRUE;
    ward->nurse_patient_network->construction	   = NETWORK_CONSTRUCTION_WATTS_STROGATZ;
    ward->nurse_patient_network->daily_fraction    = 1.0;

}

/*****************************************************************************************
*  Name:		build_ward_networks
*  Returns:		void
******************************************************************************************/
void build_ward_networks( model *model, ward* ward )
{
    ward->doctor_patient_network->n_edges = 0;
    ward->nurse_patient_network->n_edges  = 0;

    if (ward->patients->size > 0 )
    {
        int idx, n_hcw_working;
        long *hc_workers_doctors;
        long *hc_workers_nurse;

        hc_workers_doctors = calloc(ward->n_worker[DOCTOR], sizeof (long) );
        n_hcw_working = 0;

        //get list of ward's working doctors
        for( idx = 0; idx < ward->n_worker[DOCTOR]; idx++ )
            if( healthcare_worker_working( &(model->population[ ward->doctors[idx].pdx ]) ))
                hc_workers_doctors[n_hcw_working++] = ward->doctors[idx].pdx;
            
        //if there are doctors working, rebuild doctor -> patient network
        if( n_hcw_working > 0 )
            build_hcw_patient_network( ward, ward->doctor_patient_network,  hc_workers_doctors, n_hcw_working, model->params->n_patient_required_interactions[ward->type][DOCTOR], model->params->max_hcw_daily_interactions );
        
        hc_workers_nurse = calloc( ward->n_worker[NURSE], sizeof (long));
        n_hcw_working = 0;

        //get list of ward's working nurses
        for( idx = 0; idx < ward->n_worker[NURSE]; idx++ )
            if( healthcare_worker_working( &(model->population[ ward->nurses[idx].pdx ]) ))
                hc_workers_nurse[n_hcw_working++] = ward->nurses[idx].pdx;
                

        //if there are nurses working, rebuild nurse -> patient network
        if( n_hcw_working > 0 )
            build_hcw_patient_network( ward, ward->nurse_patient_network,  hc_workers_nurse, n_hcw_working, model->params->n_patient_required_interactions[ward->type][NURSE], model->params->max_hcw_daily_interactions );
        
        free( hc_workers_doctors );
        free( hc_workers_nurse );
    }
}

/*****************************************************************************************
*  Name:		build_ward_networks
*  Returns:		void
******************************************************************************************/
void build_hcw_patient_network( ward* ward, network *network, long *hc_workers, int n_hcw_working, int n_patient_required_interactions, int max_hcw_daily_interactions )
{
    if( n_patient_required_interactions > 0 )
    {
        int idx, hdx, n_total_interactions, n_pos;
        float patient_interactions_per_hcw;
        long *all_required_interactions, *capped_hcw_interactions;

        //Determine the even distribution of patients' required interactions among working healthcare workers in ward
        patient_interactions_per_hcw = (n_patient_required_interactions * ward->patients->size) / (float)n_hcw_working;

        //TODO: should there be different max interactions for doctors / nurses?
        //Check whether the number of required interactions spread across working hcw is greater than the max possible. If it is, set it to the max.
        patient_interactions_per_hcw = (patient_interactions_per_hcw > max_hcw_daily_interactions) ? max_hcw_daily_interactions : patient_interactions_per_hcw;

        n_total_interactions = round(patient_interactions_per_hcw * n_hcw_working);

        all_required_interactions = calloc( n_patient_required_interactions * ward->patients->size, sizeof(long) );
        capped_hcw_interactions   = calloc( n_total_interactions, sizeof(long) );

        network->n_edges = 0;
        network->n_vertices       = n_hcw_working + ward->patients->size;

        n_pos = 0;

        for( idx = 0; idx < ward->patients->size; idx++ )
        {
            long current_patient_idx = list_element_at(ward->patients, idx);
            for( int i = 0; i < n_patient_required_interactions; i++)
                all_required_interactions[n_pos++] = current_patient_idx;
        }

        //shuffle list of all interactions
        gsl_ran_shuffle( rng, all_required_interactions, n_pos, sizeof(long) );

        idx = 0;
        hdx = 0;
        //assign network edges between hcw and randomly picked patient interactions
        //TODO: shift patterns eg day off
        while( idx < n_total_interactions )
        {
            network->edges[network->n_edges].id1 = hc_workers[ hdx ];
            network->edges[network->n_edges].id2 = all_required_interactions[ idx++ ];

            network->n_edges++;
            hdx++;
            if( hdx >= n_hcw_working )
                hdx = 0;
        }

        free( all_required_interactions );
        free( capped_hcw_interactions );
    }
}

/*****************************************************************************************
*  Name:		build_ward_networks
*  Returns:		int
******************************************************************************************/
int add_patient_to_ward( ward *ward, individual *indiv )
{
    if( ward->patients->size < ward->n_beds )
    {
        list_push_back( indiv->idx, ward->patients );
        return TRUE;
    }
    return FALSE;
}

/*****************************************************************************************
*  Name:		remove_patient_from_ward
*  Returns:		void
******************************************************************************************/
void remove_patient_from_ward( ward* ward, individual *indiv)
{
    list_remove_element( indiv->idx, ward->patients );
}

/*****************************************************************************************
*  Name:		ward_available_beds
*  Returns:		int
******************************************************************************************/
int ward_available_beds( ward* ward)
{
    return ward->n_beds - ward->patients->size;
}

/*****************************************************************************************
*  Name:		destroy_ward
*  Returns:		void
******************************************************************************************/
void destroy_ward( ward* ward )
{
    destroy_network( ward->doctor_patient_network );
    destroy_network( ward->nurse_patient_network );
    destroy_list( ward->patients );
    free( ward->patients );
    free( ward->doctors );
    free( ward->nurses );
}
