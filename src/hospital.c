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

/*****************************************************************************************
*  Name:		destroy_hospital
*  Description: Destroys the model structure and releases its memory
******************************************************************************************/
void destroy_hospital( hospital *hospital)
{
    free( hospital->doctor_pdxs );
    free( hospital->nurse_pdxs );
};


