/*
 * individual.c
 *
 *  Created on: 30 Mar 2020
 *      Author: vuurenk
 */

#include "hospital.h"
#include "constant.h"
#include "utilities.h"
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

    hospital->hospital_idx = hdx;
    hospital->available_beds = params->hospital_n_beds;
    hospital->available_icus = params->hospital_n_icus;

    hospital->doctor_pdxs = calloc( params->n_total_doctors, sizeof(long) );
    hospital->nurse_pdxs = calloc( params->n_total_nurses, sizeof(long) );
}

/*****************************************************************************************
*  Name:		add_nurse
*  Description: adds population id of a doctor to the list of
*               doctor population ids at the hospital
******************************************************************************************/
void add_healthcare_worker_to_hospital(hospital *hospital, int idx, long pdx, int type)
{
    if( type == DOCTOR )
        hospital->doctor_pdxs[idx] = pdx;
    else if( type == NURSE )
        hospital->nurse_pdxs[idx] = pdx;
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


