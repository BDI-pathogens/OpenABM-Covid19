/*
 * individual.c
 *
 *  Created on: 30 Mar 2020
 *      Author: vuurenk
 */

#include "hospital.h"
#include "params.h"
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
    network *hospital_network,
    int hdx
)
{
    if( hospital->hospital_idx != 0 )
        print_exit( "a hospital can only be initialised once!");

    hospital->hospital_idx = hdx;
    hospital->hospital_network = hospital_network;

    hospital->available_beds = params->hospital_n_beds;
    hospital->available_icus = params->hospital_n_icus;

    //TODO: Change this to store the actual list of workers.
    hospital->doctor_pdxs = calloc(params->n_doctors, sizeof(long) );
    hospital->nurse_pdxs = calloc(params->n_nurses, sizeof(long) );
}
