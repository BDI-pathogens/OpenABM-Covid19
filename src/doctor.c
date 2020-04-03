/*
 * doctor.c
 *
 *  Created on: 03 Apr 2020
 *      Author: vuurenk
 */

#include "doctor.h"

/*****************************************************************************************
*  Name:		initialize_hospital
*  Description: initializes and individual at the start of the simulation, note can
*  				only be called once per individual
*  Returns:		void
******************************************************************************************/
void initialise_doctor(
    doctor *doctor,
    long pdx,
    int hospital_idx,
    int ward_idx,
    int ward_type
)
{
    doctor->pdx          = pdx;
    doctor->hospital_idx = hospital_idx;
    doctor->ward_idx     = ward_idx;
    doctor->ward_type    = ward_type;
}
