/*
 * doctor.c
 *
 *  Created on: 03 Apr 2020
 *      Author: vuurenk
 */

#include "doctor.h"

/*****************************************************************************************
*  Name:		initialise_doctor
*  Description: initialises a doctor and assigns them to a hospital at the start of the
*               simulation. Can only be called once per individual.
*  Returns:		void
******************************************************************************************/
void initialise_doctor(
    doctor *doctor,
    individual *indiv,
    int hospital_idx,
    int ward_idx,
    int ward_type
)
{
    doctor->pdx          = indiv->idx;
    doctor->hospital_idx = hospital_idx;
    doctor->ward_idx     = ward_idx;
    doctor->ward_type    = ward_type;
}
