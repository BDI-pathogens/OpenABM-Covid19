/*
 * hsopital.c
 *
 *  Created on: 03 Apr 2020
 *      Author: vuurenk
 */

#include "nurse.h"

/*****************************************************************************************
*  Name:		initialize_nurse
*  Description: initializes and individual at the start of the simulation, note can
*  				only be called once per individual
*  Returns:		void
******************************************************************************************/
void initialise_nurse(
    nurse *nurse,
    long pdx,
    int hospital_idx,
    int ward_idx,
    int ward_type
)
{
    nurse->pdx          = pdx;
    nurse->hospital_idx = hospital_idx;
    nurse->ward_idx     = ward_idx;
    nurse->ward_type    = ward_type;
}
