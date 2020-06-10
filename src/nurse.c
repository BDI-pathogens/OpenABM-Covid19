/*
 * hospital.c
 *
 *  Created on: 03 Apr 2020
 *      Author: vuurenk
 */

#include "nurse.h"

/*****************************************************************************************
*  Name:		initialise_nurse
*  Description: initialises a nurse at the start of the simulation and assigns them to
*               a hospital. Can only be called once per individual.
*  Returns:		void
******************************************************************************************/
void initialise_nurse(
    nurse *nurse,
    individual *indiv,
    int hospital_idx,
    int ward_idx,
    int ward_type
)
{
    nurse->pdx          = indiv->idx;
    nurse->hospital_idx = hospital_idx;
    nurse->ward_idx     = ward_idx;
    nurse->ward_type    = ward_type;
}
