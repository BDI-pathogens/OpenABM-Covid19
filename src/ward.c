/*
 * doctor.c
 *
 *  Created on: 03 Apr 2020
 *      Author: vuurenk
 */

#include "ward.h"
#include "constant.h"

/*****************************************************************************************
*  Name:		initialize_hospital
*  Description: initializes and individual at the start of the simulation, note can
*  				only be called once per individual
*  Returns:		void
******************************************************************************************/
void initialise_ward(
    ward *ward,
    int ward_idx,
    int type
)
{
    ward->ward_idx = ward_idx;
    ward->type = type;
}

void set_up_ward_networks( ward* ward )
{
    int interaction_type = ( ward->type == COVID_GENERAL ) ? HOSPITAL_DOCTOR_PATIENT_GENERAL : HOSPITAL_DOCTOR_PATIENT_ICU;
    ward->doctor_patient_network = new_network( ward->n_doctors, interaction_type );
    ward->doctor_patient_network->edges = NULL;
    //TODO: there must be a better way of getting these interactiont type enums
    interaction_type = ( ward->type == COVID_GENERAL ) ? HOSPITAL_DOCTOR_PATIENT_GENERAL : HOSPITAL_DOCTOR_PATIENT_ICU;
    ward->nurse_patient_network = new_network( ward->n_nurses, interaction_type );
    ward->nurse_patient_network->edges = NULL;
}

void destroy_ward( ward* ward )
{
    free( ward->doctor_patient_network );
    free( ward->nurse_patient_network );
}

