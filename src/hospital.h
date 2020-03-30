/*
 * individual.h
 *
 *  Created on: 30 Mar 2020
 *      Author: vuurenk
 */

#ifndef HOSPITAL_H_
#define HOSPITAL_H_

/************************************************************************/
/******************************* Includes *******************************/
/************************************************************************/

#include "structure.h"
#include "utilities.h"
#include "constant.h"
#include "params.h"
#include "network.h"
#include "individual.h"

/************************************************************************/
/****************************** Structures  *****************************/
/************************************************************************/

typedef struct hospital hospital;

struct hospital {
    int hospital_idx;

    network *hospital_network;

    int available_beds;
    int available_icus;

    long *doctor_pdxs;
    long *nurse_pdxs;

    //TODO: Have a list of doctor and nurses in a hospital
//    individual *doctor_pdxs;
//    individual *nurse_pdxs;
//    individual *patient_pdxs;
};

#endif /* HOSPITAL_H_ */
