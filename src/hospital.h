/*
 * individual.h
 *
 *  Created on: 30 Mar 2020
 *      Author: vuurenk
 */

#ifndef INDIVIDUAL_H_
#define INDIVIDUAL_H_

/************************************************************************/
/******************************* Includes *******************************/
/************************************************************************/


/************************************************************************/
/****************************** Structures  *****************************/
/************************************************************************/

typedef struct hospital hospital;

struct hospital{
    int hospital_idx;

    int available_beds;
    int available_icus;

    long *doctor_pdxs;
    long *nurse_pdxs;
};

#endif /* INDIVIDUAL_H_ */
