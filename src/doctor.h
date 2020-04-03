/*
 * doctor.h
 *
 *  Created on: 03 Apr 2020
 *      Author: vuurenk
 */

#ifndef DOCTOR_H_
#define DOCTOR_H_

/************************************************************************/
/******************************* Includes *******************************/
/************************************************************************/
//#include "individual.h"

/************************************************************************/
/****************************** Structures  *****************************/
/************************************************************************/

typedef struct doctor doctor;

struct doctor
{
    int hospital_idx;               //hospital index number
    int ward_idx;                   //ward index number
    int ward_type;
    long pdx;                       //population index number
    //individual *indiv; //TODO: maybe just use this instead of pop index???
};

/************************************************************************/
/******************************  Functions  *****************************/
/************************************************************************/

void initialise_doctor( doctor *doctor, long, int, int, int );

#endif /* DOCTOR_H_ */
