/*
 * nurse.h
 *
 *  Created on: 03 Apr 2020
 *      Author: vuurenk
 */

#ifndef nurse_H_
#define nurse_H_

/************************************************************************/
/******************************* Includes *******************************/
/************************************************************************/
//#include "individual.h"

/************************************************************************/
/****************************** Structures  *****************************/
/************************************************************************/

typedef struct nurse nurse;

struct nurse
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

void initialise_nurse( nurse *nurse, long, int, int, int );

#endif /* nurse_H_ */
