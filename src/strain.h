/*
 * strain.h
 *
 *  Created on: 31 Mar 2021
 *      Author: nikbaya
 */

#ifndef STRAIN_H_
#define STRAIN_H_

/************************************************************************/
/****************************** Structures  *****************************/
/************************************************************************/

typedef struct strain strain;

struct strain{
	long idx;
	float transmission_multiplier;
};

#endif /* STRAIN_H_ */