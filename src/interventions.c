/*
 * interventions.c
 *
 *  Created on: 18 Mar 2020
 *      Author: hinchr
 */

#include "model.h"
#include "individual.h"
#include "utilities.h"
#include "constant.h"
#include "params.h"
#include "network.h"
#include "disease.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_cdf.h>

/*****************************************************************************************
*  Name:		intervention_on_quarantine_release
*  Description: Release an individual held in quarantine
*  Returns:		void
******************************************************************************************/
void intervention_quarantine_release( model *model, individual *indiv )
{
	remove_event_from_event_list( model, indiv->quarantine_event );
	if( indiv->quarantine_release_event != NULL )
		remove_event_from_event_list( model, indiv->quarantine_release_event );
	set_quarantine_status( indiv, model->params, model->time, FALSE );
}

