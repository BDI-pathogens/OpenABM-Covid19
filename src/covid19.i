/* covid19.i */
%module covid19
%{
#define SWIG_FILE_WITH_INIT
#include "model.h"
#include "params.h"
#include "constant.h"
#include "input.h"
#include "individual.h"
#include "utilities.h"
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_cdf.h>
%}

%rename (create_model) new_model(parameters *params);
%rename (create_event) new_event(model *model);

%nodefaultdtor;

%inline %{
extern gsl_rng * rng;
%}

%include "model.h"
%include "params.h"
%include "constant.h"
%include "input.h"
%include "individual.h"
%include "utilities.h"
%include model_utils.i 
%include params_utils.i

