/* covid19.i */
%module covid19
%{
#define SWIG_FILE_WITH_INIT
#include "model.h"
#include "params.h"
#include "constant.h"
#include "input.h"
%}

/*
parameters params;
model *model;
model* new_model(parameters *params);
int one_time_step(model *model);
void destroy_model(model *model);
void destroy_params(parameters *params);
*/
%rename (create_model) new_model(parameters *params);
%rename (create_event) new_event(model *model);
%include "model.h"
%include "params.h"
%include "constant.h"
%include "input.h"
