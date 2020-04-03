/* covid19.i */
%module covid19
%{
#define SWIG_FILE_WITH_INIT
#include "model.h"
#include "params.h"
#include "constant.h"
#include "input.h"
#include "swig_utils.h"
%}

%rename (create_model) new_model(parameters *params);
%rename (create_event) new_event(model *model);

%rename (create_model) new_model(parameters *params);
%rename (create_event) new_event(model *model);
%include "model.h"
%include "params.h"
%include "constant.h"
%include "input.h"

