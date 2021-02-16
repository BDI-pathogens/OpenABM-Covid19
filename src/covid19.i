/* covid19.i */
%module covid19
%{
#define SWIG_FILE_WITH_INIT
#define SWIG_PYTHON_CAST_MODE
#include "model.h"
#include "params.h"
#include "constant.h"
#include "input.h"
#include "individual.h"
#include "interventions.h"
#include "utilities.h"
#include "disease.h"
#include "network.h"
%}

%rename (create_model) new_model(parameters *params);
%rename (create_event) new_event(model *pmodel);

%nodefaultdtor;

/* These structs shouldn't be used directly (memory is managed by create_model
 * and destroy_model). Ignoring these ctors is necessary to suppress NOTEs
 * from `R CMD check`. */
%nodefaultctor directory;
%nodefaultdtor directory;
%nodefaultctor edge;
%nodefaultdtor edge;
%nodefaultctor event;
%nodefaultdtor event;
%nodefaultctor event_list;
%nodefaultdtor event_list;
%nodefaultctor individual;
%nodefaultdtor individual;
%nodefaultctor infection_event;
%nodefaultdtor infection_event;
%nodefaultctor interaction;
%nodefaultdtor interaction;
%nodefaultctor interaction_block;
%nodefaultdtor interaction_block;
%nodefaultctor incomplete_gamma_p_params;
%nodefaultdtor incomplete_gamma_p_params;
%nodefaultctor trace_token;
%nodefaultdtor trace_token;

%include "model.h"
%include "params.h"
%include "constant.h"
%include "input.h"
%include "individual.h"
%include "interventions.h"
%include "utilities.h"
%include "disease.h"
%include "network.h"
%include model_utils.i 
%include params_utils.i
%include network_utils.i

