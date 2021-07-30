%module strain_utils

#ifndef SWIGR
#include <string.h>
%include <cstring.i>
%cstring_bounded_output(char* outstr, 1024);
#endif

%inline %{

short strain_idx( strain *pstrain ) {
    return pstrain->idx;
}

float strain_transmission_multiplier( strain *pstrain ) {
    return pstrain->transmission_multiplier;
}

long strain_total_infected( strain *pstrain ) {
    return pstrain->total_infected;
}

%}
