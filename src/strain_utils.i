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

short strain_transmission_multiplier( strain *pstrain ) {
    return pstrain->transmission_multiplier;
}

%}
