%module network_utils

#include <string.h>
%include <cstring.i>
%cstring_bounded_output(char* outstr, 1024);

%inline %{
int network_n_edges( network *network ) {
    return network->n_edges;
}

char *network_name( network *network ) {
	return network->name;
}

%}