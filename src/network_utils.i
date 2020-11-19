%module network_utils

#include <string.h>
%include <cstring.i>
%cstring_bounded_output(char* outstr, 1024);

%inline %{
int network_n_edges( network *network ) {
    return network->n_edges;
}

int network_n_vertices( network *network ) {
    return network->n_vertices;
}

char *network_name( network *network ) {
	return network->name;
}

int network_skip_hospitalised( network *network ) {
	return network->skip_hospitalised;
}

int network_skip_quarantined( network *network ) {
	return network->skip_quarantined;
}

int network_type( network *network ) {
	return network->type;
}

double network_daily_fraction( network *network ) {
	return network->daily_fraction;
}

%}