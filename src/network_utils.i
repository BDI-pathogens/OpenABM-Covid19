%module network_utils

#ifndef SWIGR
#include <string.h>
%include <cstring.i>
%cstring_bounded_output(char* outstr, 1024);
#endif

%inline %{
int network_n_edges( network *pnetwork ) {
    return pnetwork->n_edges;
}

int network_n_vertices( network *pnetwork ) {
    return pnetwork->n_vertices;
}

char *network_name( network *pnetwork ) {
	return pnetwork->name;
}

int network_skip_hospitalised( network *pnetwork ) {
	return pnetwork->skip_hospitalised;
}

int network_skip_quarantined( network *pnetwork ) {
	return pnetwork->skip_quarantined;
}

int network_type( network *pnetwork ) {
	return pnetwork->type;
}

double network_daily_fraction( network *pnetwork ) {
	return pnetwork->daily_fraction;
}

float network_transmission_multiplier( network *pnetwork ) {
	return pnetwork->transmission_multiplier;
} 

float network_transmission_multiplier_type( network *pnetwork ) {
	return pnetwork->transmission_multiplier_type;
} 

float network_transmission_multiplier_combined( network *pnetwork ) {
	return pnetwork->transmission_multiplier_combined;
} 

void set_network_transmission_multiplier( network *pnetwork, float val ) {
	update_transmission_multiplier( pnetwork, val );
	pnetwork->transmission_multiplier = val;
}

int get_network( network *pnetwork, long *id1_array, long *id2_array) {
    
    long idx;
    
    // Loop through all edges in the network
    for(idx = 0; idx < pnetwork->n_edges; idx++)
    {
        id1_array[idx] = pnetwork->edges[idx].id1;
        id2_array[idx] = pnetwork->edges[idx].id2;
    }
    
	return TRUE;
}

%}