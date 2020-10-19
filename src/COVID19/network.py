"""
Class representing a network in the simulation

Created: 15 October 2020
Author: roberthinch
"""

import covid19

class Network:
    """
    Network object has information about a specific network
    """

    def __init__(self, model, network_id):
        
        network = covid19.get_network_by_id( model, network_id )
        
        self.network_id        = network_id
        self.n_edges           = covid19.network_n_edges( network )
        self.n_vertices        = covid19.network_n_vertices( network )
        self.name              = covid19.network_name( network ) 
        self.skip_hospitalised = covid19.network_skip_hospitalised( network )
        self.skip_quarantined  = covid19.network_skip_quarantined( network )
        self.type              = covid19.network_type( network )
        self.daily_fraction    = covid19.network_daily_fraction( network )

    def show(self):
        print( "network_id = " + str( self.network_id ) )
#        print( "name       = " + self.name)
        print( self.name)
        print( "n_edges    = " + str( self.n_edges ) )

   