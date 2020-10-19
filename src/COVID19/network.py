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
        
        c_network = covid19.get_network_by_id( model.c_model, network_id )
        self.network_id        = network_id
        self.c_network         = c_network

    def n_edges(self):
        return covid19.network_n_edges( self.c_network )
    
    def n_vertices(self):
        return covid19.network_n_vertices( self.c_network )
    
    def name(self):
        return covid19.network_name( self.c_network ) 
    
    def skip_hospitalised(self):
        return covid19.network_skip_hospitalised( self.c_network )
    
    def skip_quarantined(self):
        return covid19.network_skip_quarantined( self.c_network )
    
    def type(self):
        return covid19.network_type( self.c_network )
    
    def daily_fraction(self):
        return covid19.network_daily_fraction( self.c_network )
    
    def update_daily_fraction(self,daily_fraction):
        return covid19.update_daily_fraction(self.c_network,daily_fraction)

    def show(self):
        print( "network_id        = " + str( self.network_id ) )
        print( "name              = " + self.name() )
        print( "n_edges           = " + str( self.n_edges() ) )
        print( "n_vertices        = " + str( self.n_vertices() ) )
        print( "skip_hospitalised = " + str( self.skip_hospitalised() ) )
        print( "skip_quarantined  = " + str( self.skip_quarantined() ) )
        print( "type              = " + str( self.type() ) )
        print( "daily_fraction    = " + str( self.daily_fraction() ) )
        

   