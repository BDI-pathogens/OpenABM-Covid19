"""
Class representing a network in the simulation

Created: 15 October 2020
Author: roberthinch
"""

import covid19, pandas as pd

class Network:
    """
    Network object has information about a specific network
    """

    def __init__(self, model, network_id):
        
        c_network = covid19.get_network_by_id( model.c_model, network_id )
        self._network_id       = network_id
        self.c_network         = c_network

    def n_edges(self):
        return covid19.network_n_edges( self.c_network )
    
    def n_vertices(self):
        return covid19.network_n_vertices( self.c_network )
    
    def name(self):
        return covid19.network_name( self.c_network ) 
    
    def network_id(self):
        return self._network_id   
    
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
    
    def set_network_transmission_multiplier(self,multiplier):
        covid19.set_network_transmission_multiplier(self.c_network,multiplier)
        
    def transmission_multiplier(self):
        return self.c_network.transmission_multiplier
    
    def transmission_multiplier_type(self):
        return self.c_network.transmission_multiplier_type

    def transmission_multiplier_combined(self):
        return self.c_network.transmission_multiplier_combined

    def show(self):
        print( "network_id        = " + str( self.network_id() ) )
        print( "name              = " + self.name() )
        print( "n_edges           = " + str( self.n_edges() ) )
        print( "n_vertices        = " + str( self.n_vertices() ) )
        print( "skip_hospitalised = " + str( self.skip_hospitalised() ) )
        print( "skip_quarantined  = " + str( self.skip_quarantined() ) )
        print( "type              = " + str( self.type() ) )
        print( "daily_fraction    = " + str( self.daily_fraction() ) )
        print( "transmission_mult = " + str( self.transmission_multiplier() ) )
        print( "transmission_mult_type = " + str( self.transmission_multiplier_type() ) )
        print( "transmission_mult_comb = " + str( self.transmission_multiplier_combined() ) )
    
    def get_network(self):
        """Return pandas.DataFrame of the network"""
        n_edges = self.n_edges()
        id1   = covid19.longArray(n_edges)
        id2   = covid19.longArray(n_edges)
        
        return_status = covid19.get_network(self.c_network, id1, id2)

        list_id1 = [None]*n_edges
        list_id2 = [None]*n_edges
        
        for idx in range(n_edges):
            list_id1[idx] = id1[idx]
            list_id2[idx] = id2[idx]
        
        df_network = pd.DataFrame( {
            'ID1': list_id1, 
            'ID2': list_id2
        })
        
        return df_network
