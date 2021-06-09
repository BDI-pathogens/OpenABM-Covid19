"""
Class representing a strain

Created: 2nd June 2021
Author: roberthinch
"""

import covid19, pandas as pd

class Strain:
    """
    Strain object has information about a specific strain
    """

    def __init__(self, model, strain_id):
    
        c_strain        = covid19.get_strain_by_id( model.c_model, strain_id )
        self._strain_id = strain_id
        self.c_strain   = c_strain
        
    def idx(self):
        return covid19.strain_idx( self.c_strain )    
    
    def transmission_multiplier(self):
        return covid19.strain_transmission_multiplier( self.c_strain )    

    def show(self):
        print( "idx                     = " + str( self.idx() ) )
        print( "transmission_multiplier = " + str( self.transmission_multiplier() ) )
