"""
Class representing a vaccine

Created: 27 May 2021
Author: roberthinch
"""

import covid19, pandas as pd

class Vaccine:
    """
    Vaccine object has information about a specific vaccine
    """

    def __init__(self, model, vaccine_id):
        
        c_vaccine        = covid19.get_vaccine_by_id( model.c_model, vaccine_id )
        self._vaccine_id = vaccine_id
        self.c_vaccine   = c_vaccine
        
    def idx(self):
        return covid19.vaccine_idx( self.c_vaccine )    
  
    def full_efficacy(self):
        return covid19.vaccine_full_efficacy( self.c_vaccine )

    def symptoms_efficacy(self):
        return covid19.vaccine_symptoms_efficacy( self.c_vaccine )
    
    def severe_efficacy(self):
        return covid19.vaccine_severe_efficacy( self.c_vaccine )

    def time_to_protect(self):
        return covid19.vaccine_time_to_protect( self.c_vaccine )

    def vaccine_protection_period(self):
        return covid19.vaccine_vaccine_protection_period( self.c_vaccine )
    
    def name(self):
        return covid19.vaccine_name( self.c_vaccine )

    def show(self):
        print( "idx               = " + str( self.idx() ) )
        print( "name              = " + self.name() )
        print( "full_efficacy     = " + str( self.full_efficacy() ) )
        print( "symptoms_efficacy = " + str( self.symptoms_efficacy() ) )
        print( "time_to_protect   = " + str( self.time_to_protect() ) )
        print( "vaccine_protection_period  = " + str( self.vaccine_protection_period() ) )
       

