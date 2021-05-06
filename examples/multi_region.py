import pandas as pd
import sys
import time
import os

sys.path.append("../src/COVID19")
from multiRegion import MultiRegionModel

          
if __name__ == '__main__':
    # some OS limit the number of process which can be spawned by default
    # for OSX it can ge increased with the following line, probably needs to be
    # changed for other operating systems 
    os.system( "ulimit -Sn 10000")
    
    max_steps = 100
    n_pops    = 10
    n_total   = 100000
    
    params = pd.DataFrame( { 
        "n_total" : [ n_total ] * n_pops, 
        "idx" : range( n_pops ),
        "quarantine_days" : 1,
        "days_of_interactions" : 1,
        "rebuild_networks" : 0
    } )
    model = MultiRegionModel( params )
    
    t_start = time.time()
    initial_growth = True
    last_infected = 0
    for day in range( max_steps ) :   
        model.one_step_wait()      
        
        total_infected = sum(model.result_array( "total_infected"))
        print( "Time " + str( day ) + 
               "; daily_infected = " + str( total_infected - last_infected ) + 
               "; total_infected = " + str( total_infected ) )
       
        if initial_growth == True :      
          
            if total_infected  > n_pops * n_total * 0.01 :
                initial_growth = False
                for idx  in range(n_pops):
                    model.update_running_params("relative_transmission_occupation", 0.3, index_val = idx)
                    model.update_running_params("relative_transmission_random", 0.3 , index_val = idx)

        last_infected = total_infected
        
    t_end = time.time()
    
    print( "time = " + str( t_end - t_start) + 
           "s; n_steps = " + str(max_steps )+ 
           "; time per step = " + str( (t_end - t_start)/max_steps) )
        
    del( model )

    
