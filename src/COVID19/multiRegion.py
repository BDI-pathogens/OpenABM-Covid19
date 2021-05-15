import multiprocessing
import time
from numpy import double
import pandas as pd
import sys
import datetime
import math
from numpy.dual import norm
import numpy as np

import covid19
from model import PYTHON_SAFE_UPDATE_PARAMS
from model import Model

STEP_TYPE_TIME = 0
STEP_TYPE_PARAM_UPDATE = 1

RESULT_FIELDS = [
    "time",
    "total_infected",
    "total_death",
    "n_critical",
    "n_critical_cs"
]
RESULT_FIELDS_NON_NORMALIZE = [
    "time"
]

class WorkerJob(object):
    def __init__(
            self,
            e_go,
            e_global_wait,
            e_job,
            step_type,
            idx,
            data,
            n_regions,
            init_params,
            update_param,
            update_value,
            seed_inf,
            seed_strain_mult
        ):
        self._e_go   = e_go
        self._e_global_wait = e_global_wait
        self._e_job  = e_job
        self._step_type = step_type
        self._idx    = idx
        self._data   = data
        self._n_regions = n_regions
        self._init_params = init_params
        self._update_param = update_param
        self._update_value = update_value
        self._seed_inf     = seed_inf
        self._seed_strain_mult = seed_strain_mult
        self._n_critical_cs = 0

        self._e_go.wait()
        self._initialize_model()
        self._e_job.set()
        self._e_global_wait.wait()
          
    def _initialize_model(self):   
        self._model = Model( params = self._init_params )
           
    def _one_step_work(self):
        
        if self._seed_inf[ self._idx ] > 0 :
            self._model.seed_infect( self._seed_inf[ self._idx ], strain_multiplier = self._seed_strain_mult[ self._idx ])
        self._model.one_time_step()
        res = self._model.one_time_step_results()
        
        # add cumsum of n_critical by hand 
        self._n_critical_cs = self._n_critical_cs + res[ "n_critical" ]
        res["n_critical_cs"] = self._n_critical_cs 
        
        for fdx in range( len( RESULT_FIELDS ) ) :
            self._data[self._idx + (fdx * self._n_regions) ]= res[ RESULT_FIELDS[fdx]]        
      
    def _update_running_params(self):
        
        param = PYTHON_SAFE_UPDATE_PARAMS[self._update_param.value ]
        self._model.update_running_params(param,self._update_value.value)
                
    def run(self):
        self._e_go.wait()
        if self._step_type.value == STEP_TYPE_TIME :
            self._one_step_work()
        elif self._step_type.value == STEP_TYPE_PARAM_UPDATE :
            self._update_running_params()      
        self._e_job.set()
        self._e_global_wait.wait()
        
            
def create_WorkerJob(
        e_go, 
        e_global_wait,
        e_job, 
        step_type,
        idx,
        shared_data,
        n_regions,
        init_params,
        update_param,
        update_value,
        seed_inf,
        seed_strain_mult
    ):
    worker = WorkerJob(e_go, e_global_wait,e_job,step_type,idx,shared_data,n_regions,init_params,update_param,update_value,seed_inf,seed_strain_mult)
    
    while True :
        worker.run()
        
        
class MultiRegionModel(object):
    def __init__(self,params,index_col="idx", max_steps = 1000,turn_off_contract_tracing = True ):

        n_regions = len( params.index ) 
        
        self._n_regions   = n_regions
        self._max_steps   = max_steps
        self._e_global_wait = multiprocessing.Event()
        self._e_jobs_go   = []
        self._e_jobs_wait = []
        self._processes   = []
        self._pause_region = [False] * n_regions
        self._time_offsets = [0] * n_regions
     
        self._results      = multiprocessing.Array("i", n_regions * len( RESULT_FIELDS ) )   
        self._index_values = params[ index_col ].tolist()
        self._n_total      = params[ "n_total" ].tolist()
        self._index_col    = index_col
        self._step_type    = multiprocessing.Value('i',0)
        self._update_param = multiprocessing.Value('i',0)
        self._update_value = multiprocessing.Value('d', 0)
        self._seed_inf     = multiprocessing.Array("i", n_regions )   
        self._seed_strain_mult = multiprocessing.Array("d", n_regions )   
        self._result_ts_init()
        
        params.drop(columns=[ index_col ],inplace=True)
        params = params.to_dict('records')
        
        for j in range( n_regions) :
            e_job = multiprocessing.Event()
            e_go = multiprocessing.Event()
            self._results[j] = 0
            self._seed_inf[j] = 0
            self._seed_strain_mult[j] = 1
            
            init_params = params[j]
            if "rng_seed" not in init_params : 
                init_params["rng_seed"] = j
            if turn_off_contract_tracing :
                init_params[ "days_of_interactions" ] = 1
                init_params[ "quarantine_days" ]      = 1  
           
            self._e_jobs_go.append(e_go)
            self._e_jobs_wait.append(e_job)
            process = multiprocessing.Process(
                name   = 'block', 
                target = create_WorkerJob,
                args   = (e_go,self._e_global_wait,e_job,self._step_type,j,self._results,n_regions,init_params,self._update_param,self._update_value,self._seed_inf, self._seed_strain_mult)
            )
            process.start()
            self._processes.append(process)
            
        self._time = 0;
        self.one_step_wait()

    def __del__(self):
        for j in range( self._n_regions ) :
            self._processes[j].terminate()
            
    def one_step_wait(self):
        self._step_type.value = STEP_TYPE_TIME
        
        # sets the first set of jobs off and blocks the end of the process
        self._e_global_wait.clear()
        for j in range( self._n_regions ) :
            if not self._pause_region[j] :
                self._e_jobs_go[j].set()
        
        # wait for the job to be done and rest counters
        for j in range( self._n_regions ) :
            if not self._pause_region[j] :
                self._e_jobs_wait[j].wait()
       
        for j in range( self._n_regions ) :
            self._e_jobs_wait[j].clear()
            self._e_jobs_go[j].clear()
               
        # get the the jovs ready for the next setp
        self._e_global_wait.set()
        self._time = self._time + 1
        
        self._results_ts_dt = pd.concat( [ self._results_ts_dt, self.result_dt() ] )
        
    def update_running_params(self,param,value,index=None,index_val=None):

        if ( index == None ) & ( index_val == None ) :
            for j in range( self._n_regions) :
                self.update_running_params(param, value, index=j)
            return
        
        if ( index == None) & ( index_val != None ) :
            index = self._index_values.index(index_val)
        
        # set the param to update and value in shared memory        
        param_idx = PYTHON_SAFE_UPDATE_PARAMS.index(param)
        self._step_type.value    = STEP_TYPE_PARAM_UPDATE  
        self._update_param.value = param_idx
        self._update_value.value = value
                
        # fire off the update in the relevant job
        self._e_global_wait.clear()
        self._e_jobs_go[index].set()
        self._e_jobs_wait[index].wait()
        self._e_jobs_go[index].clear()
        self._e_jobs_wait[index].clear()
        self._e_global_wait.set()
        
    def _set_pause_in_region(self,region,pause):
        
        if region < 0 or region >= self._n_regions :
            raise( "region index is invalid" )
    
        self._pause_region[region] = pause
      
    def _set_pause(self,pause):
        
        for region in range( self._n_regions ) : 
           self._pause_region[region] = pause
           
    def result_array(self,field):
        
        if not field in RESULT_FIELDS :
            raise( "field not known" )
        
        fdx = RESULT_FIELDS.index(field)
        fdx = fdx * self._n_regions
        
        res = [];
        for idx in range( self._n_regions ) :
            res.append(self._results[fdx+idx])
           
        # adjust the time of each model of offsets due to synchronisation
        if field == "time" :   
            for idx in range( self._n_regions ) :
                res[idx] = res[idx] + self._time_offsets[idx]
                
        return res
    
    def result_dt(self):
        
        dt = pd.DataFrame( data = { self._index_col : self._index_values })
        
        for field in RESULT_FIELDS :
            dt[ field ] = self.result_array( field )
                    
        return dt      
    
    def result_ts(self,normalize=False):   
        dt = self._results_ts_dt
        
        if normalize :
            dt_pop = pd.DataFrame( data = { "n_total":self._n_total, self._index_col : self._index_values } )
            dt = pd.merge( dt, dt_pop, on = self._index_col )
            
            for col in RESULT_FIELDS :
                if col not in RESULT_FIELDS_NON_NORMALIZE :
                    dt[ col ] = dt[ col ] / dt[ "n_total" ] * 1e5
            dt.drop( columns = [ "n_total" ], inplace = True )
            
        return dt
    
    def _result_ts_init(self):   
        self._results_ts_dt = pd.DataFrame(columns = RESULT_FIELDS )
         
    def step_to_synch_point(self,synch_param,synch_value,verbose = True, maxSteps = 100):
        
        # reset the results_ts and the offsets
        self._result_ts_init()
        self._time_offsets = [0] * self._n_regions
        initial_time      = self.result_array( "time" )
        self._set_pause( False )
        
        if not isinstance(synch_value, list ) :
            synch_value = [ synch_value ] * self._n_regions
        
        for step in range( maxSteps ) :
        
            # pause procesess that have reached the synch point
            res = self.result_array( synch_param )    
            n_waiting = 0     
            for j in range( n_regions ) :
                if res[ j ] >= synch_value[ j ] :
                    self._set_pause_in_region( j, True )
                    n_waiting = n_waiting + 1;
            
            if verbose :
                print( "step " + str( step ) + "; " + str( n_waiting ) + "/" + str( self._n_regions ) + " reached synch point" )
             
            # if all at synch point, wait  
            if n_waiting == self._n_regions :
                model._set_pause( False )
                break;
            
             
            model.one_step_wait()
        
        # failure to synch
        if n_waiting != self._n_regions :
            model._set_pause( False )
            if verbose :
                print( "Not all regions reached the target after maxSteps" )
            return False    
    
        # if synched calculate the time offset between regions
        final_time = self.result_array( "time" )
        dif_time   = final_time
        for j in range( n_regions ) :
            dif_time[j] = dif_time[j] - initial_time[j]
        t_align    = min( dif_time )
        for j in range( n_regions ) :
            self._time_offsets[j] = - dif_time[j] - initial_time[j]
        
        # replace the stored ts data with the aligned time
        dt_result  = self.result_ts()
        dt_offsets = pd.DataFrame( data = { "time_offset": self._time_offsets,  self._index_col : self._index_values } ) 
        dt_result  = pd.merge( dt_result, dt_offsets, on = self._index_col )
        dt_result[ "time" ] = dt_result[ "time" ] + dt_result[ "time_offset" ]
        dt_result = dt_result[ dt_result["time"] > -t_align ]
        dt_result.drop( columns = [ "time_offset" ], inplace = True )
        dt_result.drop_duplicates( inplace=True )
        self._results_ts_dt = dt_result

        return True
      