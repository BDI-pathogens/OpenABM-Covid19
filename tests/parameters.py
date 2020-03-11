"""
ParameterSet class definition for basic modifications of parameter files for COVID19-IBM

Created: 10 March 2020
Author: p-robot (W. Probert)
"""

from collections import OrderedDict

class ParameterSet(object):
    """
    Class representing a single parameter set for COVID19-IBM
    
    Stores the parameter file as an OrderedDict.  
    
    Arguments
    ---------
    param_file: 
        path to a parameter file of the COVID19-IBM
    line_number : 
        int of the line number of the parameter file to read (header line is 0)
    
    
    Methods
    -------
    get_param(param)
        List parameter value for 'param'
    set_param(param, value)
        Set parameter value for 'param' to 'value'
    list_params()
        List ordered dictionary that stores parameters
    write_params(param_file)
        Write parameter set to csv file at location 'param_file'
    
    Example
    -------
    params = ParameterSet("./tests/data/test_parameters.csv")
    params.get_param("mean_daily_interactions")
    
    params.set_param("mean_daily_interactions", 5)
    params.write_params("new_parameter_file.csv")
    
    """
    def __init__(self, param_file, line_number = 1):
        
        # Read-in parameter file
        with open(param_file) as f:
            read_data = f.read()
        data = read_data.split("\n")
        
        # Pull header and parameter line of interest (line number from the parameter file)
        header = [c.strip() for c in data[0].split(",")]
        param_line = [c.strip() for c in data[line_number].split(",")]
        
        self.NPARAMS = len(header)
        
        # Save parameters as an ordered dict
        self.params = OrderedDict([(param, value) for param, value in zip(header, param_line)])
    
    def get_param(self, param):
        return(self.params[param])

    def set_param(self, param, value):
        self.params[param] = str(value)

    def list_params(self):
        return(self.params.keys())
    
    def write_params(self, param_file):
        
        header = ", ".join(list(self.params.keys()))
        line = ", ".join(list(self.params.values()))
        
        with open(param_file, "w+") as f:
            f.write(header + "\n" + line)

