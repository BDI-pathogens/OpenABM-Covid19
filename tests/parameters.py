"""
ParameterSet class definition for basic modifications of parameter files for COVID19-IBM

Created: 10 March 2020
Author: p-robot (W. Probert)
"""

from collections import OrderedDict
import itertools

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
    
    write_varying_params(params, values_list, param_file, 
        index_var = "param_id", reset_index = True)
        
        params : list, values: list, param_file: str of path
        
        Write several parameter sets to a single file at location
        'param_file' for all combinations of lists of parameter values
        given in 'values_list' and parameter names given in list 'params'.  
        The index_var parameter specifies the ID column of the new file.
        If reset_index = True then a new ID column is produced to index
        the parameter sets.  
    
    Example
    -------
    # Load a parameter set
    params = ParameterSet("./tests/data/test_parameters.csv")
    
    # Get parameter value for parameter called "n_total"
    params.get_param("n_total")
    
    # Set parameter called "n_total" to 500
    params.set_param("n_total", 500)
    
    # Write parameter set to file in a file called "new_parameter_file.csv"
    params.write_params("new_parameter_file.csv")
    
    # Write the same parameter set to file for 5 different values of the random seed (rng_seed)
    params = ParameterSet("./tests/data/test_parameters.csv")
    params.write_varying_params(["rng_seed"], [range(5)], "new_parameter_file.csv")
    
    
    # Write the same parameter set to file for 5 different values of the random seed (rng_seed) and
    # for different values of "infectious_rate".  
    
    params = ParameterSet("./tests/data/test_parameters.csv")
    
    param_names = ["rng_seed", "infectious_rate"]
    values_list = [range(5), [0.1, 0.2, 0.3]]
    
    params.write_varying_params(param_names, values_list, "new_parameter_file.csv")
    
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
    
    def write_varying_params(self, params, values_list, param_file, 
            index_var = "param_id", reset_index = True):
        
        header = ", ".join(list(self.params.keys()))
        
        lines = []; lines.append(header)
        
        index = 1
        for values in list(itertools.product(*values_list)):
            for (param, v) in zip(params, values):
                # Adjust the parameter value to v
                self.set_param(param, v)
                
                if reset_index:
                    self.set_param(index_var, index)
                
                # Create a list of parameter values to save
                lines.append(", ".join(list(self.params.values())))
                index += 1
        
        with open(param_file, "w+") as f:
            f.write("\n".join(lines))
