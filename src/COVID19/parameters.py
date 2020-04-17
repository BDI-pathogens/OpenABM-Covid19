"""
ParameterSet class definition for basic modifications of parameter files for COVID19-IBM

Created: 10 March 2020
Author: p-robot (W. Probert)
"""

import copy
import itertools
import json
import sys
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

    # Write the same parameter set to file for 5 different values of the
    # random seed (rng_seed)
    params = ParameterSet("./tests/data/test_parameters.csv")
    params.write_varying_params(["rng_seed"], [range(5)], "new_parameter_file.csv")


    # Write the same parameter set to file for 5 different values of the random seed
    # (rng_seed) and for different values of "infectious_rate".

    params = ParameterSet("./tests/data/test_parameters.csv")

    param_names = ["rng_seed", "infectious_rate"]
    values_list = [range(5), [0.1, 0.2, 0.3]]

    params.write_varying_params(param_names, values_list, "new_parameter_file.csv")

    """

    def __init__(self, param_file, line_number=1):

        # Read-in parameter file
        with open(param_file) as f:
            read_data = f.read()
        data = read_data.split("\n")

        # Pull header and parameter line of interest
        # (line number from the parameter file)
        header = [c.strip() for c in data[0].split(",")]
        param_line = [c.strip() for c in data[line_number].split(",")]

        self._NPARAMS = len(header)

        # Save parameters as an ordered dict
        self.params = OrderedDict(
            [(param, value) for param, value in zip(header, param_line)]
        )

    def get_param(self, param):
        """Get parameter value

        Arguments
        ---------
        param : str
            Parameter name

        Returns
        -------
        parameter value as a str
        """
        # Check the parameter exists in the parameter set dictionary
        assert param in self.params, "Parameter {} does not exist.".format(param)

        return self.params[param]

    def set_param(self, param, value=None):
        """Set parameter value

        Arguments
        ---------
        param : str or dict
            Parameter name (or a dict of parameter name-value key-value pairs)
        value : int, float, str
            Parameter value to set as value for parameter `param`
            (None if 'param' is a dict)

        Returns
        -------
        Nothing.
        """

        if isinstance(param, dict):
            for p, v in param.items():
                self.set_param(p, v)
        else:
            # Check the parameter exists in the parameter set dictionary
            assert param in self.params, "Parameter {} does not exist.".format(param)
            self.params[param] = str(value)

    def list_params(self):
        return self.params.keys()

    def write_varying_params_from_json(self, json_file, output_param_file):
        """
        Read JSON file of parameter values to vary, write parameter set to file
        """

        # Read JSON file of parameters
        with open(json_file) as f:
            json_string = f.read()
        json_dict = json.loads(json_string)

        # Pull out the list of parameters
        parameter_dict = json_dict["parameters"]

        # Check that n_replicates and rng_seed aren't both specified
        if ("n_replicates" in json_dict) & ("rng_seed" in parameter_dict):
            print("Both n_replicates and rng_seed are specified - only specify one")
            sys.exit()

        # Adjust single parameter values
        for param, value in parameter_dict.items():
            if not isinstance(value, list):
                self.set_param(param, value)

        # Make a dict of param values that are varying
        params = [k for k, v in parameter_dict.items() if isinstance(v, list)]
        values_list = [v for k, v in parameter_dict.items() if isinstance(v, list)]

        if "n_replicates" in json_dict:
            n_replicates = json_dict["n_replicates"]
            params = params + ["rng_seed"]
            values_list = values_list + [range(n_replicates)]

        # Write varying parameters to file
        self.write_varying_params(params, values_list, output_param_file)

    def write_params(self, param_file):
        """Write parameters to CSV file

        Arguements
        ----------
        param_file: str
            Path to CSV file where parameter file should be written
        """

        header = ", ".join(list(self.params.keys()))
        line = ", ".join(list(self.params.values()))

        with open(param_file, "w+") as f:
            f.write(header + "\n" + line)

    def write_varying_params(
        self,
        params,
        values_list,
        output_param_file,
        index_var="param_id",
        reset_index=True,
    ):
        """
        Write parameters to file from lists of parameter values across which to vary

        Arguments
        ---------
        params : list
            str of parameter names over which to vary

        values_list : list of lists

        param_file : str
            path to output parameter file

        index_var = "param_id", reset_index = True
        """
        header = ", ".join(list(self.params.keys()))

        lines = []
        lines.append(header)

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

        with open(output_param_file, "w+") as f:
            f.write("\n".join(lines))

    def write_univariate_sensitivity_from_json(self, json_file, output_param_file):
        """
        Read JSON file of parameter values to vary in a univariate sensitivity analysis,
        write parameter set to file
        """

        # Read JSON file of parameters
        with open(json_file) as f:
            json_string = f.read()
        json_dict = json.loads(json_string)

        # Pull out the list of parameters
        parameter_dict = json_dict["parameters"]

        # Check that n_replicates and rng_seed aren't both specified
        if ("n_replicates" in json_dict) & ("rng_seed" in parameter_dict):
            print("Both n_replicates and rng_seed are specified - only specify one")
            sys.exit()

        # Save central parameters
        central_params = copy.copy(self.params)

        lines = []
        # Adjust single parameter values
        for param, value_list in parameter_dict.items():

            if not isinstance(value_list, list):
                value_list = [value_list]

            for value in value_list:
                # Set the parameter value
                self.set_param(param, value)

                # Save the line
                line = ", ".join(list(self.params.values()))
                lines.append(line)

                # Reset the parameter value
                self.set_param(param, central_params[param])

        # Write lines to file
        header = ",".join(list(self.params.keys()))
        param_lines = "\n".join(lines)

        with open(output_param_file, "w+") as f:
            f.write(header + "\n" + param_lines)

    @property
    def NPARAMS(self):
        "Number of parameters"
        return self._NPARAMS


if __name__ == "__main__":
    """
    Usage:
    python parameters.py <input_parameter_file> <json_file> <output_parameter_file>

    Arguments
    ----------
    input_parameter_file : str
        path to input parameter file with "baseline" parameter values

    json_file : str
        path to JSON file of parameters to adjust (or vary).  JSON file has a
        key "parameters" within which there are key/value pairs of parameter
        names/values.  JSON file has an optional key of "n_replicates" with
        int value which will generate 'n_replicates' replicates for each set
        of parameter values.

    output_parameter_file : str
        path of where to save output parameter file after adjustments in JSON
        file are made


    Returns
    -------
    Writes a new parameter file to disk at location <output_parameter_file>.

    """

    input_parameter_file = sys.argv[1]
    json_file = sys.argv[2]
    output_parameter_file = sys.argv[3]

    # Read input parameter file and instantiate object
    p = ParameterSet(input_parameter_file)

    # Write new parameter file
    p.write_varying_params_from_json(json_file, output_parameter_file)
