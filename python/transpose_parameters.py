#!/usr/bin/env python3
"""
Convert a parameter set with two columns (param name \t value) to a CSV with header as parameter
name and first line of parameter values.  
"""

import sys

if len(sys.argv) > 1:
    long_parameter_file = sys.argv[1]
else:
    long_parameter_file = "./tests/data/baseline_parameters_transpose.csv"

if len(sys.argv) > 2:
    wide_parameter_file = sys.argv[2]
else:
    wide_parameter_file = "./tests/data/baseline_parameters.csv"

if __name__ == "__main__":
    with open(long_parameter_file) as f:
        data = f.readlines()

    # Discard the header
    data = data[1:]

    # Split lines
    parameters = [row.strip().split() for row in data]

    # Separate parameter names and values
    parameter_names = [row[0].strip() for row in parameters]
    parameter_values = [row[1].strip() for row in parameters]

    # Create header and first parameter line
    header = ",".join(parameter_names)
    line = ",".join(parameter_values)

    # Write to file
    with open(wide_parameter_file, "w+") as f:
        f.write(header + "\n" + line + "\n")
