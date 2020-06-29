#!/usr/bin/env python3
"""
Convert a parameter set with two columns (param name \t value) to a CSV with header as parameter
name and first line of parameter values.  Create markdown tables of parameters for documentation
folder.  This script ensures that the baseline parameters and parameter tables in the documentation
are consistent.  

Created: June 2020
Author: p-robot
"""

import sys
import pandas as pd, numpy as np
from os.path import join

def create_markdown_from_df(df, title = ""):
    """
    Create text string of markdown table from pandas dataframe of OpenABM-Covid19 parameters
    Used in automated creation of markdown tables for model documentation.  
    
    Arguments
    ---------
    df : pandas.DataFrame
        Dataframe of OpenABM-Covid19 parameters (transposed), with rows as parameters
        and with the following columns: 
        - Name (str, parameter name)
        - Value (float/int, default value)
        - Symbol (str, symbol for this parameter used in markdown documentation)
        - Description (str, description of the parameter)
        - Source (str, source for the default value)
    title : str
        Title string for header of the markdown file
    
    Returns
    -------
    str of markdown table of the form:
    
    | Name | .... | Source |
    | ---- | ---- |  ----  |
    |  11  | .... |   1N   |
    |  21  | .... |   2N   |
    | .... | .... |  ....  |
    |  M1  | .... |   MN   |
    """
    NCOLS = df.shape[1]
    
    title_text = ["# " + title]
    header = ["| " + " | ".join(df.columns) + " | "]
    hline = ["| " + "".join([" ---- |" for i in range(NCOLS)])]
    
    table_body = list()
    for i, row in df.iterrows():
        
        table_row = "| `{}` | {} | {} | {} | {} |".format(
            row.Name, 
            str(row.Value), 
            row.Symbol, 
            row.Description, 
            row.Source)
        
        table_body.append(table_row)
    
    output = title_text + header + hline + table_body
    
    return("\n".join(output))


if __name__ == "__main__":
    
    # Parse command line arguments
    if len(sys.argv) > 1:
        long_parameter_file = sys.argv[1]
    else:
        long_parameter_file = join("tests", "data", "baseline_parameters_transpose.csv")

    if len(sys.argv) > 2:
        wide_parameter_file = sys.argv[2]
    else:
        wide_parameter_file = join("tests", "data", "baseline_parameters.csv")
    
    # Read "long/transpose" format of parameter file
    df = pd.read_csv(long_parameter_file, dtype = str)
    
    # Write parameters in a form readable by the model
    df[["Name", "Value"]].set_index("Name").transpose().to_csv(wide_parameter_file, index = False)
    
    # Generate markdown tables for each parameter type (first strip on white space)
    parameter_types = df["Parameter type"].dropna().str.strip().unique()
    
    for t in parameter_types:
        df_type = df.loc[df["Parameter type"] == t]
        df_type = df_type.replace(np.nan, "-").drop(columns = ["Parameter type"])
        markdown_table = create_markdown_from_df(df_type, title = "Table: " + t)
        
        markdown_file = t.lower().replace(" ", "_") + '.md'
        with open(join("documentation", "parameters", markdown_file), 'w') as f:
            f.write(markdown_table)
    
    # Generate table for all parameters
    df_all = df.replace(np.nan, "-").drop(columns = ["Parameter type"])
    markdown_table = create_markdown_from_df(df_all, title = "Table: Parameter dictionary")
    
    with open(join("documentation", "parameters", "parameter_dictionary.md"), 'w') as f:
        f.write(markdown_table)
