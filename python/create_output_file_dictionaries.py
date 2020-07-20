import sys
from os.path import join
import pandas as pd, numpy as np

def create_markdown_from_df(df, title = "", include_file_type = False):
    """
    Create text string of markdown table from pandas dataframe of OpenABM-Covid19 parameters
    Used in automated creation of markdown tables for model documentation.

    Arguments
    ---------
    df : pandas.DataFrame
        Dataframe of OpenABM-Covid19 parameters (transposed), with rows as columns
        and with the following columns:
        - Name (str, parameter name)
        - Description (str, description of the parameter)
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
        
        if include_file_type:
            table_row = "| `{}` | {} | {} |".format(
                row["Column name"],
                row.Description,
                row["File type"])
        else:
            table_row = "| `{}` | {} |".format(
                row["Column name"],
                row.Description)
        
        table_body.append(table_row)

    output = title_text + header + hline + table_body

    return("\n".join(output))


if __name__ == "__main__":

    # Parse command line arguments
    if len(sys.argv) > 1:
        output_file_csv = sys.argv[1]
    else:
        output_file_csv = join("documentation", "output_files", "output_file_dictionary.csv")

    if len(sys.argv) > 2:
        wide_parameter_file = sys.argv[2]
    else:
        wide_parameter_file = join("tests", "data", "baseline_parameters.csv")

    df = pd.read_csv(output_file_csv, dtype = str)

    # Generate markdown tables for each output file type (first strip on white space)
    parameter_types = df["File type"].dropna().str.strip().unique()

    for t in parameter_types:
        df_type = df.loc[df["File type"] == t]
        df_type = df_type.replace(np.nan, "-").drop(columns = ["File type"])
        markdown_table = create_markdown_from_df(df_type, title = "Table: " + t)

        markdown_file = t.lower().replace(" ", "_") + '.md'
        with open(join("documentation", "output_files", markdown_file), 'w') as f:
            f.write(markdown_table)

    # Generate table for all parameters
    df_all = df.replace(np.nan, "-")
    markdown_table = create_markdown_from_df(df_all, title = "Table: Output file dictionary", include_file_type = True)

    with open(join("documentation", "output_files", "output_file_dictionary.md"), 'w') as f:
        f.write(markdown_table)

