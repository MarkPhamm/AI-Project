import pandas as pd
pd.set_option('display.max_columns', None)
import numpy as np
#-----------------------------

def filter_columns(df, include_strings):
    """
    Filter DataFrame columns containing the specified string in include_strings.

    Args:
    df (pandas.DataFrame): Input DataFrame.
    include_strings (str): String to be included in the column names.

    Returns:
    pandas.DataFrame: Filtered DataFrame.
    """
    filtered_columns = []
    if isinstance(include_strings, str):
        include_strings = [include_strings]  # Convert single string to a list
    for col in df.columns:
        if any(include_string in col.lower() for include_string in include_strings):
            filtered_columns.append(col)
    return df[filtered_columns]


def rename_columns(df):
   # convert form camel case to snake case
   df.columns = (df.columns
                  .str.replace('(?<=[a-z])(?=[A-Z])', '_', regex=True)
                  .str.lower()
               )
   # convert " " to _
   df.columns = df.columns.str.replace(' ', '_')
   return df

def drop_negative_values(df):
    """
    Drop rows containing any values less than 0 in any column.

    Parameters:
        df (pandas.DataFrame): DataFrame containing the data.

    Returns:
        pandas.DataFrame: DataFrame with rows containing negative values dropped.
    """
    # Create a mask for rows with any negative values
    mask = (df < 0).any(axis=1)
    
    # Use the mask to filter out rows containing negative values
    df_filtered = df[~mask]
    
    return df_filtered