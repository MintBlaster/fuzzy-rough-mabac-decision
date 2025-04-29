import pandas as pd
import numpy as np


def replace_with_tuple(value, linguistic_dict):
    """
    Replace linguistic values with their corresponding tuples from the dictionary.

    Args:
        value: The linguistic value to replace
        linguistic_dict: Dictionary mapping linguistic values to tuples

    Returns:
        tuple or original value: The corresponding tuple or original value if not found
    """
    if value in linguistic_dict:
        return linguistic_dict[value]
    return value


def process_matrix(df, linguistic_dict):
    """
    Process a matrix by replacing linguistic values with tuples.

    Args:
        df (pandas.DataFrame): DataFrame with linguistic values
        linguistic_dict: Dictionary mapping linguistic values to tuples

    Returns:
        pandas.DataFrame: DataFrame with values replaced by tuples
    """
    return df.map(lambda x: replace_with_tuple(x, linguistic_dict))


def create_linguistic_dict(linguistic_df):
    """
    Create a dictionary mapping linguistic terms to their tuple representations.

    Args:
        linguistic_df (pandas.DataFrame): DataFrame with linguistic terms and their values

    Returns:
        dict: Dictionary mapping terms to tuples
    """
    return dict(zip(linguistic_df['name'],
                    zip(linguistic_df['theta_lower'],
                        linguistic_df['mu_lower'],
                        linguistic_df['nu_lower'],
                        linguistic_df['theta_upper'],
                        linguistic_df['mu_upper'],
                        linguistic_df['nu_upper'])))


def normalization_expert_matrix(df, criteria_types, t_value=8):
    """
    Process each tuple in the DataFrame based on criteria types.
    For benefit criteria: keep values as is
    For cost criteria: subtract theta from t_value and swap mu/nu values

    Args:
        df (pd.DataFrame): DataFrame containing tuples
        criteria_types (dict): Dictionary mapping criteria names to 'cost' or 'benefit'
        t_value (int): The t value for theta transformation (default=8)

    Returns:
        pd.DataFrame: Processed DataFrame
    """
    # Create a copy to avoid modifying the original
    result_df = df.copy()

    # Iterate through each cell in the DataFrame
    for col in result_df.columns:
        criteria_type = criteria_types.get(col, 'benefit')  # Default to 'benefit'

        for idx in result_df.index:
            value = result_df.at[idx, col]

            if isinstance(value, tuple):
                theta_lower, mu_lower, nu_lower, theta_upper, mu_upper, nu_upper = value

                if criteria_type == 'benefit':
                    # For benefit criteria, keep as is
                    new_tuple = value
                else:  # cost criteria
                    # For cost criteria:
                    # 1. Subtract theta values from t_value
                    # 2. Swap mu and nu values
                    new_theta_lower = t_value - theta_lower
                    new_theta_upper = t_value - theta_upper
                    new_mu_lower = nu_lower
                    new_nu_lower = mu_lower
                    new_mu_upper = nu_upper
                    new_nu_upper = mu_upper

                    new_tuple = (
                        new_theta_lower,
                        new_mu_lower,
                        new_nu_lower,
                        new_theta_upper,
                        new_mu_upper,
                        new_nu_upper
                    )

                # Store the new tuple back in the DataFrame
                result_df.at[idx, col] = new_tuple

    return result_df