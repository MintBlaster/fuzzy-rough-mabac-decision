import numpy as np
import pandas as pd


def get_score(values, t=8):
    """
    Compute the final score based on a tuple of theta, mu, and nu values.

    Args:
        values (tuple or list): (theta_lower, mu_lower, nu_lower, theta_upper, mu_upper, nu_upper)
        t (int, optional): Scale parameter. Defaults to 8.

    Returns:
        float: The calculated final score.
    """
    theta_lower, mu_lower, nu_lower, theta_upper, mu_upper, nu_upper = values

    inner_bracket = mu_lower + mu_upper - nu_lower - nu_upper
    nominator = 1 + (0.5 * inner_bracket)
    fraction = nominator / 2
    final_score = fraction * (theta_lower + theta_upper) / (2 * t)

    return final_score




def add_final_scores(aggregated_df, t=8):
    """
    Compute final score for each row and add it as a new column.

    Args:
        aggregated_df (pandas.DataFrame): Must contain required columns.
        t (int, optional): Scale parameter. Defaults to 8.

    Returns:
        pandas.DataFrame: Input DataFrame with added 'score' column.
    """
    # Define the exact order of values to be passed as tuple
    values = aggregated_df[[
        "theta_lower", "mu_lower", "nu_lower",
        "theta_upper", "mu_upper", "nu_upper"
    ]].values

    scores = [get_score(row, t=t) for row in values]

    aggregated_df = aggregated_df.copy()
    aggregated_df["score"] = scores

    return aggregated_df



def add_normalized_scores(df):
    """
    Add a normalized score column to the DataFrame.

    Args:
        df (pandas.DataFrame): DataFrame with 'score' column

    Returns:
        pandas.DataFrame: Same as input but with an extra 'normalized' column
    """
    score_sum = df['score'].sum()
    if score_sum > 0:
        df['normalized'] = df['score'] / score_sum
    else:
        df['normalized'] = 0
    return df