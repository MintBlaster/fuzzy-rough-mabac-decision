def get_score(row, t=8):
    """
    Compute the final score for a single row based on theta, mu, and nu values.

    Args:
        row (pandas.Series): A row from the DataFrame containing required columns.
        t (int, optional): Scale parameter. Defaults to 8.

    Returns:
        float: The calculated final score.
    """

    # Directly use column names
    theta_lower = row["theta_lower"]
    mu_lower = row["mu_lower"]
    nu_lower = row["nu_lower"]
    theta_upper = row["theta_upper"]
    mu_upper = row["mu_upper"]
    nu_upper = row["nu_upper"]

    # Calculate the final score
    inner_bracket = mu_lower + mu_upper - nu_lower - nu_upper
    nominator = 1 + (0.5 * inner_bracket)
    fraction = nominator / 2
    final_score = fraction * (theta_lower + theta_upper) / (2 * t)

    return final_score


def add_final_scores(aggregated_df, t=8):
    """
    Given an aggregated DataFrame (experts or criteria),
    compute final score for each row and add it as a new column.

    Args:
        aggregated_df (pandas.DataFrame): DataFrame with theta_lower, mu_lower, nu_lower, etc.
        t (int, optional): Scale parameter. Defaults to 8.

    Returns:
        pandas.DataFrame: Same as input but with an extra 'score' column.
    """

    # Apply row-wise
    scores = aggregated_df.apply(lambda row: get_score(row, t=t), axis=1)

    # Add new column
    aggregated_df = aggregated_df.copy()
    aggregated_df["score"] = scores

    return aggregated_df
