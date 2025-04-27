import math
import numpy as np
import pandas as pd


def aggregate_linguistic(theta_values, t_value=8, zeta_value=2, weights=None):
    """
    Aggregates linguistic values

    Args:
        theta_values: Array of linguistic values to aggregate
        t_value: Maximum linguistic term value (default=8)
        zeta_value: Operational parameter affecting aggregation behavior (default=2)
        weights: Optional weights for each value (default=equal weights)

    Returns:
        Aggregated linguistic value
    """

    theta_values = np.asarray(theta_values)

    # Use equal weights if none provided
    if weights is None:
        weights = np.ones(len(theta_values)) / len(theta_values)
    else:
        weights = np.asarray(weights)

    theta_ratio = theta_values / t_value

    complement_ratio = 1 - theta_ratio

    denominator = 1 + zeta_value * complement_ratio

    fraction = (1 + zeta_value) / denominator

    weighted_product = np.prod(fraction ** weights)

    middle_term = ((1 + zeta_value) / weighted_product) - 1

    aggregated_linguistic = t_value * (1 - (1 / zeta_value) * middle_term)

    return aggregated_linguistic


def aggregate_membership(mu_values, zeta_value=2, weights=None):
    """Aggregates membership values."""
    mu_values = np.asarray(mu_values)

    if weights is None:
        weights = np.ones(len(mu_values)) / len(mu_values)
    else:
        weights = np.asarray(weights)

    mu_squared = mu_values ** 2
    inner_bracket = 1 - mu_squared
    denominator = 1 + zeta_value * inner_bracket
    fraction = (1 + zeta_value) / denominator
    inner_loop_product = np.prod(fraction ** weights)

    middle_part = ((1 + zeta_value) / inner_loop_product) - 1
    aggregated_membership = 1 - (1 / zeta_value) * middle_part
    aggregated_membership = math.sqrt(aggregated_membership)

    return aggregated_membership


def aggregate_non_membership(nu_values, zeta_value=2, weights=None):
    """Aggregates non-membership values."""
    nu_values = np.asarray(nu_values)

    if weights is None:
        weights = np.ones(len(nu_values)) / len(nu_values)
    else:
        weights = np.asarray(weights)

    numerator = 1 + zeta_value * nu_values ** 2
    denominator = 1 + zeta_value
    fraction = numerator / denominator
    inner_loop_product = np.prod(fraction ** weights)

    outer_numerator = (1 + zeta_value) * inner_loop_product - 1
    aggregated_non_membership = math.sqrt(outer_numerator / zeta_value)

    return aggregated_non_membership


def aggregate_terms(data, weights=None, t=8):
    """Aggregates terms for both lower and upper approximations."""
    if weights is None:
        weights = np.ones(len(data)) / len(data)
    else:
        weights = np.asarray(weights)

    theta_lower = data['theta_lower'].values
    mu_lower = data['mu_lower'].values
    nu_lower = data['nu_lower'].values
    theta_upper = data['theta_upper'].values
    mu_upper = data['mu_upper'].values
    nu_upper = data['nu_upper'].values

    result_dict = {
        'theta_lower': aggregate_linguistic(theta_lower, t_value=t, weights=weights),
        'mu_lower': aggregate_membership(mu_lower, weights=weights),
        'nu_lower': aggregate_non_membership(nu_lower, weights=weights),
        'theta_upper': aggregate_linguistic(theta_upper, t_value=t, weights=weights),
        'mu_upper': aggregate_membership(mu_upper, weights=weights),
        'nu_upper': aggregate_non_membership(nu_upper, weights=weights)
    }

    return result_dict

def get_expert_aggregation(expert_matrix, linguistic_df, expert_weights=None):
    """
    Process a matrix of expert evaluations and compute aggregated results for each expert.

    Args:
        expert_matrix (DataFrame or array-like): Matrix where each row corresponds to one expert
                                                and each column contains linguistic terms
        linguistic_df (DataFrame): DataFrame mapping linguistic terms to their values
        expert_weights (list, optional): Not used anymore here, aggregation is per expert individually.

    Returns:
        pandas.DataFrame: Aggregated values for each expert (row per expert)
    """
    # Convert to numpy array if DataFrame
    if isinstance(expert_matrix, pd.DataFrame):
        matrix_array = expert_matrix.values
    else:
        matrix_array = np.asarray(expert_matrix)

    n_experts = matrix_array.shape[0]

    expert_aggregations = []

    # Process each expert's evaluation individually
    for i, expert_terms in enumerate(matrix_array):
        # Filter the DataFrame to get values for the expert's terms
        filtered_df = linguistic_df[linguistic_df["name"].isin(expert_terms)]

        # Aggregate terms for this expert (equal weights inside expert)
        result = aggregate_terms(filtered_df)

        expert_aggregations.append(result)

    # Convert list of dicts to DataFrame
    final_df = pd.DataFrame(expert_aggregations)

    # Add Expert IDs if you want
    final_df.index = [f"Expert_{i+1}" for i in range(n_experts)]

    return final_df


def get_criterion_aggregation(matrix, t_value=8, zeta_value=2, lower_scale=0.7, upper_scale=1.0):
    """
    Process a comparison matrix and calculate aggregated values for each criterion.

    Args:
        matrix (pandas.DataFrame or numpy.ndarray): The comparison matrix with values 0, 0.5, and 1
        t_value (int): Scale parameter for linguistic values, default 8
        zeta_value (float): Operational parameter, default 2
        lower_scale (float): Scale for lower approximation transformation, default 0.7
        upper_scale (float): Scale for upper approximation transformation, default 1.0

    Returns:
        pandas.DataFrame: DataFrame with aggregated values for each criterion in row format
    """
    # Convert to numpy array if it's a DataFrame
    if isinstance(matrix, pd.DataFrame):
        matrix_array = matrix.to_numpy()
    else:
        matrix_array = np.asarray(matrix)

    n_criteria = matrix_array.shape[0]  # Number of criteria

    # Initialize arrays to store results
    results = {
        'theta_lower': np.zeros(n_criteria),
        'mu_lower': np.zeros(n_criteria),
        'nu_lower': np.zeros(n_criteria),
        'theta_upper': np.zeros(n_criteria),
        'mu_upper': np.zeros(n_criteria),
        'nu_upper': np.zeros(n_criteria)
    }

    # For each criterion
    for i in range(n_criteria):
        row_values = matrix_array[i, :]

        # Initialize DataFrames to store transformed values
        theta_lower_values = []
        mu_lower_values = []
        nu_lower_values = []
        theta_upper_values = []
        mu_upper_values = []
        nu_upper_values = []

        # Transform each comparison value according to the linguistic term mapping
        for delta in row_values:
            # Lower approximation values
            if delta == 1:
                # If δᵢⱼ = 1 then [(ℓ₈, (0.84, 0.1)), (ℓ₅, (0.95, 0.05))]
                theta_lower_values.append(8 * lower_scale)  # ℓ₈
                mu_lower_values.append(0.84 * lower_scale)  # 0.84
                nu_lower_values.append(0.1 * lower_scale)  # 0.1
            elif delta == 0.5:
                # If δᵢⱼ = 0.5 then [(ℓ₄, (0.5, 0.4)), (ℓ₅, (0.6, 0.3))]
                theta_lower_values.append(4 * lower_scale)  # ℓ₄
                mu_lower_values.append(0.5 * lower_scale)  # 0.5
                nu_lower_values.append(0.4 * lower_scale)  # 0.4
            elif delta == 0:
                # If δᵢⱼ = 0 then [(ℓ₀.₀₁, (0.01, 0.9)), (ℓ₁, (0.1, 0.8))]
                theta_lower_values.append(0.01 * lower_scale)  # ℓ₀.₀₁
                mu_lower_values.append(0.01 * lower_scale)  # 0.01
                nu_lower_values.append(0.9 * lower_scale)  # 0.9
            else:
                # For any other values (fallback)
                theta_lower_values.append(delta * t_value * lower_scale)
                mu_lower_values.append(delta * lower_scale)
                nu_lower_values.append((1 - delta) * lower_scale)

            # Upper approximation values
            if delta == 1:
                # If δᵢⱼ = 1 then [(ℓ₈, (0.84, 0.1)), (ℓ₅, (0.95, 0.05))]
                theta_upper_values.append(5 * upper_scale)  # ℓ₅
                mu_upper_values.append(0.95 * upper_scale)  # 0.95
                nu_upper_values.append(0.05 * upper_scale)  # 0.05
            elif delta == 0.5:
                # If δᵢⱼ = 0.5 then [(ℓ₄, (0.5, 0.4)), (ℓ₅, (0.6, 0.3))]
                theta_upper_values.append(5 * upper_scale)  # ℓ₅
                mu_upper_values.append(0.6 * upper_scale)  # 0.6
                nu_upper_values.append(0.3 * upper_scale)  # 0.3
            elif delta == 0:
                # If δᵢⱼ = 0 then [(ℓ₀.₀₁, (0.01, 0.9)), (ℓ₁, (0.1, 0.8))]
                theta_upper_values.append(1 * upper_scale)  # ℓ₁
                mu_upper_values.append(0.1 * upper_scale)  # 0.1
                nu_upper_values.append(0.8 * upper_scale)  # 0.8
            else:
                # For any other values (fallback)
                theta_upper_values.append(delta * t_value * upper_scale)
                mu_upper_values.append(delta * upper_scale)
                nu_upper_values.append((1 - delta) * upper_scale)

        data = pd.DataFrame({
            'theta_lower': theta_lower_values,
            'mu_lower': mu_lower_values,
            'nu_lower': nu_lower_values,
            'theta_upper': theta_upper_values,
            'mu_upper': mu_upper_values,
            'nu_upper': nu_upper_values
        })

        # Calculate weights based on the matrix values (normalized row)
        weights = row_values / np.sum(row_values) if np.sum(row_values) > 0 else np.ones(n_criteria) / n_criteria

        # Aggregate for this criterion
        aggregated = aggregate_terms(data, weights=weights, t=t_value)

        # Store results
        for key in aggregated:
            results[key][i] = aggregated[key]

    # Convert results to DataFrame
    results_df = pd.DataFrame(results)

    # Add criterion identifier if original input was DataFrame
    if isinstance(matrix, pd.DataFrame):
        results_df.index = matrix.index

    return results_df
