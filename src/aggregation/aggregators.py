import math
import numpy as np
import pandas as pd


def aggregate_linguistic(linguistic_inputs, max_linguistic_term=8, zeta=2, weights=None):
    """
    Aggregates linguistic values using a weighted power mean approach.

    Formula:
    L = t * (1 - (1/ζ) * ((1+ζ)/∏(i=1 to n)((1+ζ)/(1+ζ*(1-θᵢ/t)))^ωᵢ - 1))

    Args:
        linguistic_inputs: Array of linguistic values to aggregate
        max_linguistic_term: Maximum linguistic term value (default=8)
        zeta: Operational parameter affecting aggregation behavior (default=2)
        weights: Optional weights for each value (default=equal weights)

    Returns:
        Aggregated linguistic value
    """
    linguistic_inputs = np.asarray(linguistic_inputs)

    # Use equal weights if none provided
    if weights is None:
        weights = np.ones(len(linguistic_inputs)) / len(linguistic_inputs)
    else:
        weights = np.asarray(weights)

    # Calculate ratio of each input to maximum term
    normalized_inputs = linguistic_inputs / max_linguistic_term

    # Calculate complement of normalized inputs (1 - θᵢ/t)
    input_complements = 1 - normalized_inputs

    # Calculate denominator term (1 + ζ*(1-θᵢ/t))
    modified_denominators = 1 + zeta * input_complements

    # Calculate base expressions for product ((1+ζ)/(1+ζ*(1-θᵢ/t)))
    base_expressions = (1 + zeta) / modified_denominators

    # Calculate weighted product ∏(i=1 to n)((1+ζ)/(1+ζ*(1-θᵢ/t)))^ωᵢ
    weighted_product = np.prod(base_expressions ** weights)

    # Calculate middle term ((1+ζ)/weighted_product - 1)
    compensation_factor = ((1 + zeta) / weighted_product) - 1

    # Calculate final result: t * (1 - (1/ζ) * compensation_factor)
    aggregated_result = max_linguistic_term * (1 - (1 / zeta) * compensation_factor)

    return aggregated_result


def aggregate_membership(membership_degrees, zeta=2, weights=None):
    """
    Aggregates membership values using a weighted power mean approach.

    Formula:
    μ = √(1 - (1/ζ) * ((1+ζ)/∏(i=1 to n)((1+ζ)/(1+ζ*(1-μᵢ²)))^ωᵢ - 1))

    Args:
        membership_degrees: Array of membership values to aggregate
        zeta: Operational parameter affecting aggregation behavior (default=2)
        weights: Optional weights for each value (default=equal weights)

    Returns:
        Aggregated membership value
    """
    membership_degrees = np.asarray(membership_degrees)

    if weights is None:
        weights = np.ones(len(membership_degrees)) / len(membership_degrees)
    else:
        weights = np.asarray(weights)

    # Calculate squared membership values (μᵢ²)
    squared_memberships = membership_degrees ** 2

    # Calculate complement term (1 - μᵢ²)
    membership_complements = 1 - squared_memberships

    # Calculate denominator term (1 + ζ*(1-μᵢ²))
    modified_denominators = 1 + zeta * membership_complements

    # Calculate base expressions for product ((1+ζ)/(1+ζ*(1-μᵢ²)))
    base_expressions = (1 + zeta) / modified_denominators

    # Calculate weighted product ∏(i=1 to n)((1+ζ)/(1+ζ*(1-μᵢ²)))^ωᵢ
    weighted_product = np.prod(base_expressions ** weights)

    # Calculate compensation factor ((1+ζ)/weighted_product - 1)
    compensation_factor = ((1 + zeta) / weighted_product) - 1

    # Calculate final result without square root: 1 - (1/ζ) * compensation_factor
    inner_result = 1 - (1 / zeta) * compensation_factor

    # Apply square root to get final membership value
    aggregated_membership = math.sqrt(inner_result)

    return aggregated_membership


def aggregate_non_membership(non_membership_degrees, zeta=2, weights=None):
    """
    Aggregates non-membership values using a weighted power mean approach.

    Formula:
    ν = √((1+ζ)*∏(i=1 to n)((1+ζ*νᵢ²)/(1+ζ))^ωᵢ - 1)/ζ)

    Args:
        non_membership_degrees: Array of non-membership values to aggregate
        zeta: Operational parameter affecting aggregation behavior (default=2)
        weights: Optional weights for each value (default=equal weights)

    Returns:
        Aggregated non-membership value
    """
    non_membership_degrees = np.asarray(non_membership_degrees)

    if weights is None:
        weights = np.ones(len(non_membership_degrees)) / len(non_membership_degrees)
    else:
        weights = np.asarray(weights)

    # Calculate squared non-membership values (νᵢ²)
    squared_non_memberships = non_membership_degrees ** 2

    # Calculate numerator for each term (1 + ζ*νᵢ²)
    weighted_numerators = 1 + zeta * squared_non_memberships

    # Calculate denominator constant (1 + ζ)
    common_denominator = 1 + zeta

    # Calculate base expression for product ((1+ζ*νᵢ²)/(1+ζ))
    base_expressions = weighted_numerators / common_denominator

    # Calculate weighted product ∏(i=1 to n)((1+ζ*νᵢ²)/(1+ζ))^ωᵢ
    weighted_product = np.prod(base_expressions ** weights)

    # Calculate numerator for final expression ((1+ζ)*weighted_product - 1)
    final_numerator = (1 + zeta) * weighted_product - 1

    # Calculate final result without square root: final_numerator / ζ
    inner_result = final_numerator / zeta

    # Apply square root to get final non-membership value
    aggregated_non_membership = math.sqrt(inner_result)

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

def get_expert_aggregation(expert_matrix, linguistic_df):
    """
    Process a matrix of expert evaluations and compute aggregated results for each expert.

    Args:
        expert_matrix (DataFrame or array-like): Matrix where each row corresponds to one expert
                                                and each column contains linguistic terms
        linguistic_df (DataFrame): DataFrame mapping linguistic terms to their values

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


def aggregate_expert_matrices(expert_matrices, weights_file="../results/expert_aggregation.csv", t_value=8):

    expert_weights_df = pd.read_csv(weights_file)
    weights = expert_weights_df['normalized'].values

    if len(weights) != len(expert_matrices):
        raise ValueError(
            f"Number of weights ({len(weights)}) doesn't match number of matrices ({len(expert_matrices)})")

    # Initialize empty aggregated matrix
    aggregated_matrix = pd.DataFrame(
        index=expert_matrices[0].index,
        columns=expert_matrices[0].columns
    )

    # Set the "Alternative" column properly
    if 'Alternative' in expert_matrices[0].columns:
        aggregated_matrix['Alternative'] = expert_matrices[0]['Alternative']

    for row_idx in aggregated_matrix.index:
        for col in aggregated_matrix.columns:
            if col == 'Alternative':
                continue

            valid_expert_indices = []
            valid_theta_lower = []
            valid_mu_lower = []
            valid_nu_lower = []
            valid_theta_upper = []
            valid_mu_upper = []
            valid_nu_upper = []

            for i, matrix in enumerate(expert_matrices):
                cell_value = matrix.at[row_idx, col]
                if isinstance(cell_value, tuple) and len(cell_value) == 6:
                    valid_expert_indices.append(i)
                    valid_theta_lower.append(cell_value[0])
                    valid_mu_lower.append(cell_value[1])
                    valid_nu_lower.append(cell_value[2])
                    valid_theta_upper.append(cell_value[3])
                    valid_mu_upper.append(cell_value[4])
                    valid_nu_upper.append(cell_value[5])

            if len(valid_expert_indices) == 0:
                continue

            valid_weights = weights[valid_expert_indices]
            valid_weights = valid_weights / np.sum(valid_weights)

            position_data = pd.DataFrame({
                'theta_lower': valid_theta_lower,
                'mu_lower': valid_mu_lower,
                'nu_lower': valid_nu_lower,
                'theta_upper': valid_theta_upper,
                'mu_upper': valid_mu_upper,
                'nu_upper': valid_nu_upper
            })

            aggregated_values = aggregate_terms(position_data, weights=valid_weights, t=t_value)

            aggregated_matrix.at[row_idx, col] = (
                aggregated_values['theta_lower'],
                aggregated_values['mu_lower'],
                aggregated_values['nu_lower'],
                aggregated_values['theta_upper'],
                aggregated_values['mu_upper'],
                aggregated_values['nu_upper']
            )

    return aggregated_matrix
