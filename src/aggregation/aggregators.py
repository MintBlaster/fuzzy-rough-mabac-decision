import numpy as np
import pandas as pd
from .core import aggregate_terms


def get_expert_aggregation(expert_matrix, linguistic_df, zeta=2):
    """
    Process a matrix of expert evaluations and compute aggregated results for each expert.

    Args:
        expert_matrix (DataFrame or array-like): Matrix where each row corresponds to one expert
                                                and each column contains linguistic terms
        linguistic_df (DataFrame): DataFrame mapping linguistic terms to their values
        zeta (float): Operational parameter for aggregation, default 2

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
        result = aggregate_terms(filtered_df, zeta=zeta)

        expert_aggregations.append(result)

    # Convert list of dicts to DataFrame
    final_df = pd.DataFrame(expert_aggregations)

    final_df.index = [f"Expert_{i + 1}" for i in range(n_experts)]

    return final_df


def get_criterion_aggregation(matrix, t_value=8, zeta=2, lower_scale=0.7, upper_scale=1.0):
    """
    Process a comparison matrix and calculate aggregated values for each criterion.

    Args:
        matrix (pandas.DataFrame or numpy.ndarray): The comparison matrix with values 0, 0.5, and 1
        t_value (int): Scale parameter for linguistic values, default 8
        zeta (float): Operational parameter, default 2
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

        # Aggregate for this criterion - FIXED: Added zeta parameter
        aggregated = aggregate_terms(data, weights=weights, zeta=zeta, t=t_value)

        # Store results
        for key in aggregated:
            results[key][i] = aggregated[key]

    # Convert results to DataFrame
    results_df = pd.DataFrame(results)

    # Add criterion identifier if original input was DataFrame
    if isinstance(matrix, pd.DataFrame):
        results_df.index = matrix.index

    return results_df


def aggregate_expert_matrices(expert_matrices, weights_df=None, zeta=2, t_value=8):
    """
    Aggregate multiple expert matrices using weights from a DataFrame or using equal weights.

    Args:
        expert_matrices: List of DataFrames containing expert evaluations
        weights_df: DataFrame containing expert weights with a 'normalized' column,
                   or None for equal weights (default: None)
        zeta: Operational parameter for aggregation, default 2
        t_value: Scale parameter for linguistic values, default 8

    Returns:
        pandas.DataFrame: Aggregated matrix
    """
    # Handle weights - either from DataFrame or create equal weights
    if weights_df is not None:
        weights = weights_df['normalized'].values
    else:
        # If no weights provided, use equal weights for all experts
        weights = np.ones(len(expert_matrices)) / len(expert_matrices)

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

            aggregated_values = aggregate_terms(position_data, weights=valid_weights, zeta=zeta, t=t_value)

            aggregated_matrix.at[row_idx, col] = (
                aggregated_values['theta_lower'],
                aggregated_values['mu_lower'],
                aggregated_values['nu_lower'],
                aggregated_values['theta_upper'],
                aggregated_values['mu_upper'],
                aggregated_values['nu_upper']
            )

    return aggregated_matrix