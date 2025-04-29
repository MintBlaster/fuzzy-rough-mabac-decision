import math
import numpy as np
import pandas as pd
from src.score import get_score


###### EXPERT MATRIX WEIGHT PRODUCT ######

def expert_matrix_weight_product(lower_tuple, upper_tuple, alpha, zeta=2, t_value=8):
    """
    Calculate weighted result for fuzzy sets.

    The calculation follows these formulas:

    For theta values:
    θ^α = t * ((1+ζ)/ζ) * (1 - (1 - (θ/t) * (ζ/(1+ζ)))^α)

    For membership values (μ):
    μ^α = ((1+ζ)/ζ) * (1 - (1 - μ * (ζ/(1+ζ)))^α)

    For non-membership values (ν):
    ν^α = ((1+ζ) * ((1 + ζ*ν)/(1+ζ))^α - 1)/ζ

    Where:
    - θ, μ, ν are the fuzzy values
    - α is the normalized weight
    - ζ (zeta) is an operational parameter
    - t is the maximum linguistic term value

    Args:
        lower_tuple (tuple): (theta_lower, mu_lower, nu_lower)
        upper_tuple (tuple): (theta_upper, mu_upper, nu_upper)
        alpha (float): Normalized weight value
        zeta (float): Operational parameter (default=2)
        t_value (int): Max linguistic term (default=8)

    Returns:
        tuple: (theta_result_lower, mu_result_lower, nu_result_lower,
                theta_result_upper, mu_result_upper, nu_result_upper)
    """
    # Unpack tuples
    theta_lower, mu_lower, nu_lower = lower_tuple
    theta_upper, mu_upper, nu_upper = upper_tuple

    # NumPy optimized implementation
    # Common factors
    zeta_factor = zeta / (1 + zeta)
    zeta_factor_inverse = (1 + zeta) / zeta

    # For theta values
    theta_inner_lower = 1 - (theta_lower / t_value) * zeta_factor
    theta_inner_upper = 1 - (theta_upper / t_value) * zeta_factor

    theta_result_lower = t_value * zeta_factor_inverse * (1 - np.power(theta_inner_lower, alpha))
    theta_result_upper = t_value * zeta_factor_inverse * (1 - np.power(theta_inner_upper, alpha))

    # For mu values
    mu_inner_lower = 1 - mu_lower * zeta_factor
    mu_inner_upper = 1 - mu_upper * zeta_factor

    mu_result_lower = zeta_factor_inverse * (1 - np.power(mu_inner_lower, alpha))
    mu_result_upper = zeta_factor_inverse * (1 - np.power(mu_inner_upper, alpha))

    # For nu values
    nu_inner_lower = (1 + zeta * nu_lower) / (1 + zeta)
    nu_inner_upper = (1 + zeta * nu_upper) / (1 + zeta)

    nu_result_lower = ((1 + zeta) * np.power(nu_inner_lower, alpha) - 1) / zeta
    nu_result_upper = ((1 + zeta) * np.power(nu_inner_upper, alpha) - 1) / zeta

    return (
        theta_result_lower, mu_result_lower, nu_result_lower,
        theta_result_upper, mu_result_upper, nu_result_upper
    )


def calculate_weighted_result(aggregated_expert_matrix, criteria_aggregated, zeta=2, t_value=8):
    """
    Calculate weighted result by multiplying expert matrix values with criteria weights.

    Args:
        aggregated_expert_matrix (pandas.DataFrame): Aggregated expert matrix
        criteria_aggregated (pandas.DataFrame): Aggregated criteria with normalized weights
        zeta (float): Parameter for formula calculation (defaults to 2)
        t_value (int): Max linguistic term (defaults to 8)

    Returns:
        pandas.DataFrame: Weighted result matrix
    """
    result_matrix = pd.DataFrame(index=aggregated_expert_matrix.index,
                                 columns=aggregated_expert_matrix.columns)

    if 'Alternative' in aggregated_expert_matrix.columns:
        result_matrix['Alternative'] = aggregated_expert_matrix['Alternative']

    # Prepare criteria weights as a NumPy array for faster access
    criteria_indices = [f"C{idx + 1}" for idx in range(len(aggregated_expert_matrix.columns) - 1
                                                       if 'Alternative' in aggregated_expert_matrix.columns
                                                       else len(aggregated_expert_matrix.columns))]

    alphas = np.array([criteria_aggregated.loc[c_idx, 'normalized']
                       if c_idx in criteria_aggregated.index else np.nan
                       for c_idx in criteria_indices])

    for col_idx, col in enumerate(aggregated_expert_matrix.columns):
        if col == 'Alternative':
            continue

        adjusted_idx = col_idx - 1 if 'Alternative' in aggregated_expert_matrix.columns else col_idx
        if adjusted_idx < len(alphas) and not np.isnan(alphas[adjusted_idx]):
            alpha = alphas[adjusted_idx]

            for i in aggregated_expert_matrix.index:
                current_value = aggregated_expert_matrix.loc[i, col]

                # Extracting tuple values
                if isinstance(current_value, tuple) and len(current_value) == 6:
                    theta_lower, mu_lower, nu_lower, theta_upper, mu_upper, nu_upper = current_value

                    # Split into lower and upper tuples
                    lower_tuple = (theta_lower, mu_lower, nu_lower)
                    upper_tuple = (theta_upper, mu_upper, nu_upper)

                    # Pass tuples to the function
                    result_matrix.at[i, col] = expert_matrix_weight_product(
                        lower_tuple, upper_tuple, alpha, zeta, t_value
                    )
                else:
                    result_matrix.at[i, col] = current_value
        else:
            criteria_idx = f"C{adjusted_idx + 1}"
            print(f"Warning: No normalization value found for {criteria_idx}")

    return result_matrix


############# EXPERT MATRIX Border Approximation Area (BAA) Calculation #############

def aggregate_column_values(column_tuples, omega=None, zeta=2, t_value=8):
    """
    Aggregate multiple fuzzy tuples into a single tuple.

    Args:
        column_tuples (list): List of tuples, each containing
                             (theta_lower, mu_lower, nu_lower, theta_upper, mu_upper, nu_upper)
        omega (list, optional): Weights for each tuple in the aggregation.
                              If None, equal weights 1/n are used.
        zeta (float): Operational parameter (default=2)
        t_value (int): Max linguistic term (default=8)

    Returns:
        tuple: Aggregated (theta_lower, mu_lower, nu_lower, theta_upper, mu_upper, nu_upper)
    """
    # If omega is not provided, use equal weights
    if omega is None:
        omega = np.ones(len(column_tuples)) / len(column_tuples)
    else:
        omega = np.array(omega)

    # Ensure omega is the right length
    if len(omega) != len(column_tuples):
        raise ValueError(f"Omega length ({len(omega)}) must match number of tuples ({len(column_tuples)})")

    # Convert tuple list to NumPy arrays for efficient processing
    column_array = np.array(column_tuples)

    # Extract components more efficiently
    theta_lower_values = column_array[:, 0]
    mu_lower_values = column_array[:, 1]
    nu_lower_values = column_array[:, 2]
    theta_upper_values = column_array[:, 3]
    mu_upper_values = column_array[:, 4]
    nu_upper_values = column_array[:, 5]

    # Call helper functions to aggregate each component
    theta_lower_agg = aggregate_theta_values(theta_lower_values, omega, zeta, t_value)
    mu_lower_agg = aggregate_mu_values(mu_lower_values, omega, zeta)
    nu_lower_agg = aggregate_nu_values(nu_lower_values, omega, zeta)
    theta_upper_agg = aggregate_theta_values(theta_upper_values, omega, zeta, t_value)
    mu_upper_agg = aggregate_mu_values(mu_upper_values, omega, zeta)
    nu_upper_agg = aggregate_nu_values(nu_upper_values, omega, zeta)

    # Return the aggregated tuple
    return (
        theta_lower_agg, mu_lower_agg, nu_lower_agg,
        theta_upper_agg, mu_upper_agg, nu_upper_agg
    )


def aggregate_theta_values(theta_values, omega, zeta, t_value):
    """
    Aggregate multiple theta values into a single value using weighted geometric mean.

    Formula:
    θ_BAA = t * ((1 + ζ) * Π[(1 + ζ * (θᵢ/t))/(1 + ζ)]^ωᵢ - 1) / ζ

    Args:
        theta_values (list or numpy.ndarray): List of theta values to aggregate
        omega (list or numpy.ndarray or optional): Weights for each value (default: None)
        zeta (float): Operational parameter
        t_value (int): Max linguistic term

    Returns:
        float: Aggregated theta value
    """
    # NumPy optimized implementation
    theta_array = np.array(theta_values)

    # Calculate the components of the geometric mean
    nominator = 1 + zeta * (theta_array / t_value)
    denominator = 1 + zeta

    # Calculate the weighted product using NumPy
    # Note: We need to take log, multiply, then exp to avoid numerical overflow
    log_terms = np.log(nominator / denominator) * omega
    log_product = np.sum(log_terms)
    loop_product = np.exp(log_product)

    # Final calculation
    result = t_value * (((1 + zeta) * loop_product - 1) / zeta)

    return result


def aggregate_mu_values(mu_values, omega, zeta):
    """
    Aggregate multiple mu values into a single value using weighted geometric mean.

    Formula:
    μ_BAA = √(((1 + ζ) * Π[(1 + ζ * μᵢ²)/(1 + ζ)]^ωᵢ - 1) / ζ)

    Args:
        mu_values (list or numpy.ndarray): List of mu values to aggregate
        omega (list or numpy.ndarray or optional): Weights for each value (default: None)
        zeta (float): Operational parameter

    Returns:
        float: Aggregated mu value
    """
    # NumPy optimized implementation
    mu_array = np.array(mu_values)

    # Calculate components
    nominator = 1 + zeta * np.power(mu_array, 2)
    denominator = 1 + zeta

    # Weighted product using log-sum-exp technique
    log_terms = np.log(nominator / denominator) * omega
    log_product = np.sum(log_terms)
    loop_product = np.exp(log_product)

    # Final calculation
    result = np.sqrt(((1 + zeta) * loop_product - 1) / zeta)

    return result


def aggregate_nu_values(nu_values, omega, zeta):
    """
    Aggregate multiple nu values into a single value using weighted geometric mean.

    Formula:
    ν_BAA = √(1 - (1/ζ) * ((((1 + ζ)/Π[(1 + ζ)/(1 + ζ * (1 - νᵢ²))]^ωᵢ) - 1))

    Args:
        nu_values (list or numpy.ndarray): List of nu values to aggregate
        omega (list or numpy.ndarray or optional): Weights for each value (default: None)
        zeta (float): Operational parameter

    Returns:
        float: Aggregated nu value
    """
    # NumPy optimized implementation
    nu_array = np.array(nu_values)

    # Calculate components
    denominator = 1 + zeta * (1 - np.power(nu_array, 2))
    nominator = 1 + zeta

    # Weighted product using log-sum-exp
    log_terms = np.log(nominator / denominator) * omega
    log_product = np.sum(log_terms)
    loop_product = np.exp(log_product)

    # Final calculation
    zeta_loop = (((1 + zeta) / loop_product) - 1)
    result = np.sqrt(1 - (1 / zeta) * zeta_loop)

    return result


def calculate_border_approximation_area(matrix, omega=None, zeta=2, t_value=8):
    """
    Calculate Border Approximation Area (BAA) matrix from a decision matrix.
    Transforms an m×n matrix into a 1×n matrix of BAA values.

    Args:
        matrix (pandas.DataFrame): The decision matrix of size m×n, where m is the number of
                                  alternatives and n is the number of criteria.
        omega (list or numpy.ndarray or optional): Weights for each value (default: None)
        zeta (float): Operational parameter (default=2)
        t_value (int): Max linguistic term (default=8)

    Returns:
        pandas.DataFrame: A 1×n DataFrame containing the BAA values for each criterion.
    """
    # Ignore any non-criteria columns (like 'Alternative')
    criteria_columns = [col for col in matrix.columns if col != 'Alternative']

    # Create a BAA matrix with only the criteria columns
    baa_matrix = pd.DataFrame(index=['BAA'], columns=criteria_columns)

    # If omega is a dictionary, convert to list based on matrix index
    omega_list = None
    if omega is not None:
        if isinstance(omega, dict):
            # If omega is a dictionary mapping alternative names to weights
            if 'Alternative' in matrix.columns:
                omega_list = np.array([omega.get(alt, 1.0 / len(matrix))
                                       for alt in matrix['Alternative']])
            else:
                # If no 'Alternative' column, use index as keys
                omega_list = np.array([omega.get(idx, 1.0 / len(matrix))
                                       for idx in matrix.index])
        else:
            # If omega is already a list
            omega_list = np.array(omega)

            # Ensure omega list has correct length
            if len(omega_list) != len(matrix):
                raise ValueError(f"Omega length ({len(omega_list)}) must match number of alternatives ({len(matrix)})")

    for col in criteria_columns:
        column_values = matrix[col].tolist()

        # Check if values are tuples (Interval Type-2 Fermatean fuzzy values)
        if isinstance(column_values[0], tuple) and len(column_values[0]) == 6:
            # Aggregate values for this column into a single tuple
            baa_matrix.at['BAA', col] = aggregate_column_values(column_values, omega_list, zeta, t_value)
        else:
            # For non-tuple values, use a weighted average if omega is provided
            try:
                column_array = np.array(column_values)
                if omega_list is not None:
                    baa_matrix.at['BAA', col] = np.sum(column_array * omega_list)
                else:
                    baa_matrix.at['BAA', col] = np.mean(column_array)
            except:
                baa_matrix.at['BAA', col] = column_values[0]

    return baa_matrix



####### CALCULATE DISTANCE BETWEEN BORDERLINE AND ALTERNATIVES #######


def calculate_distance_matrix(weighted_matrix, baa_matrix, t_value=8):
    """
    Calculate distance matrix between alternatives and BAA values.

    Args:
        weighted_matrix (pandas.DataFrame): The weighted decision matrix with alternatives
        baa_matrix (pandas.DataFrame): The border approximation area matrix
        t_value (int): Max linguistic term (default=8)

    Returns:
        pandas.DataFrame: Distance matrix showing the distance of each alternative from BAA
    """
    # Extract alternatives from the weighted matrix
    alternatives = weighted_matrix['Alternative'].tolist()

    # Get criteria columns (exclude 'Alternative' column)
    criteria_columns = [col for col in weighted_matrix.columns if col != 'Alternative']

    # Create empty distance matrix with alternatives and criteria
    distance_matrix = pd.DataFrame(index=alternatives, columns=criteria_columns)

    # Calculate distance for each alternative and criterion
    for alt in alternatives:
        alt_row = weighted_matrix[weighted_matrix['Alternative'] == alt]

        for criterion in criteria_columns:
            # Get the alternative value and BAA value for this criterion
            alt_value = alt_row[criterion].iloc[0]  # Tuple from weighted matrix
            baa_value = baa_matrix.at['BAA', criterion]  # Tuple from BAA matrix

            # Calculate distance between alternative and BAA for this criterion
            distance = calculate_tuple_distance(alt_value, baa_value, t_value)
            distance_matrix.at[alt, criterion] = distance

    return distance_matrix


def calculate_tuple_distance(alt_tuple, baa_tuple, t_value=8):
    """
    Calculate distance between two IT2FF tuples.

    Args:
        alt_tuple (tuple): Alternative tuple (θL, μL, νL, θU, μU, νU)
        baa_tuple (tuple): BAA tuple (θL, μL, νL, θU, μU, νU)
        t_value (int): Max linguistic term

    Returns:
        float: Distance between the tuples
    """


    if get_score(alt_tuple) > get_score(baa_tuple):
        return get_distance(alt_tuple, baa_tuple, t_value)

    elif get_score(alt_tuple) < get_score(baa_tuple):
        return - 1 * get_distance(baa_tuple, alt_tuple, t_value)

    elif get_score(alt_tuple) == get_score(baa_tuple) :
        return 0

def get_distance(alt_tuple, baa_tuple, t_value):

    alt_theta_l, alt_mu_l, alt_nu_l, alt_theta_u, alt_mu_u, alt_nu_u = alt_tuple
    baa_theta_l, baa_mu_l, baa_nu_l, baa_theta_u, baa_mu_u, baa_nu_u = baa_tuple

    # Calculate distance for each component
    distance = ( 1 / 4 * t_value ) * (
        abs(alt_theta_l * alt_mu_l - baa_theta_l * baa_mu_l) +
        abs(alt_theta_l * alt_nu_l - baa_theta_l * baa_nu_l) +
        abs(alt_theta_u * alt_mu_u - baa_theta_u * baa_mu_u) +
        abs(alt_theta_u * alt_nu_u - baa_theta_u * baa_nu_u)
    )

    return distance