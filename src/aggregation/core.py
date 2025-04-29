import math
import numpy as np


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


def aggregate_terms(data, weights=None, zeta=2, t=8):
    """
    Aggregates terms for both lower and upper approximations.

    Args:
        data: DataFrame containing theta_lower, mu_lower, nu_lower, theta_upper, mu_upper, nu_upper
        weights: Optional weights for each row (default=equal weights)
        t: Scale parameter for linguistic values (default=8)
        zeta : Operational parameter affecting aggregation behavior (default=2)

    Returns:
        Dictionary with aggregated values for all terms
    """
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
        'theta_lower': aggregate_linguistic(theta_lower, max_linguistic_term=t, zeta=zeta, weights=weights),
        'mu_lower': aggregate_membership(mu_lower, zeta=zeta, weights=weights),
        'nu_lower': aggregate_non_membership(nu_lower, zeta=zeta, weights=weights),
        'theta_upper': aggregate_linguistic(theta_upper, max_linguistic_term=t, zeta=zeta, weights=weights),
        'mu_upper': aggregate_membership(mu_upper, zeta=zeta, weights=weights),
        'nu_upper': aggregate_non_membership(nu_upper, zeta=zeta, weights=weights)
    }

    return result_dict