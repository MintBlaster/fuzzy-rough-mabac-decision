import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Import modules from our organized structure
from src.aggregation.aggregators import get_expert_aggregation, get_criterion_aggregation, aggregate_expert_matrices
from src.score.scores import add_final_scores, add_normalized_scores
from src.processing.normalization import create_linguistic_dict, process_matrix, normalization_expert_matrix
from src.utils.data_utils import calculate_weighted_result, calculate_border_approximation_area, \
    calculate_distance_matrix

# Define criteria types
criteria_types = {
    'Cost': 'cost',
    'Infrastructure': 'benefit',
    'Financial Support': 'benefit',
    'Carbon Emission': 'benefit',
    'Utilization of RE': 'benefit',
    'Water Consumption': 'cost',
    'Waste & Byproducts': 'benefit',
    'Hydrogen Purity': 'benefit',
    'Distribution & Storage': 'benefit',
    'Maturity': 'benefit',
    'Experience': 'benefit',
    'Resilience': 'benefit',
    'Geographical Proximity': 'benefit',
    'Safety Measures': 'benefit',
    'Geographical Area': 'benefit',
    'Time Limit': 'benefit',
    'Delivery Services': 'benefit',
    'After-Sales Services': 'benefit',
    'Production Capacity': 'benefit',
    'Product Availability': 'benefit',
    'Product Variation': 'benefit',
    'Product Performance': 'benefit'
}


def run_analysis(zeta):
    # Step 1: Load data
    linguistic_df = pd.read_csv("../data/linguistic_terms.csv")
    expert_matrix = np.array([
        ["H", "MH", "VH"],  # Expert 1
        ["VH", "M", "MH"],  # Expert 2
        ["MH", "H", "H"]  # Expert 3
    ])

    # Step 2: Process expert aggregation
    expert_aggregated = get_expert_aggregation(expert_matrix, linguistic_df, zeta=zeta)
    expert_aggregated = add_final_scores(expert_aggregated)
    expert_aggregated = add_normalized_scores(expert_aggregated)

    # Step 3: Process criteria
    criteria_matrix = pd.read_csv("../data/criteria_matrix.csv", index_col=0)
    criteria_aggregated = get_criterion_aggregation(criteria_matrix, zeta=zeta)
    criteria_aggregated = add_final_scores(criteria_aggregated)
    criteria_aggregated = add_normalized_scores(criteria_aggregated)

    # Step 4: Process expert matrices
    expert_matrix1 = pd.read_csv("../data/expert_matrix1.csv")
    expert_matrix2 = pd.read_csv("../data/expert_matrix2.csv")
    expert_matrix3 = pd.read_csv("../data/expert_matrix3.csv")

    # Create linguistic dictionary
    linguistic_dict = create_linguistic_dict(linguistic_df)

    # Process each matrix
    expert_matrix1 = process_matrix(expert_matrix1, linguistic_dict)
    expert_matrix2 = process_matrix(expert_matrix2, linguistic_dict)
    expert_matrix3 = process_matrix(expert_matrix3, linguistic_dict)

    # Normalize matrices based on criteria types
    em1_n = normalization_expert_matrix(expert_matrix1, criteria_types)
    em2_n = normalization_expert_matrix(expert_matrix2, criteria_types)
    em3_n = normalization_expert_matrix(expert_matrix3, criteria_types)

    normalized_matrices = [em1_n, em2_n, em3_n]

    aggregated_expert_matrix = aggregate_expert_matrices(normalized_matrices, weights_df=expert_aggregated, zeta=zeta)

    # Step 5: Calculate weighted results
    weighted_expert_matrix = calculate_weighted_result(aggregated_expert_matrix, criteria_aggregated, zeta=zeta)

    # Step 6: Calculate Border Approximation Area
    baa_matrix = calculate_border_approximation_area(weighted_expert_matrix, zeta=zeta)

    # Step 7: Calculate distance matrix
    distance_matrix = calculate_distance_matrix(weighted_expert_matrix, baa_matrix)

    # Step 8: Calculate final rankings
    distance_sums = distance_matrix.sum(axis=1)

    # Create closeness coefficients
    closeness_df = pd.DataFrame({
        'Alternative': distance_matrix.index,
        'Closeness Coefficient': distance_sums
    })

    # Find the maximum value and corresponding alternative
    best_alternative = distance_sums.idxmax()
    best_score = distance_sums.max()

    return {
        'closeness_df': closeness_df,
        'best_alternative': best_alternative,
        'best_score': best_score
    }


def main():

    os.makedirs("../results", exist_ok=True)

    # Range of zeta values from 1.0 to 10.0 with step 0.1
    zeta_values = np.arange(.1, 10.1, 0.1)

    # Store results for each zeta value
    results = []

    # Run analysis for each zeta value
    for zeta in zeta_values:
        try:
            print(f"Running analysis with zeta = {zeta:.1f}")
            analysis_result = run_analysis(zeta)

            # Get all alternatives and their scores
            closeness_df = analysis_result['closeness_df']

            # Store result for this zeta value
            for idx, row in closeness_df.iterrows():
                results.append({
                    'zeta': zeta,
                    'alternative': row['Alternative'],
                    'score': row['Closeness Coefficient']
                })

            # Print the best alternative for this zeta
            print(
                f"Best alternative for zeta = {zeta:.1f}: {analysis_result['best_alternative']} with score {analysis_result['best_score']:.4f}")

        except Exception as e:
            print(f"Error with zeta = {zeta:.1f}: {e}")

    # Convert results to DataFrame for easier analysis
    results_df = pd.DataFrame(results)

    # Save full results to CSV
    results_df.to_csv("../results/zeta_analysis_full_results.csv", index=False)

    # Create a pivot table to reorganize data
    pivot_df = results_df.pivot(index='zeta', columns='alternative', values='score')

    # Save pivot table to CSV
    pivot_df.to_csv("../results/zeta_vs_alternatives_scores.csv")

    # Plot results
    plt.figure(figsize=(15, 8))
    for alternative in pivot_df.columns:
        plt.plot(pivot_df.index, pivot_df[alternative], marker='o', markersize=3, label=alternative)

    plt.xlabel('Zeta Value')
    plt.ylabel('Closeness Coefficient')
    plt.title('Effect of Zeta Value on Alternative Rankings')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("../results/zeta_analysis_results.png", dpi=300)
    plt.show()

    # Find the best alternative across all zeta values
    best_per_zeta = results_df.loc[results_df.groupby('zeta')['score'].idxmax()]

    # Save best alternatives per zeta to CSV
    best_per_zeta.to_csv("../results/best_alternative_per_zeta.csv", index=False)

    # Count occurrences of each alternative as the best
    best_counts = best_per_zeta['alternative'].value_counts()

    # Create and save summary dataframe
    summary_df = pd.DataFrame({
        'alternative': best_counts.index,
        'count': best_counts.values,
        'percentage': best_counts.values / len(zeta_values) * 100
    })
    summary_df.to_csv("../results/best_alternative_summary.csv", index=False)

    print("\n=== Summary of Best Alternatives ===")
    for alt, count in best_counts.items():
        print(f"{alt}: Best in {count} of {len(zeta_values)} zeta values ({count / len(zeta_values) * 100:.1f}%)")

    # Print zeta ranges where each alternative is best
    print("\n=== Zeta Ranges for Best Alternatives ===")
    zeta_ranges_dict = {}
    for alt in best_counts.index:
        zeta_ranges = best_per_zeta[best_per_zeta['alternative'] == alt]['zeta'].tolist()
        zeta_ranges_str = ", ".join([f"{z:.1f}" for z in zeta_ranges])
        print(f"{alt}: zeta values = [{zeta_ranges_str}]")
        zeta_ranges_dict[alt] = zeta_ranges

    # Save zeta ranges to CSV
    zeta_ranges_df = pd.DataFrame({
        'alternative': list(zeta_ranges_dict.keys()),
        'zeta_ranges': [", ".join([f"{z:.1f}" for z in ranges]) for ranges in zeta_ranges_dict.values()]
    })
    zeta_ranges_df.to_csv("../results/zeta_ranges_per_alternative.csv", index=False)

    # Plot a histogram of best alternatives
    plt.figure(figsize=(12, 6))
    best_counts.plot(kind='bar')
    plt.xlabel('Alternative')
    plt.ylabel('Count of times ranked best')
    plt.title('Frequency of Alternatives Ranked as Best Across Different Zeta Values')
    plt.tight_layout()
    plt.savefig("../results/best_alternative_frequency.png", dpi=300)
    plt.show()


if __name__ == "__main__":
    main()