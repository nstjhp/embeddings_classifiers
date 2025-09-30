import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from scipy.stats import median_abs_deviation

# --- EXAMPLE USAGE ---
# **Example 1: Calculate and Save Summary Statistics**
# python code/analyse_stability.py \
#   --input-csv final_results/aggregated_holdout_negatives.csv \
#   --save-summary-path final_results/summary_stats_holdout.csv
# 
# **Example 2: Create Static Distribution Plots**
# Visualise the overall distributions of mean confidence, standard deviation, and the coefficient of variation.
# Will generate a PNG file showing three subplots.
# python code/analyse_stability.py \
#   --input-csv final_results/aggregated_holdout_negatives.csv \
#   --plot-distributions mean std median mad min max range q25 q75 iqr cv \
#   --plot-output-path final_results/distributions_plot.png
# 
# **Example 3: Generate the Interactive Plot **
# python code/analyse_stability.py \
#   --input-csv final_results/aggregated_holdout_negatives.csv \
#   --interactive-plot-path final_results/interactive_stability_plot.html
#
# **Example 4: Do everything! **
# python code/analyse_stability.py --input-csv final_results/aggregated_independent_data.csv --save-summary-path final_results/summary_stats_maybes.csv --plot-distributions mean std median mad min max range q25 q75 iqr cv --plot-output-path final_results/distributions_plot_maybes.png  --interactive-plot-path final_results/interactive_stability_plot_maybes.html

# --- Part 1: Summary Statistics ---

def calculate_summary_stats(df: pd.DataFrame, protein_col: str = 'protein') -> pd.DataFrame:
    """
    Calculates detailed summary statistics for each protein across multiple prediction runs.

    Args:
        df: DataFrame in wide format (protein_id, prob_run_1, prob_run_2, ...).
        protein_col: The name of the column containing protein IDs.

    Returns:
        A DataFrame with one row per protein and columns for each summary statistic.
    """
    print("Calculating summary statistics...")
    if protein_col not in df.columns:
        raise ValueError(f"Protein column '{protein_col}' not found in DataFrame.")

    prob_cols = [col for col in df.columns if col.endswith('_prob')]
    if not prob_cols:
        raise ValueError("No probability columns (ending in '_prob') found.")
    
    print(f"Found {len(prob_cols)} probability columns to analyse.")
    
    # Isolate probability data for calculations
    prob_data = df[prob_cols]

    # Create a new DataFrame for the results
    summary_df = pd.DataFrame()
    
    # --- Calculate Statistics ---
    summary_df['mean'] = prob_data.mean(axis=1)
    summary_df['std'] = prob_data.std(axis=1)
    summary_df['median'] = prob_data.median(axis=1)
    
    # Median Absolute Deviation (MAD) - a robust measure of spread
    summary_df['mad'] = median_abs_deviation(prob_data, axis=1, scale='normal')
    
    summary_df['min'] = prob_data.min(axis=1)
    summary_df['max'] = prob_data.max(axis=1)
    summary_df['range'] = summary_df['max'] - summary_df['min']
    
    summary_df['q25'] = prob_data.quantile(0.25, axis=1)
    summary_df['q75'] = prob_data.quantile(0.75, axis=1)
    summary_df['iqr'] = summary_df['q75'] - summary_df['q25']
    
    # Coefficient of Variation (CV) - normalised measure of variability
    # Add a small epsilon to avoid division by zero for proteins with a mean of 0
    summary_df['cv'] = summary_df['std'] / (summary_df['mean'] + 1e-9)

    summary_df.index = df[protein_col]
    print("Summary statistics calculated successfully.")
    return summary_df.reset_index()


# --- Part 2: Static Distribution Visualizations ---

def plot_distributions(summary_df: pd.DataFrame, columns_to_plot: list, output_path: str):
    """
    Plots histograms and KDEs for selected summary statistics.

    Args:
        summary_df: The DataFrame of summary statistics.
        columns_to_plot: A list of column names to visualise.
        output_path: Path to save the output plot image.
    """
    print(f"Generating distribution plots for: {', '.join(columns_to_plot)}...")
    
    n_plots = len(columns_to_plot)
    n_cols = 3 if n_plots > 2 else 2
    n_rows = (n_plots + 1) // 3
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4 * n_rows), squeeze=False)
    axes = axes.flatten()

    for i, col in enumerate(columns_to_plot):
        if col not in summary_df.columns:
            print(f"Warning: Column '{col}' not found in summary data. Skipping plot.")
            continue
        sns.histplot(summary_df[col], kde=True, ax=axes[i], bins=50)
        axes[i].set_title(f'Distribution of {col.capitalize()}', fontsize=16)
        axes[i].set_xlabel(col.capitalize(), fontsize=12)
        axes[i].set_ylabel('Frequency', fontsize=12)

    # Hide any unused subplots
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])
        
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"Saved distribution plot to {output_path}")


# --- Part 3: Interactive Exploration Plot ---

def create_interactive_plot(summary_df: pd.DataFrame, output_path: str, protein_col: str = 'protein'):
    """
    Creates an interactive scatter plot to explore prediction variability.

    Args:
        summary_df: The DataFrame of summary statistics.
        output_path: Path to save the output interactive HTML file.
        protein_col: The name of the column containing protein IDs.
    """
    print("Generating interactive variability plot...")
    
    fig = px.scatter(
        summary_df,
        x='mean',
        y='std',
        color='range',
        opacity=0.4,
        color_continuous_scale=px.colors.sequential.Turbo,
        hover_data=[protein_col, 'median', 'cv', 'range'],
        labels={
            'mean': 'Mean Predicted Probability',
            'std': 'Standard Deviation of Predictions',
            'iqr': 'Interquartile Range'
        },
        title='Prediction Stability Analysis: Mean vs. Standard Deviation'
    )

    fig.update_layout(
        xaxis_title='Mean Probability across all runs',
        yaxis_title='Standard Deviation across all runs',
        coloraxis_colorbar_title_text='Range'
    )
    
    fig.write_html(output_path)
    print(f"Saved interactive plot to {output_path}")


# --- Main CLI ---

def main():
    parser = argparse.ArgumentParser(description="Analyse and visualise protein prediction stability from aggregated runs.")
    parser.add_argument("--input-csv", type=str, required=True, help="Path to the aggregated predictions CSV file.")
    parser.add_argument("--protein-col", type=str, default="protein", help="Name of the protein ID column.")
    
    parser.add_argument("--save-summary-path", type=str, help="Optional. Path to save the summary statistics CSV.")
    
    parser.add_argument("--plot-distributions", nargs='+', help="A list of summary stats to plot distributions for (e.g., mean std cv iqr).")
    parser.add_argument("--plot-output-path", type=str, help="Path to save the static distribution plot image. Required if --plot-distributions is used.")
    
    parser.add_argument("--interactive-plot-path", type=str, help="Optional. Path to save the interactive HTML variability plot.")
    
    args = parser.parse_args()

    if args.plot_distributions and not args.plot_output_path:
        parser.error("--plot-output-path is required when using --plot-distributions.")

    # --- Load Data ---
    print(f"Loading aggregated predictions from {args.input_csv}...")
    try:
        agg_df = pd.read_csv(args.input_csv)
    except FileNotFoundError:
        print(f"Error: Input file not found at {args.input_csv}")
        return
    except Exception as e:
        print(f"Error loading CSV file: {e}")
        return

    print(f"Loaded data with shape: {agg_df.shape}")

    # --- Run Analysis ---
    summary_stats_df = calculate_summary_stats(agg_df, protein_col=args.protein_col)

    if args.save_summary_path:
        summary_stats_df.to_csv(args.save_summary_path, index=False)
        print(f"Summary statistics saved to {args.save_summary_path}")

    if args.plot_distributions:
        plot_distributions(summary_stats_df, args.plot_distributions, args.plot_output_path)
    
    if args.interactive_plot_path:
        create_interactive_plot(summary_stats_df, args.interactive_plot_path, protein_col=args.protein_col)

    print("\nAnalysis complete.")

if __name__ == "__main__":
    main()
