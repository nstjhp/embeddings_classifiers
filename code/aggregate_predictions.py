import argparse
import pandas as pd
import os

def main():
    """
    Aggregates prediction CSVs from multiple runs into a single wide-format CSV.
    """
    parser = argparse.ArgumentParser(description="Aggregate predictions from stability analysis runs.")
    parser.add_argument("--predictions-dir", type=str, required=True, help="Base directory containing the run prediction CSVs. Example format: predictions/{JOBID}_{MODEL_TYPE}/{dataset_name}_preds_{i:03d}.csv - give the dir here.")
    parser.add_argument("--n-runs", type=int, required=True, help="Total number of runs to aggregate.")
    parser.add_argument("--model-type", type=str, required=True, help="Identifier for the model (e.g., 'xgb').")
    parser.add_argument("--dataset-name", type=str, required=True, help="Identifier for the dataset (e.g., 'holdout_negatives').")
    parser.add_argument("--output-path", type=str, required=True, help="Path to save the final aggregated CSV.")
    parser.add_argument("--protein-col", type=str, default="protein", help="Name of the protein ID column.")
    
    args = parser.parse_args()

    # Validate that predictions directory exists
    if not os.path.exists(args.predictions_dir):
        print(f"Error: Predictions directory not found: {args.predictions_dir}")
        return

    if not os.path.isdir(args.predictions_dir):
        print(f"Error: Predictions path is not a directory: {args.predictions_dir}")
        return

    master_df = None
    files_processed = 0

    print(f"Starting aggregation for dataset: {args.dataset_name}")

    for i in range(1, args.n_runs + 1):
        pred_file = os.path.join(args.predictions_dir, f"{args.dataset_name}_preds_{i:03d}.csv")

        if not os.path.exists(pred_file):
            print(f"Warning: Prediction file not found for run {i}, skipping: {pred_file}")
            continue
            
        try:
            run_df = pd.read_csv(pred_file)
            # Validate that required columns exist
            if args.protein_col not in run_df.columns:
                print(f"Error: Protein column '{args.protein_col}' not found in {pred_file}")
                continue

            if "calibrated_probability" not in run_df.columns:
                print(f"Error: 'calibrated_probability' column not found in {pred_file}")
                continue

            # Rename the probability column to be unique for this run
            prob_col_name = f"{args.model_type}_run_{i:02d}_prob"
            run_df = run_df.rename(columns={"calibrated_probability": prob_col_name})

            if master_df is None:
                master_df = run_df
            else:
                # Merge the new run's predictions into the master DataFrame
                master_df = pd.merge(master_df, run_df, on=args.protein_col, how='outer')
            
            files_processed += 1
            print(f"Processed run {i}/{args.n_runs}")

        except Exception as e:
            print(f"Error processing file {pred_file}: {e}")

    if master_df is not None  and files_processed > 0:
        os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
        master_df.to_csv(args.output_path, index=False)
        print(f"\nAggregation complete. Saved {len(master_df)} proteins with {files_processed}/{args.n_runs} predictions each to {args.output_path}")
    else:
        print(f"Error: No prediction files were found or processed. No output generated. Expected {args.n_runs} files but processed {files_processed}.")

if __name__ == "__main__":
    main()
