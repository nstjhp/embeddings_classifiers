import argparse
import pandas as pd
import joblib
import os

# --- Wrapper class must be defined here too for joblib to load it ---
class CalibratedModel:
    """A wrapper to hold a model and its calibrator as a single unit."""
    def __init__(self, model, calibrator):
        self.model = model
        self.calibrator = calibrator

    def predict_proba(self, X):
        """Generates calibrated probabilities for new data."""
        raw_scores = self.model.predict_proba(X)[:, 1]
        return self.calibrator.predict(raw_scores)

def main():
    """
    Loads a trained and calibrated model pipeline to predict on a new dataset.
    """
    parser = argparse.ArgumentParser(description="Use a trained model to predict protein classifications.")
    
    parser.add_argument("--model-path", type=str, required=True, help="Path to the bundled model/calibrator .joblib file.")
    parser.add_argument("--data-path", type=str, required=True, help="Path to the input CSV data for prediction.")
    parser.add_argument("--out-path", type=str, required=True, help="Path to save the output CSV with predictions.")
    parser.add_argument("--protein-col", type=str, default="protein", help="Name of the column with protein IDs.")
    parser.add_argument("--cols-to-drop", nargs="*", default=['h5_index', 'FDHevidence'], help="Name(s) of unneeded columns e.g. --cols-to-drop h5_index nar_evidence.")
    
    args = parser.parse_args()

    print("Loading bundled model and calibrator...")
    try:
        pipeline = joblib.load(args.model_path)
    except FileNotFoundError as e:
        print(f"Error: {e}. Please check the model path.")
        return
    except Exception as e:
        print(f"An error occurred while loading the model: {e}")
        print("Ensure the script's CalibratedModel class matches the one used during saving.")
        return

    print(f"Loading data for prediction from {args.data_path}...")
    try:
        df = pd.read_csv(args.data_path)
    except FileNotFoundError:
        print(f"Error: Input data file not found at {args.data_path}")
        return
    
    if args.protein_col not in df.columns:
        print(f"Error: Protein ID column '{args.protein_col}' not found in the data.")
        return

    # Drop columns that are not features (only if they exist)
    if args.cols_to_drop:
        df = df.drop(columns=[c for c in args.cols_to_drop if c in df.columns])

    protein_ids = df[args.protein_col]
    feature_cols = [col for col in df.columns if col != args.protein_col]
    X_predict = df[feature_cols]

    print(f"Generating predictions for {len(X_predict)} samples...")
    # Validate that we have features to predict on
    if X_predict.shape[1] == 0:
        print("Error: No feature columns found for prediction.")
        return

    calibrated_scores = pipeline.predict_proba(X_predict)

    # --- Create output DataFrame and save ---
    output_df = pd.DataFrame({
        args.protein_col: protein_ids,
        "calibrated_probability": calibrated_scores
    })
    
    os.makedirs(os.path.dirname(args.out_path), exist_ok=True)
    output_df.to_csv(args.out_path, index=False)
    print(f"Successfully saved predictions to {args.out_path}")

if __name__ == "__main__":
    main()
