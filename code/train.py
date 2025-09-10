import argparse
import time
import warnings
import os
import joblib
import json
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.calibration import IsotonicRegression
from sklearn.preprocessing import Normalizer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
)
from sklearn.utils.class_weight import compute_sample_weight

# Import helpers from utils.py
from utils import (
    load_data,
    calculate_at_k_metrics,
    calculate_ece,
    find_operating_point,
    HGB_PARAM_GRID,
    XGB_PARAM_GRID,
)

warnings.filterwarnings("ignore", category=UserWarning)

# --- Wrapper class to bundle model and calibrator ---
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
    parser = argparse.ArgumentParser(description="Train and evaluate binary classifiers on protein embeddings.")
    # --- Data Args ---
    parser.add_argument("--data-path", type=str, required=True, help="Path to the input CSV data.")
    parser.add_argument("--label-col", type=str, required=True, help="Name of the column containing the labels (0 or 1).")
    parser.add_argument("--balance", action=argparse.BooleanOptionalAction, default=True, help="Toggle negative class downsampling.")
    parser.add_argument("--n-negatives", type=int, help="Exact number of negatives to sample.")
    parser.add_argument("--negative-ratio", type=float, help="Ratio of negatives to positives to sample.")
    parser.add_argument("--shuffle", action=argparse.BooleanOptionalAction, default=True, help="Toggle dataset shuffling.")
    # --- Splitting & Randomness Args ---
    parser.add_argument("--random-state", type=int, default=42, help="Global random seed.")
    parser.add_argument("--test-size", type=float, default=0.2, help="Fraction of data for the final test set.")
    parser.add_argument("--val-size", type=float, default=0.2, help="Fraction of training data for validation/calibration.")
    # --- Model & Training Args ---
    parser.add_argument("--rf-n-jobs", type=int, default=-1, help="Number of jobs for RF/XGB.")
    parser.add_argument("--model", nargs="+", default=["hgb", "xgb"], choices=["rf", "gb", "hgb", "xgb", "logreg", "logreg_l2norm", "all"], help="Models to train.")
    parser.add_argument("--operating-point", default="youden", choices=["youden", "fpr@1", "fpr@5", "ppv@30", "f05"], help="Strategy to select classification threshold.")
    parser.add_argument("--k", nargs="+", type=int, default=[50, 100, 500], help="Values of K for P@K/R@K metrics.")
    # --- Tuning Args ---
    parser.add_argument("--tune", nargs="*", choices=["hgb", "xgb"], default=[], help="Which models to tune.")
    parser.add_argument("--tune-n", type=int, default=20, help="Number of iterations for random search tuning.")
    parser.add_argument("--tune-seed", type=int, default=123, help="Random seed for tuning.")
    # --- Hyperparameters (don't mix with tuning!) ---
    parser.add_argument("--hyperparams", type=str, help="JSON string of fixed hyperparameters to use for the model, skipping tuning.")
    # --- Output Args ---
    parser.add_argument("--out-csv", type=str, help="Path to save prediction probabilities CSV for the training run.")
    parser.add_argument("--save-model-path", type=str, help="Path to save the bundled model and calibrator file (e.g., ./models/{model_name}.joblib). Can use format strings.")

    args = parser.parse_args()

    if args.tune and args.hyperparams:
        parser.error("Cannot use --tune and --hyperparams simultaneously. Use --tune to find params, then use --hyperparams to run with fixed params.")

    if "all" in args.model:
        args.model = ["rf", "gb", "hgb", "xgb", "logreg", "logreg_l2norm"]

    np.random.seed(args.random_state)
    
    # 1. DATA LOADING AND SPLITTING
    X, y, protein_ids = load_data(
        path=args.data_path, label_col = args.label_col, balance=args.balance, n_negatives=args.n_negatives,
        negative_ratio=args.negative_ratio, random_state=args.random_state, shuffle=args.shuffle
    )
    # Stratified split into train+val and test
    X_train_val, X_test, y_train_val, y_test, ids_train_val, ids_test = train_test_split(
        X, y, protein_ids, test_size=args.test_size, stratify=y, random_state=args.random_state
    )
    # Stratified split of train+val into train (for fitting) and val (for calibration/thresholding)
    X_train, X_val, y_train, y_val, _, _ = train_test_split(
        X_train_val, y_train_val, ids_train_val, test_size=args.val_size, stratify=y_train_val, random_state=args.random_state
    )
    print(f"Data splits: Train={X_train.shape}, Val/Calib={X_val.shape}, Test={X_test.shape}")

    all_predictions = pd.DataFrame({"protein": protein_ids})
    all_predictions['split'] = 'not_used'
    all_predictions.loc[ids_train_val.index, 'split'] = 'train_val'
    all_predictions.loc[ids_test.index, 'split'] = 'test'
    all_results = []
    
    # --- MODEL TRAINING AND EVALUATION LOOP ---
    for model_name in args.model:
        print(f"\n--- Training model: {model_name.upper()} ---")
        start_time = time.time()
        
        model = None
        fit_params = {}

        # --- Model Initialisation with Hyperparameter Override ---
        user_params = {}
        if args.hyperparams:
            try:
                user_params = json.loads(args.hyperparams)
                print(f"Using fixed hyperparameters: {user_params}")
            except json.JSONDecodeError:
                raise ValueError("Invalid JSON string passed to --hyperparams.")
        # Base parameters that should always be set
        
        if model_name == 'rf':
            base_rf_params = {
                'n_estimators': 500, 'class_weight': "balanced", 
                'n_jobs': args.rf_n_jobs, 'random_state': args.random_state
            }
            final_params = {**base_rf_params, **user_params}
            model = RandomForestClassifier(**final_params)

        elif model_name == 'gb':
            base_gb_params = {
                    'n_estimators': 300, 'learning_rate': 0.05, 
                    'max_depth': 3, 'random_state': args.random_state
            }
            final_params = {**base_gb_params, **user_params}
            model = GradientBoostingClassifier(**final_params)
            fit_params['sample_weight'] = compute_sample_weight("balanced", y_train)

        elif model_name == 'hgb':
            base_hgb_params = {'random_state': args.random_state}
            final_params = {**base_hgb_params, **user_params}
            model = HistGradientBoostingClassifier(**final_params)
            fit_params['sample_weight'] = compute_sample_weight("balanced", y_train)

        elif model_name == 'xgb':
            neg_count, pos_count = np.bincount(y_train)
            scale_pos_weight = neg_count / pos_count if pos_count > 0 else 1
            base_xgb_params = {
                'tree_method': "hist", 'eval_metric': "aucpr", 'n_estimators': 4000,
                'early_stopping_rounds': 50, 'scale_pos_weight': scale_pos_weight,
                'n_jobs': args.rf_n_jobs, 'random_state': args.random_state, 'use_label_encoder': False
            }
            final_params = {**base_xgb_params, **user_params}
            model = xgb.XGBClassifier(**final_params)
            fit_params['eval_set'] = [(X_val, y_val)]
            fit_params['verbose'] = False

        elif model_name == 'logreg':
            model = LogisticRegression(solver='lbfgs', max_iter=2000, class_weight="balanced", random_state=args.random_state)
        elif model_name == 'logreg_l2norm':
            model = Pipeline([
                ('normalizer', Normalizer(norm='l2')),
                ('logreg', LogisticRegression(solver='lbfgs', max_iter=2000, class_weight="balanced", random_state=args.random_state))
            ])
        
        # --- Hyperparameter Tuning (Optional) ---
        tuning_results = {}
        if model_name in args.tune:
            print(f"Tuning {model_name.upper()}...")
            param_grid = HGB_PARAM_GRID if model_name == 'hgb' else XGB_PARAM_GRID
            if model_name == 'xgb':
                 param_grid['scale_pos_weight'] = [model.scale_pos_weight * m for m in [0.5, 1.0, 2.0]]

            search = RandomizedSearchCV(
                estimator=model,
                param_distributions=param_grid,
                n_iter=args.tune_n,
                scoring='average_precision',
                n_jobs=args.rf_n_jobs,
                cv=3,
                refit=True, # Refits the best estimator on the whole training data
                random_state=args.tune_seed
            )
            # XGB uses a different fit paradigm with eval_set
            if model_name == 'xgb':
                # For tuning, we don't want to use the final validation set.
                # A proper approach would be nested CV, but for a quick tune we can split train again.
                X_train_sub, X_tune_val, y_train_sub, y_tune_val = train_test_split(X_train, y_train, test_size=0.2, stratify=y_train, random_state=args.tune_seed)
                search.fit(X_train_sub, y_train_sub, eval_set=[(X_tune_val, y_tune_val)], verbose=False)
            else:
                search.fit(X_train, y_train)

            model = search.best_estimator_
            tuning_results['best_params'] = str(search.best_params_)
            tuning_results['val_pr_auc'] = search.best_score_
            print(f"Best tuning PR-AUC: {search.best_score_:.4f}")

        # --- Model Fitting ---
        # If tuning was run, model is already refit. If not, fit it now.
        if model_name not in args.tune:
             model.fit(X_train, y_train, **fit_params)
        
        if hasattr(model, 'best_iteration'):
            best_iter = getattr(model, 'best_iteration', getattr(model, 'best_iteration', None))
        else:
            best_iter = getattr(model, 'best_iteration_', getattr(model, 'n_iter_', None))
        if hasattr(model, 'named_steps'): # For pipeline
            best_iter = getattr(model.named_steps['logreg'], 'n_iter_', [None])[0]

        # --- Calibration ---
        print("Calibrating model...")
        raw_scores_val = model.predict_proba(X_val)[:, 1]
        calibrator = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1)
        calibrator.fit(raw_scores_val, y_val, sample_weight=compute_sample_weight("balanced", y_val))

        # --- Save the bundled model and calibrator ---
        if args.save_model_path:
            # Create the bundled object
            calibrated_pipeline = CalibratedModel(model, calibrator)
            
            # Allow formatting the path with the model name
            final_path = args.save_model_path.format(model_name=model_name)
            
            # Ensure the directory exists
            os.makedirs(os.path.dirname(final_path), exist_ok=True)
            
            joblib.dump(calibrated_pipeline, final_path)
            print(f"Saved bundled model and calibrator to {final_path}")

        # --- Getting Predictions (using the bundled object's logic for consistency) ---
        temp_pipeline = CalibratedModel(model, calibrator)
        calibrated_scores_val = temp_pipeline.predict_proba(X_val)
        calibrated_scores_test = temp_pipeline.predict_proba(X_test)

        # Add predictions to the output dataframe
        full_calibrated_preds = temp_pipeline.predict_proba(X)
        all_predictions[f'{model_name}_calibrated_prob'] = pd.Series(full_calibrated_preds, index=X.index)
        
        # --- Threshold Selection ---
        threshold = find_operating_point(y_val, calibrated_scores_val, args.operating_point)
        y_pred_test = (calibrated_scores_test >= threshold).astype(int)

        # --- Metrics Calculation ---
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred_test).ravel()
        metrics = {
            "model": model_name, "operating_point": args.operating_point, "Youden_J_threshold": threshold,
            "accuracy": accuracy_score(y_test, y_pred_test), "balanced_accuracy": balanced_accuracy_score(y_test, y_pred_test),
            "precision": precision_score(y_test, y_pred_test, zero_division=0), "recall": recall_score(y_test, y_pred_test, zero_division=0),
            "f1_score": f1_score(y_test, y_pred_test, zero_division=0), "roc_auc": roc_auc_score(y_test, calibrated_scores_test),
            "pr_auc": average_precision_score(y_test, calibrated_scores_test), "brier_score": brier_score_loss(y_test, calibrated_scores_test),
            "ece": calculate_ece(y_test.values, calibrated_scores_test),
            "max_positive_prob": np.max(calibrated_scores_test[y_test == 1]) if sum(y_test==1)>0 else np.nan,
            "confusion_matrix": f"{{'tn':{tn},'fp':{fp},'fn':{fn},'tp':{tp}}}",
            "tpr": tp / (tp + fn) if (tp + fn) > 0 else 0, "fpr": fp / (fp + tn) if (fp + tn) > 0 else 0,
            "pred_pos_rate": np.mean(y_pred_test), "at_k": str(calculate_at_k_metrics(y_test.values, calibrated_scores_test, args.k)),
            "val_pr_auc": tuning_results.get('val_pr_auc', np.nan),
            "best_iteration": best_iter,
            "best_params": tuning_results.get('best_params', np.nan),
        }
        all_results.append(metrics)
        print(f"Finished {model_name.upper()} in {time.time() - start_time:.2f}s. Test PR-AUC: {metrics['pr_auc']:.4f}")

    # --- Dummy Classifier Baseline ---
    print("\n--- Evaluating Dummy Classifier Baseline ---")
    dummy = DummyClassifier(strategy="most_frequent")
    dummy.fit(X_train, y_train)
    y_pred_dummy = dummy.predict(X_test)
    y_prob_dummy = dummy.predict_proba(X_test)[:, 1]

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred_dummy).ravel()
    dummy_metrics = {
        "model": "dummy_most_frequent", "operating_point": "N/A", "Youden_J_threshold": np.nan,
        "accuracy": accuracy_score(y_test, y_pred_dummy), "balanced_accuracy": balanced_accuracy_score(y_test, y_pred_dummy),
        "precision": precision_score(y_test, y_pred_dummy, zero_division=0), "recall": recall_score(y_test, y_pred_dummy, zero_division=0),
        "f1_score": f1_score(y_test, y_pred_dummy, zero_division=0), "roc_auc": 0.5,
        "pr_auc": average_precision_score(y_test, y_prob_dummy), "brier_score": brier_score_loss(y_test, y_prob_dummy),
        "ece": calculate_ece(y_test.values, y_prob_dummy), "max_positive_prob": np.nan,
        "confusion_matrix": f"{{'tn':{tn},'fp':{fp},'fn':{fn},'tp':{tp}}}",
        "tpr": tp / (tp + fn) if (tp + fn) > 0 else 0, "fpr": fp / (fp + tn) if (fp + tn) > 0 else 0,
        "pred_pos_rate": np.mean(y_pred_dummy), "at_k": str(calculate_at_k_metrics(y_test.values, y_prob_dummy, args.k)),
    }
    all_results.append(dummy_metrics)

    # 3. OUTPUT RESULTS
    results_df = pd.DataFrame(all_results)
    results_df = results_df.drop_duplicates(subset=["model", "operating_point"]).set_index("model")

    pd.set_option('display.max_rows', 500)
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 500)
    pd.set_option('display.max_colwidth', None)# show full width of fields

    print("\n\n--- CLASSIFICATION RESULTS ON TEST SET ---")
    print(results_df[[
        "operating_point", "Youden_J_threshold", "accuracy", "balanced_accuracy",
        "precision", "recall", "f1_score", "roc_auc", "pr_auc", "brier_score",
        "ece", "tpr", "fpr", "confusion_matrix"
    ]])

    print("\n--- PR-AUC SUMMARY (GAIN OVER BASELINE) ---")
    baseline_pr_auc = results_df.loc['dummy_most_frequent', 'pr_auc']
    pr_summary = results_df[['pr_auc']].copy()
    pr_summary['baseline_pr_auc'] = baseline_pr_auc
    pr_summary['pr_gain'] = pr_summary['pr_auc'] - pr_summary['baseline_pr_auc']
    print(pr_summary.sort_values('pr_auc', ascending=False))

    if args.out_csv:
        all_predictions.to_csv(args.out_csv, index=False)
        print(f"\nSaved training run predictions to {args.out_csv}")

    print(results_df)

if __name__ == "__main__":
    main()
