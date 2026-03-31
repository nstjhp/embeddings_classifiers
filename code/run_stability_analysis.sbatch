#!/bin/bash
#SBATCH --job-name=hgb_train_pred_agg
#SBATCH --output=/lisc/data/scratch/dome/wagner/pullen/embed_classifiers/general_classifier/jobfiles/job%A_%a_%N.out
#SBATCH --error=/lisc/data/scratch/dome/wagner/pullen/embed_classifiers/general_classifier/jobfiles/job%A_%a_%N.err
#SBATCH --mail-type=END,FAIL
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
# #SBATCH --mem-per-cpu=16000M # only for CPU memory
#SBATCH --mem=5000M # only for CPU memory
#SBATCH --time=0-1:00:00
#SBATCH --partition=basic

# Exit the slurm script if a command fails
set -e

JOB_PARAM_STRING="${SLURM_ARRAY_JOB_ID}_${SLURM_JOB_ID}_${SLURMD_NODENAME}_${SLURM_CPUS_PER_TASK}cpus"
# These commands will go into stdout
echo "Job ID: ${SLURM_JOB_ID}"
echo "Job Array ID: ${SLURM_ARRAY_JOB_ID}"
echo "TMPDIR: ${TMPDIR}"
echo "Node: ${SLURMD_NODENAME}"
echo "CPUs per task: ${SLURM_CPUS_PER_TASK}"
echo "JOB_PARAM_STRING: $JOB_PARAM_STRING"

# For use in the Python logging, here we set MY_SLURM_PROCESS_ID
# and base it on whether this is an array job (-n checks if a string is non-empty)
if [[ -n "${SLURM_ARRAY_JOB_ID}" && -n "${SLURM_ARRAY_TASK_ID}" ]]; then
    # This is an array job
    export MY_SLURM_PROCESS_ID="${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
else
    # This is not an array job; use SLURM_JOB_ID instead
    export MY_SLURM_PROCESS_ID="${SLURM_JOB_ID}"
fi
echo "MY_SLURM_PROCESS_ID: ${MY_SLURM_PROCESS_ID}"

# --- Configuration ---
N_RUNS=50
MODEL_TYPE="xgb" # Model to train (e.g., xgb, hgb, rf)
MAIN_DIR="/lisc/data/scratch/dome/wagner/pullen/embed_classifiers/general_classifier"
MODEL_SAVE_DIR="${MAIN_DIR}/models/${MY_SLURM_PROCESS_ID}_${MODEL_TYPE}"
PRED_SAVE_DIR="${MAIN_DIR}/predictions/${MY_SLURM_PROCESS_ID}_${MODEL_TYPE}"
FINAL_RESULTS_DIR="${MAIN_DIR}/final_results/${MY_SLURM_PROCESS_ID}_${MODEL_TYPE}"
BASE_DATA_PATH="${MAIN_DIR}/data/NAR/nar_POS_NEG_embeddings.csv"
HOLDOUT_NEGATIVES_PATH="${MAIN_DIR}/data/NAR/nar_holdout_negatives.csv"
INDEPENDENT_TEST_PATH="${MAIN_DIR}/data/NAR/nar_PUTATIVE_embeddings.csv"
LABEL_COLUMN_NAME="nar_evidence"

TRAINING_SCRIPT="${MAIN_DIR}/code/train.py"
PRED_SCRIPT="${MAIN_DIR}/code/predict.py"
AGGREGATE_SCRIPT="${MAIN_DIR}/code/aggregate_predictions.py"
ANALYSIS_SCRIPT="${MAIN_DIR}/code/analyse_stability.py"

# for XGB
#FIXED_HYPERPARAMETERS='{"reg_alpha": 0.1, "subsample": 0.8, "min_child_weight": 1, "colsample_bytree": 0.8, "max_depth": 8, "learning_rate": 0.05, "reg_lambda": 5, "scale_pos_weight": 2.0}'
# for HGB
FIXED_HYPERPARAMETERS='{"l2_regularization": 0.0, "learning_rate": 0.1, "max_leaf_nodes": 63, "min_samples_leaf": 50}'

# --- Setup Directories ---
#echo "Setting up directories..."
#mkdir -p models predictions final_results

# --- Data Preparation (One-time setup) ---
#echo "Preparing data splits..."
# This assumes you have a tool or script to split your negatives.
# For example, using Python:
# python -c "import pandas as pd; df = pd.read_csv('$BASE_DATA_PATH'); \
# negatives = df[df['label'] == 0]; \
# holdout = negatives.sample(frac=0.5, random_state=42); \
# training_negs = negatives.drop(holdout.index); \
# positives = df[df['label'] == 1]; \
# pd.concat([positives, training_negs]).to_csv('training_data_pool.csv', index=False); \
# holdout.to_csv('$HOLDOUT_NEGATIVES_PATH', index=False)"
#
# For this script, we'll assume a 'training_data_pool.csv' exists,
# containing all positives and the negatives available for training/sampling.
TRAINING_POOL_PATH="${MAIN_DIR}/data/NAR/nar_training_data_pool.csv"
if [ ! -f "$TRAINING_POOL_PATH" ]; then
    echo "Error: training_data_pool.csv not found. Please prepare it first."
    exit 1
fi


# --- 1. Training Loop ---
echo "Starting training loop for $N_RUNS runs..."
# make a directory to save N_RUNS models
mkdir -p "$MODEL_SAVE_DIR"
for i in $(seq 1 $N_RUNS)
do
  echo "--- RUN $i / $N_RUNS ---"
  
  # Define paths for this run
  MODEL_SAVE_PATH="${MODEL_SAVE_DIR}/pipeline_$(printf %03d ${i}).joblib"
  
  # Train the model. Each run gets a different random seed for data sampling.
  /usr/bin/time python "$TRAINING_SCRIPT" \
    --data-path "$TRAINING_POOL_PATH" \
    --label-col "$LABEL_COLUMN_NAME" \
    --model "$MODEL_TYPE" \
    --balance \
    --random-state "$i" \
    --hyperparams "$FIXED_HYPERPARAMETERS" \
    --save-model-path "$MODEL_SAVE_PATH"
#    --rf-n-jobs 8 \
#    --test-size 0.2 --val-size 0.2 \
done

# --- 2. Prediction Loop ---
echo "Starting prediction loop..."
# make a directory to save N_RUNS predictions
mkdir -p "$PRED_SAVE_DIR"

for i in $(seq 1 $N_RUNS)
do
  echo "--- PREDICTING WITH MODEL FROM RUN $i ---"
  
  MODEL_PATH="${MODEL_SAVE_DIR}/pipeline_$(printf %03d ${i}).joblib"
  
  # Predict on holdout negatives
  /usr/bin/time python "$PRED_SCRIPT" \
    --model-path "$MODEL_PATH" \
    --data-path "$HOLDOUT_NEGATIVES_PATH" \
    --out-path "${PRED_SAVE_DIR}/holdout_negatives_preds_$(printf %03d ${i}).csv" \
    --cols-to-drop h5_index "$LABEL_COLUMN_NAME" dataset_tag
    
  # Predict on independent test set
  /usr/bin/time python $PRED_SCRIPT \
    --model-path "$MODEL_PATH" \
    --data-path "$INDEPENDENT_TEST_PATH" \
    --out-path "${PRED_SAVE_DIR}/independent_data_preds_$(printf %03d ${i}).csv" \
    --cols-to-drop h5_index "$LABEL_COLUMN_NAME" dataset_tag

  # Predict on training pool
  /usr/bin/time python "$PRED_SCRIPT" \
    --model-path "$MODEL_PATH" \
    --data-path "$TRAINING_POOL_PATH" \
    --out-path "${PRED_SAVE_DIR}/training_pool_preds_$(printf %03d ${i}).csv" \
    --cols-to-drop h5_index "$LABEL_COLUMN_NAME" dataset_tag
done

# --- 3. Aggregation Step ---
echo "Aggregating all predictions..."
/usr/bin/time python $AGGREGATE_SCRIPT \
  --predictions-dir "$PRED_SAVE_DIR" \
  --n-runs "$N_RUNS" \
  --model-type "$MODEL_TYPE" \
  --dataset-name "holdout_negatives" \
  --output-path "${FINAL_RESULTS_DIR}/aggregated_holdout_negatives.csv"

/usr/bin/time python $AGGREGATE_SCRIPT \
  --predictions-dir "$PRED_SAVE_DIR" \
  --n-runs "$N_RUNS" \
  --model-type "$MODEL_TYPE" \
  --dataset-name "independent_data" \
  --output-path "${FINAL_RESULTS_DIR}/aggregated_independent_data.csv"

/usr/bin/time python $AGGREGATE_SCRIPT \
  --predictions-dir "$PRED_SAVE_DIR" \
  --n-runs "$N_RUNS" \
  --model-type "$MODEL_TYPE" \
  --dataset-name "training_pool" \
  --output-path "${FINAL_RESULTS_DIR}/aggregated_training_pool.csv"

echo "Aggregation done. Final results are in the '${FINAL_RESULTS_DIR}' directory."

/usr/bin/time python $ANALYSIS_SCRIPT \
  --input-csv "${FINAL_RESULTS_DIR}/aggregated_holdout_negatives.csv" \
  --save-summary-path "${FINAL_RESULTS_DIR}/summary_stats_holdout.csv" \
  --plot-distributions mean std median mad min max range q25 q75 iqr cv \
  --plot-output-path "${FINAL_RESULTS_DIR}/distributions_plot_holdout.png" \
  --interactive-plot-path "${FINAL_RESULTS_DIR}/interactive_stability_plot_holdout.html"

/usr/bin/time python $ANALYSIS_SCRIPT \
  --input-csv "${FINAL_RESULTS_DIR}/aggregated_independent_data.csv" \
  --save-summary-path "${FINAL_RESULTS_DIR}/summary_stats_maybes.csv" \
  --plot-distributions mean std median mad min max range q25 q75 iqr cv \
  --plot-output-path "${FINAL_RESULTS_DIR}/distributions_plot_maybes.png" \
  --interactive-plot-path "${FINAL_RESULTS_DIR}/interactive_stability_plot_maybes.html"

/usr/bin/time python $ANALYSIS_SCRIPT \
  --input-csv "${FINAL_RESULTS_DIR}/aggregated_training_pool.csv" \
  --save-summary-path "${FINAL_RESULTS_DIR}/summary_stats_training_pool.csv" \
  --plot-distributions mean std median mad min max range q25 q75 iqr cv \
  --plot-output-path "${FINAL_RESULTS_DIR}/distributions_plot_training_pool.png" \
  --interactive-plot-path "${FINAL_RESULTS_DIR}/interactive_stability_plot_training_pool.html"

echo "Summary stats calculated and graphs made. Output is in the '${FINAL_RESULTS_DIR}' directory."

cat "${FINAL_RESULTS_DIR}/summary_stats_training_pool.csv" "${FINAL_RESULTS_DIR}/summary_stats_holdout.csv" "${FINAL_RESULTS_DIR}/summary_stats_maybes.csv" > "${FINAL_RESULTS_DIR}/summary_stats_ALL.csv"

echo "All datasets combined. Output is ${FINAL_RESULTS_DIR}/summary_stats_ALL.csv"

# Append the contents of this script to the output file
echo "=== Job Script Contents ==="
cat $0
echo "==========================="

# If we reached this point, we succeeded. We clean up resources.
rm -rf $TMPDIR
