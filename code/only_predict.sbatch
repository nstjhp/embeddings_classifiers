#!/bin/bash
#SBATCH --job-name=only_pred
#SBATCH --output=/lisc/data/scratch/dome/wagner/pullen/embed_classifiers/general_classifier/jobfiles/job%A_%a_%N.out
#SBATCH --error=/lisc/data/scratch/dome/wagner/pullen/embed_classifiers/general_classifier/jobfiles/job%A_%a_%N.err
#SBATCH --mail-type=END,FAIL
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
# #SBATCH --mem-per-cpu=16000M # only for CPU memory
#SBATCH --mem=5000M # only for CPU memory
#SBATCH --time=0-0:30:00
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
MODEL_TYPE="hgb" # Model to train (e.g., xgb, hgb, rf)
BASE_DATA_PATH="/lisc/data/scratch/dome/wagner/pullen/embed_classifiers/general_classifier/data/FDH_EXP_and_no_FINAL.csv"
HOLDOUT_NEGATIVES_PATH="/lisc/data/scratch/dome/wagner/pullen/embed_classifiers/general_classifier/data/holdout_negatives.csv"
INDEPENDENT_TEST_PATH="/lisc/data/scratch/dome/wagner/pullen/embed_classifiers/general_classifier/data/only_maybes.csv"
TRAINING_POOL_PATH="/lisc/data/scratch/dome/wagner/pullen/embed_classifiers/general_classifier/data/training_data_pool.csv"
LABEL_COLUMN_NAME="FDHevidence"

TRAINING_SCRIPT="/lisc/data/scratch/dome/wagner/pullen/embed_classifiers/general_classifier/code/train.py"
PRED_SCRIPT="/lisc/data/scratch/dome/wagner/pullen/embed_classifiers/general_classifier/code/predict.py"
AGGREGATE_SCRIPT="/lisc/data/scratch/dome/wagner/pullen/embed_classifiers/general_classifier/code/aggregate_predictions.py"
ANALYSIS_SCRIPT="/lisc/data/scratch/dome/wagner/pullen/embed_classifiers/general_classifier/code/analyse_stability.py"

# for HGB
FIXED_HYPERPARAMETERS='{"l2_regularization": 0.0, "learning_rate": 0.1, "max_leaf_nodes": 63, "min_samples_leaf": 50}'

TRAINED_MODEL_DIR="models/7385738_${MODEL_TYPE}"

if [ ! -d "$TRAINED_MODEL_DIR" ]; then
    echo "Error: Trained models not found. Please prepare them first. This script is only for predicting."
    exit 1
fi
if [ ! -f "$TRAINING_POOL_PATH" ]; then
    echo "Error: training_data_pool.csv not found. Please prepare it first."
    exit 1
fi

# --- 2. Prediction Loop ---
echo "Starting prediction loop..."
# make a directory to save N_RUNS predictions
PRED_SAVE_DIR="predictions/${MY_SLURM_PROCESS_ID}_${MODEL_TYPE}"
mkdir -p "$PRED_SAVE_DIR"

for i in $(seq 1 $N_RUNS)
do
  echo "--- PREDICTING WITH MODEL FROM RUN $i ---"
  
  MODEL_PATH="${TRAINED_MODEL_DIR}/pipeline_$(printf %03d ${i}).joblib"
  
  # Predict on holdout negatives
  /usr/bin/time python "$PRED_SCRIPT" \
    --model-path "$MODEL_PATH" \
    --data-path "$TRAINING_POOL_PATH" \
    --out-path "${PRED_SAVE_DIR}/training_pool_preds_$(printf %03d ${i}).csv"
done

# --- 3. Aggregation Step ---
echo "Aggregating all predictions..."
/usr/bin/time python $AGGREGATE_SCRIPT \
  --predictions-dir "predictions" \
  --n-runs "$N_RUNS" \
  --model-type "$MODEL_TYPE" \
  --dataset-name "training_pool" \
  --output-path "final_results/${MY_SLURM_PROCESS_ID}_${MODEL_TYPE}_aggregated_training_pool.csv"

echo "Aggregation done. Final results are in the 'final_results' directory."

/usr/bin/time python $ANALYSIS_SCRIPT \
  --input-csv "final_results/${MY_SLURM_PROCESS_ID}_${MODEL_TYPE}_aggregated_training_pool.csv" \
  --save-summary-path "final_results/${MY_SLURM_PROCESS_ID}_${MODEL_TYPE}_summary_stats_training_pool.csv" \
  --plot-distributions mean std median mad min max range q25 q75 iqr cv \
  --plot-output-path "final_results/${MY_SLURM_PROCESS_ID}_${MODEL_TYPE}_distributions_plot_training_pool.png" \
  --interactive-plot-path "final_results/${MY_SLURM_PROCESS_ID}_${MODEL_TYPE}_interactive_stability_plot_training_pool.html"

echo "Summary stats calculated and graphs made. Output is in the 'final_results/' directory."

# Append the contents of this script to the output file
echo "=== Job Script Contents ==="
cat $0
echo "==========================="

# If we reached this point, we succeeded. We clean up resources.
rm -rf $TMPDIR
