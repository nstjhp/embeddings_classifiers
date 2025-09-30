#!/bin/bash
#SBATCH --job-name=find_hyperparams
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

TRAINING_POOL_PATH="/lisc/data/scratch/dome/wagner/pullen/embed_classifiers/general_classifier/data/training_data_pool.csv"
OUT_PATH="/lisc/data/scratch/dome/wagner/pullen/embed_classifiers/general_classifier/jobfiles/${MY_SLURM_PROCESS_ID}_pred_probs.csv"
MODEL_PATH="/lisc/data/scratch/dome/wagner/pullen/embed_classifiers/general_classifier/models/${MY_SLURM_PROCESS_ID}_{model_name}.joblib"
LABEL_COLUMN_NAME="FDHevidence"
TRAINING_SCRIPT="/lisc/data/scratch/dome/wagner/pullen/embed_classifiers/general_classifier/code/train.py"
HYPERPARAM_SELECTION_SCRIPT="/lisc/data/scratch/dome/wagner/pullen/embed_classifiers/general_classifier/code/select_best_params.py"
RESULTS_DIR="/lisc/data/scratch/dome/wagner/pullen/embed_classifiers/general_classifier/final_results"

# --- Configuration ---
N_TUNING_RUNS=10 # The number of different balanced datasets to tune on
MODEL_TYPE="hgb" # The model to find parameters for
TUNE_ITERATIONS=50 # Number of random search iterations per run

# Directory to store the logs from each tuning run
LOG_DIR="/lisc/data/scratch/dome/wagner/pullen/embed_classifiers/general_classifier/tuning_logs/${MY_SLURM_PROCESS_ID}"
mkdir -p "$LOG_DIR"

echo "--- Cross-Validated Hyperparameter Search ---"
echo "Starting search across $N_TUNING_RUNS different balanced datasets..."
echo "Logs for each run will be stored in the '$LOG_DIR' directory."
echo "This may take a significant amount of time..."

# --- 1. Tuning Loop ---
for i in $(seq 1 $N_TUNING_RUNS)
do
  echo
  echo "--- Tuning Run $i / $N_TUNING_RUNS ---"
  LOG_FILE="${LOG_DIR}/tuning_run_${MY_SLURM_PROCESS_ID}_${MODEL_TYPE}_$(printf %03d ${i}).log"
  
  # Run the training script in tuning mode.
  # IMPORTANT: use the loop variable 'i' as the random_state to ensure
  # that each run uses a DIFFERENT balanced sample of the data.
  /usr/bin/time python "$TRAINING_SCRIPT" \
    --data-path "$TRAINING_POOL_PATH" \
    --label-col "$LABEL_COLUMN_NAME" \
    --model "$MODEL_TYPE" \
    --balance \
    --random-state "$i" \
    --rf-n-jobs 8 \
    --test-size 0.2 --val-size 0.2 \
    --tune "$MODEL_TYPE" --tune-n "$TUNE_ITERATIONS" --tune-seed 123 \
    > "$LOG_FILE"
    
  echo "Run $i complete. See log: $LOG_FILE"
done

# --- 2. Aggregation and Analysis ---
# not strictly necessary - may remove in future
echo
echo "--- AGGREGATED TUNING RESULTS ---"
echo "Parsing results from all log files..."
echo "---------------------------------------------------------------------"
echo "Run # | Best Validation PR-AUC | Best Hyperparameters"
echo "---------------------------------------------------------------------"

# Use grep to find the relevant lines in the log files and display them.
for i in $(seq 1 $N_TUNING_RUNS)
do
    LOG_FILE="${LOG_DIR}/tuning_run_${MY_SLURM_PROCESS_ID}_${MODEL_TYPE}_$(printf %03d ${i}).log"
    
    # Extract the PR-AUC score and the parameters string
    PR_AUC=$(grep "Best tuning PR-AUC" "$LOG_FILE" | awk '{print $NF}')
    PARAMS=$(grep "Found best params" "$LOG_FILE" | sed 's/Found best params: //')

    printf "%-5s | %-24s | %s\n" "$i" "$PR_AUC" "$PARAMS"
done
echo "---------------------------------------------------------------------"

# --- 3. Select best hyperparameters ---
echo "Selecting best parameters for model ${MODEL_TYPE} from job ${MY_SLURM_PROCESS_ID}"
/usr/bin/time python "$HYPERPARAM_SELECTION_SCRIPT" \
  --log-dir "$LOG_DIR" \
  --alpha 1.0 \
  --plot-path "${RESULTS_DIR}/${MY_SLURM_PROCESS_ID}_${MODEL_TYPE}_hyperparameter_tradeoff.html"
echo
echo 'Copy JSON dictionary into `run_stability_analysis.sh` script (e.g. as --hyperparams "$FIXED_HYPERPARAMETERS")'
echo "See interactive plot at ${RESULTS_DIR}/${MY_SLURM_PROCESS_ID}_${MODEL_TYPE}_hyperparameter_tradeoff.html"
echo "---------------------------------------------------------------------"

# Append the contents of this script to the output file
echo "=== Job Script Contents ==="
cat $0
echo "==========================="

# If we reached this point, we succeeded. We clean up resources.
rm -rf $TMPDIR
