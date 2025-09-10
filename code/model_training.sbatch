#!/bin/bash
#SBATCH --job-name=tuning124
#SBATCH --output=/lisc/data/scratch/dome/wagner/pullen/embed_classifiers/general_classifier/jobfiles/job%A_%a_%N.out
#SBATCH --error=/lisc/data/scratch/dome/wagner/pullen/embed_classifiers/general_classifier/jobfiles/job%A_%a_%N.err
#SBATCH --mail-type=END,FAIL
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
# #SBATCH --mem-per-cpu=16000M # only for CPU memory
#SBATCH --mem=5000M # only for CPU memory
#SBATCH --time=0-2:00:00
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

#DATA_PATH="/lisc/data/scratch/dome/wagner/pullen/embed_classifiers/general_classifier/data/FDH_EXP_and_no_FINAL.csv"
DATA_PATH="/lisc/data/scratch/dome/wagner/pullen/embed_classifiers/general_classifier/data/training_data_pool.csv"
OUT_PATH="/lisc/data/scratch/dome/wagner/pullen/embed_classifiers/general_classifier/jobfiles/${MY_SLURM_PROCESS_ID}_pred_probs.csv"
MODEL_PATH="/lisc/data/scratch/dome/wagner/pullen/embed_classifiers/general_classifier/models/${MY_SLURM_PROCESS_ID}_{model_name}.joblib"
LABEL_COLUMN_NAME="FDHevidence"

# Run the Python script
/usr/bin/time python /lisc/data/scratch/dome/wagner/pullen/embed_classifiers/general_classifier/code/train.py \
  --data-path "$DATA_PATH" \
  --label-col "$LABEL_COLUMN_NAME" \
  --model rf gb hgb xgb logreg logreg_l2norm \
  --tune hgb xgb \
  --tune-n 12 \
  --tune-seed 123456 \
  --balance \
  --random-state 42 \
  --rf-n-jobs 8 \
  --test-size 0.2 --val-size 0.2 \
  --out-csv "$OUT_PATH" \
  --save-model-path "$MODEL_PATH"

# Append the contents of this script to the output file
echo "=== Job Script Contents ==="
cat $0
echo "==========================="

# If we reached this point, we succeeded. We clean up resources.
rm -rf $TMPDIR

