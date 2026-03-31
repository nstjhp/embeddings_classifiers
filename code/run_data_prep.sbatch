#!/bin/bash
#SBATCH --job-name=data_prep
#SBATCH --output=/lisc/data/scratch/dome/wagner/pullen/embed_classifiers/general_classifier/jobfiles/job%A_%a_%N.out
#SBATCH --error=/lisc/data/scratch/dome/wagner/pullen/embed_classifiers/general_classifier/jobfiles/job%A_%a_%N.err
#SBATCH --mail-type=END,FAIL
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH -t 00:30:00
#SBATCH --partition=basic

# Exit the slurm script if a command fails, there's an unset variable or hidden failures inside pipelines
set -euo pipefail

# JOB_PARAM_STRING="${SLURM_JOB_ID}_${SLURMD_NODENAME}_${SLURM_CPUS_PER_TASK}cpus"
JOB_PARAM_STRING="${SLURM_JOB_ID}_${SLURM_CPUS_PER_TASK}cpus"
# These commands will go into stdout
echo "Job ID: ${SLURM_JOB_ID}"
# echo "TMPDIR: ${TMPDIR}"
# echo "Node: ${SLURMD_NODENAME}"
echo "CPUs per task: ${SLURM_CPUS_PER_TASK}"
echo "JOB_PARAM_STRING: $JOB_PARAM_STRING"
export MY_SLURM_PROCESS_ID="${SLURM_JOB_ID}"
echo "MY_SLURM_PROCESS_ID: ${MY_SLURM_PROCESS_ID}"

####################
# edit these paths #
####################
DIR_NAME="NAR"
LABEL_NAME="nar"
MAIN_DIR="/lisc/data/scratch/dome/wagner/pullen/embed_classifiers/general_classifier"
INPUT_TSV="${MAIN_DIR}/data/${DIR_NAME}/test/${LABEL_NAME}_test_combined.tsv"
OUT_TRAIN="${MAIN_DIR}/data/${DIR_NAME}/test/${MY_SLURM_PROCESS_ID}_${LABEL_NAME}_POS_NEG_embeddings.csv"
OUT_HOLDOUT="${MAIN_DIR}/data/${DIR_NAME}/test/${MY_SLURM_PROCESS_ID}_${LABEL_NAME}_holdout_negatives.csv"
OUT_PUTATIVE="${MAIN_DIR}/data/${DIR_NAME}/test/${MY_SLURM_PROCESS_ID}_${LABEL_NAME}_PUTATIVE_embeddings.csv"
# NB The /lisc/fileshare filesystem is available on interactive nodes, but not on compute nodes. Files that are needed for computing need to be staged from/to /lisc/data/work or /lisc/data/scratch
# Consider doing this instead to run on login nodes
# /usr/bin/time salloc --job-name=data_prep --ntasks=1 --cpus-per-task=1 --mem=3G -t 00:30:00 code/run_data_prep.sbatch
H5_PATH="/lisc/fileshare/globdb/globdb_r226/pLM_embeddings/globdb_r226_linclust_2plus_ProtT5_embeddings.h5"
DATA_PREP_SCRIPT="${MAIN_DIR}/code/data_prep.py"

python "$DATA_PREP_SCRIPT" \
  --input-tsv "$INPUT_TSV" \
  --h5-path "$H5_PATH" \
  --out-train "$OUT_TRAIN" \
  --out-putative "$OUT_PUTATIVE" \
  --pos-tags Pos \
  --neg-tags Neg \
  --putative-tags Maybe \
  --neg-sample-frac 0.5 \
  --out-holdout "$OUT_HOLDOUT" \
  --random-state 42

# Append the contents of this script to the output file
echo "=== Job Script Contents ==="
cat $0
echo "==========================="

# If we reached this point, we succeeded. We clean up resources.
# rm -rf $TMPDIR

