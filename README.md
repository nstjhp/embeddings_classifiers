## Introduction

We aim to use the protein embeddings generated from the GlobDB for useful science! 

## Data preparation

Starting from the GlobDB embeddings h5 file, your embeddings of interest need to be in CSV file.
They will be 1024 columns wide, the header for them can just be `0,1,2,3,...,1022,1023`.

**Required columns:**
- `protein`: A unique identifier for each protein
- A label column (e.g., `FDHevidence`): Binary labels (0 or 1) for current classification tasks (identified as `label_col` in the code)
- 1024 embeddings columns as above.

**Optional columns:**
- `h5_index`: Index in the original h5 embedding file (will be dropped during training)
- Any other metadata columns (can be dropped via the `load_data` function)

**Note:** The `protein` column is preserved for tracking predictions, while `h5_index` and label columns are dropped before model training.

## Overview of the Entire Pipeline
-----

### **Phase 1: Find the Optimal & Stable Hyperparameters**

**Goal:** Run a rigorous search and find the *single best set* of model hyperparameters that are both high-performing and robust.

  * **Step 1.A: Run Cross-Validated Search**

      * **What you do:** Run a script that trains and tunes the model K times on K different balanced datasets.
      * **Command:** `sbatch ./find_best_hyperparams.sbatch`
      * **What it produces:** A directory in `tuning_logs/` containing detailed log files from each of the K tuning runs.

  * **Step 1.B: Automatically Select the Best Parameter Set**

      * **What you do:** Run a script that reads all the log files, analyses the performance vs. stability trade-off for every parameter set tested and automatically selects the best one. 
The "best" one is based on a utility score of `Mean_PR_AUC - (alpha * Std_Dev_PR_AUC)` where `alpha=1.0` is default (higher `alpha` means you care more about stability i.e. a low standard deviation).
      * **Command:** Separately you can run `python select_best_params.py`, but now it is included in the `sbatch` file automatically.
      * **What it produces:**
        1.  **A JSON string** printed to your console of the best set of hyperparameters for that model.
        2.  An interactive HTML plot (`hyperparameter_tradeoff.html`) to visualise the trade-off and justify the choice.

-----

### **Phase 2: Measure the Stability of the Chosen Model**

**Goal:** Using the single best set of hyperparameters, train the model 50 times on 50 different balanced datasets to see how sensitive its predictions are to the training data.

  * **Step 2.A: Configure the Stability Run (The only manual step)**

      * **What you do:** **Copy** the best parameters' JSON string from the output of Phase 1. **Paste** it into the `FIXED_HYPERPARAMETERS` variable inside the `run_stability_analysis.sbatch` script.

  * **Step 2.B: Execute the Full Stability Experiment**

      * **What you do:** Run the main orchestrator script. This is the longest part of the process, as it trains 50 models, makes predictions with each and then combines all the results.
      * **Command:** `sbatch ./run_stability_analysis.sbatch`. NB this includes calling `aggregate_predictions.py` (and `analyse_stability.py` see step 3)
      * **What it produces:** The final, wide-format CSV files (e.g., `final_results/aggregated_holdout_negatives.csv`), where each row is a protein and each column is the prediction from a different model run.

-----

### **Phase 3: Interpret the Stability Results**

**Goal:** To take the raw stability data from Phase 2 and compute summary statistics and visualisations.

  * **Step 3.A: Analyse and Visualise Stability**
      * **What you do:** Run the final analysis script on the aggregated CSV from Phase 2.
      * **Command:** `python analyse_stability.py` if done separately (included in above `sbatch` script).
      * **What it produces:**
        1.  A CSV file with summary statistics (mean, std, median, etc.) for each protein.
        2.  A plot of the distributions of the summary statistics.
        3.  An interactive plot (e.g. `stability_plot.html`) showing the mean prediction vs. the prediction instability for every protein.

-----

## Troubleshooting

### Common Issues:

1. **Missing columns error in predict.py**
   - Ensure your data has a `protein` column
   - The script automatically skips dropping `h5_index` and `FDHevidence` if they don't exist

2. **Aggregation script can't find prediction files**
   - Check that predictions were generated in a directory with format `predictions/{JOBID}_{MODEL_TYPE}/`
   - Verify file names match pattern: `{dataset}_preds_{i:03d}.csv` (e.g., `holdout_negatives_preds_001.csv`)

3. **Invalid hyperparameters JSON**
   - Ensure the JSON string is properly formatted: `'{"param": value}'`
   - No trailing commas in the JSON object
   - Use proper quoting for the entire string in bash scripts

4. **Tuning log parsing fails**
   - Ensure `find_best_hyperparams.sbatch` completed successfully
   - Check that log files contain the CV results markers: `--- START CV RESULTS ---` and `--- END CV RESULTS ---`
