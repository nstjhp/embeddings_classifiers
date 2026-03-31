## Introduction

We aim to use the protein embeddings generated from the GlobDB for useful science! 

## Data preparation

The script `data_prep.py` was made to help with this.

### Requirements

```bash
pandas h5py numpy
```

### Input File Format (Manifest)

The script expects a **headerless, 3-column TSV** file as follows:

| Column | Name | Description |
| :--- | :--- | :--- |
| 1 | `protein_id` | Unique identifier for the protein. |
| 2 | `h5_index` | The integer index of the embedding within the `.h5` file. |
| 3 | `dataset_tag` | A string tag used for labelling (e.g., `P1`, `N1`, `M1`) for positive, negative or putative protein sets. |

**Example:**
```text
PROT_A1    1205    P1
PROT_B2    5502    N1
PROT_C3    9910    M1
```

### Usage and Arguments

| Argument | Description |
| :--- | :--- |
| `--input-tsv` | Path to your 3-column manifest file. |
| `--h5-path` | Path to the `.h5` file containing the embeddings. |
| `--out-train` | Output path for the training pool (Positives + remaining Negatives). |
| `--out-putative` | Output path for putative proteins (labeled with `?`). |
| `--pos-tags` | List of tags to treat as Positives (default: `P1 P2`). |
| `--neg-tags` | List of tags to treat as Negatives (default: `N1 N2`). |
| `--putative-tags` | List of tags to treat as Putatives (default: `M1 M2`). |
| `--neg-sample-frac` | Fraction of negatives to reserve for the independent holdout set (e.g., `0.5`). |
| `--out-holdout` | Path to save the holdout negatives (required if using sampling). |

### Execution (SLURM)

Due to filesystem permissions where the `.h5` file is not be accessible to the compute nodes via `sbatch`, you should use `salloc` to run the preparation script on the login nodes. 
This ensures the job runs with the necessary interactive environment access.

**Example command run from the project root:**

```bash
salloc --job-name=nar_data_prep --ntasks=1 --cpus-per-task=1 --mem=3G -t 00:30:00 code/run_data_prep.sbatch
```

> **Note:** The `run_data_prep.sbatch` wrapper script should contain the specific `python data_prep.py ...` call with your desired paths and tags.

### Timings

Running the above line to extract 73199 embeddings and write the corresponding CSVs took ~210 s.

### Warnings and Errors

* **Duplicate Detection:** If duplicate protein IDs or H5 indices are present in your input file you should get at least a warning and possibly an error if it is serious. Exact duplicates across the 3 columns are dropped, but others are usually an error to fix in your data files e.g. same protein and index but different dataset tags.
* **Failure Auditing:** If an index is out of bounds or an embedding is corrupted, the script logs the specific `protein_id` to a `.extraction_failures.csv` file.
* **Tag Validation:** Checks for no overlap between your Positive, Negative, and Putative tag definitions.

### Output

2 or 3 output CSV files are produced for downstream ML:
| File | Description |
| :--- | :--- |
| `Training pool` | Positives + Negatives for use in model training |
| `Putative set` | The maybes we want to predict as pos/neg (labelled with `?`). |
| `Negative holdout set (Optional)` | Due to imbalance we normally have many more known negative (for our property of interest) proteins. We can holdout a `--neg-sample-frac` of these that will never be used in training as a negative control group. |

The columns for the embeddings will be 1024 columns wide, the header for them can just be `0,1,2,3,...,1022,1023`.
We need other metadata columns as well as follows:

**Required columns:**
- `protein`: A unique identifier for each protein
- A label column (e.g., `label`): Binary labels (0 or 1) for current classification tasks (identified as `label_col` in the code). Following the `data_prep.py` script will give a `?` to the putative set, which is fine as this dataset is obviously ignored for training.
- 1024 embeddings columns as above.

**Optional columns:**
- `h5_index`: Index in the original h5 embedding file (will be dropped during training)
- Any other metadata columns (can be dropped via the `load_data` function)

> **Note:** The `protein` column is preserved for tracking predictions, while `h5_index` and label columns are dropped before model training.

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
