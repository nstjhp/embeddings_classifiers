import argparse
from collections import Counter
import pandas as pd
import h5py


def _looks_like_integer(value: str) -> bool:
    try:
        int(str(value))
        return True
    except (TypeError, ValueError):
        return False


def _validate_tsv_shape_and_header(input_tsv: str) -> None:
    """Defensive check for a 3-column, headerless TSV."""
    rows = []
    with open(input_tsv, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.rstrip("\n")
            if not line.strip():
                continue
            rows.append(line.split("\t"))
            if len(rows) == 2:
                break

    if not rows:
        raise ValueError("Input TSV is empty.")

    if len(rows[0]) != 3:
        raise ValueError(
            f"Input TSV must have exactly 3 tab-separated columns, but first non-empty row has {len(rows[0])}."
        )

    if len(rows) > 1 and len(rows[1]) != 3:
        raise ValueError(
            f"Input TSV must have exactly 3 tab-separated columns, but second non-empty row has {len(rows[1])}."
        )

    first_idx_ok = _looks_like_integer(rows[0][1])
    second_idx_ok = True if len(rows) == 1 else _looks_like_integer(rows[1][1])

    if not first_idx_ok and second_idx_ok:
        raise ValueError(
            "The TSV appears to contain a header row. This script expects a headerless 3-column TSV: "
            "protein_id, h5_index, dataset_tag."
        )
    if not first_idx_ok:
        raise ValueError(
            "The second field in the first non-empty row is not an integer h5_index. "
            "Please provide a headerless 3-column TSV."
        )


def main():
    parser = argparse.ArgumentParser(description="Extract embeddings and prep datasets.")
    parser.add_argument("--input-tsv", type=str, required=True,
                        help="Path to 3-column TSV: [protein_id, h5_index, dataset_tag]")
    parser.add_argument("--h5-path", type=str, required=True,
                        help="Path to the .h5 embeddings file.")
    parser.add_argument("--out-train", type=str, required=True,
                        help="Output path for the training CSV (Positives & Negatives pool).")
    parser.add_argument("--out-putative", type=str, required=True,
                        help="Output path for the putatives CSV.")

    # Label definitions
    parser.add_argument("--pos-tags", nargs='+', default=['P1', 'P2'],
                        help="Dataset tags that represent Positives (label 1).")
    parser.add_argument("--neg-tags", nargs='+', default=['N1', 'N2'],
                        help="Dataset tags that represent Negatives (label 0).")
    parser.add_argument("--putative-tags", nargs='+', default=['M1', 'M2'],
                        help="Dataset tags that represent Putatives (label '?').")

    # Holdout splitting
    parser.add_argument("--neg-sample-frac", type=float, default=None,
                        help="Optional. Fraction of negatives to hold out (e.g., 0.5).")
    parser.add_argument("--out-holdout", type=str, default=None,
                        help="Output path for the holdout negatives CSV. Required if --neg-sample-frac is used.")
    parser.add_argument("--random-state", type=int, default=42,
                        help="Random seed for negative sampling.")

    args = parser.parse_args()

    if args.neg_sample_frac is not None and args.out_holdout is None:
        parser.error("--out-holdout is required when --neg-sample-frac is specified.")
    if args.neg_sample_frac is not None and not (0.0 <= args.neg_sample_frac <= 1.0):
        parser.error("--neg-sample-frac must be between 0 and 1 inclusive.")

    pos_set = set(args.pos_tags)
    neg_set = set(args.neg_tags)
    putative_set = set(args.putative_tags)
    overlap = (pos_set & neg_set) | (pos_set & putative_set) | (neg_set & putative_set)
    if overlap:
        parser.error(f"Tag categories overlap, which is ambiguous: {sorted(overlap)}")

    print(f"Loading input TSV: {args.input_tsv}")
    _validate_tsv_shape_and_header(args.input_tsv)
    df = pd.read_csv(
        args.input_tsv,
        sep='\t',
        header=None,
        names=['protein', 'h5_index', 'dataset_tag'],
        dtype=str,
    )

    total_queries = len(df)
    print(f"Total proteins in manifest: {total_queries}")

    df = df.copy()
    df["protein"] = df["protein"].astype(str)
    df["h5_index_raw"] = df["h5_index"]
    df["dataset_tag"] = df["dataset_tag"].astype(str)
    df["status"] = "ok"
    df["fail_reason"] = ""

    # 0. DUPLICATE CHECKS
    # 1) Exact full-row repeats: warn and drop duplicates
    exact_dupes = df.duplicated(subset=["protein", "h5_index", "dataset_tag"], keep=False)
    n_exact_dupes = int(exact_dupes.sum())
    if n_exact_dupes > 0:
        print(f"WARNING: Found {n_exact_dupes} rows that are exact duplicates of protein+h5_index+dataset_tag.")
        print(df.loc[exact_dupes, ["protein", "h5_index", "dataset_tag"]].head(10).to_string(index=False))
        if n_exact_dupes > 10:
            print(f"  ... and {n_exact_dupes - 10} more exact-duplicate rows.")

        before = len(df)
        df = df.drop_duplicates(subset=["protein", "h5_index", "dataset_tag"], keep="first").copy()
        print(f"Dropped {before - len(df)} exact duplicate rows.")
    
    # 2) Same protein with different h5_index: hard error
    protein_multi_idx = (
        df.groupby("protein")["h5_index"]
          .nunique(dropna=False)
          .reset_index(name="n_h5_index")
    )
    bad_proteins_idx = protein_multi_idx.loc[protein_multi_idx["n_h5_index"] > 1, "protein"]
    if len(bad_proteins_idx) > 0:
        bad_rows = df.loc[df["protein"].isin(bad_proteins_idx), ["protein", "h5_index", "dataset_tag"]]
        raise ValueError(
            "Found protein IDs associated with multiple different h5_index values. "
            "This is not allowed.\n"
            + bad_rows.head(20).to_string(index=False)
        )
    
    # 3) Same protein+h5_index but different dataset_tag: hard error
    protein_idx_multi_tag = (
        df.groupby(["protein", "h5_index"])["dataset_tag"]
          .nunique(dropna=False)
          .reset_index(name="n_dataset_tag")
    )
    bad_pairs_tag = protein_idx_multi_tag.loc[protein_idx_multi_tag["n_dataset_tag"] > 1, ["protein", "h5_index"]]
    if len(bad_pairs_tag) > 0:
        bad_rows = df.merge(bad_pairs_tag, on=["protein", "h5_index"], how="inner")
        raise ValueError(
            "Found protein+h5_index pairs associated with multiple different dataset_tag values. "
            "This is not allowed - please check your input data files.\n"
            + bad_rows[["protein", "h5_index", "dataset_tag"]].head(20).to_string(index=False)
        )
    
    # 4) Same h5_index used by multiple different proteins: hard error
    idx_multi_protein = (
        df.groupby("h5_index")["protein"]
          .nunique(dropna=False)
          .reset_index(name="n_protein")
    )
    bad_indices = idx_multi_protein.loc[idx_multi_protein["n_protein"] > 1, "h5_index"]
    if len(bad_indices) > 0:
        bad_rows = df.loc[df["h5_index"].isin(bad_indices), ["protein", "h5_index", "dataset_tag"]]
        raise ValueError(
            "Found h5_index values associated with multiple different protein IDs. "
            "This is not allowed.\n"
            + bad_rows.head(20).to_string(index=False)
        )

    # 1. DATA EXTRACTION
    print(f"Opening H5 file: {args.h5_path}")
    with h5py.File(args.h5_path, 'r') as f:
        emb_ds = f['embeddings']
        max_idx = emb_ds.shape[0] - 1

        # Validate h5_index defensively while keeping traceability per row
        parsed_idx = pd.to_numeric(df['h5_index'], errors='coerce')
        non_integer_mask = parsed_idx.isna()
        df.loc[non_integer_mask, 'status'] = 'failed'
        df.loc[non_integer_mask, 'fail_reason'] = 'Invalid non-integer index'

        valid_integer_mask = ~non_integer_mask
        parsed_idx_int = pd.Series(index=df.index, dtype='Int64')
        parsed_idx_int.loc[valid_integer_mask] = parsed_idx.loc[valid_integer_mask].astype('Int64')
        df['h5_index'] = parsed_idx_int

        in_bounds_mask = valid_integer_mask & df['h5_index'].between(0, max_idx, inclusive='both')
        out_of_bounds_mask = valid_integer_mask & ~in_bounds_mask
        df.loc[out_of_bounds_mask, 'status'] = 'failed'
        df.loc[out_of_bounds_mask, 'fail_reason'] = 'Index out of bounds'

        valid_df = df[df['status'] == 'ok'].copy()

        extracted_indices = []
        extracted_embeddings = []
        if len(valid_df) > 0:
            try:
                index_array = valid_df["h5_index"].astype(np.int64, copy=False).to_numpy()

                # HDF5 fancy indexing is safest with sorted, unique indices.
                # row_to_unique_idx lets us reconstruct the embeddings in the original row order.
                unique_h5_indices, row_to_unique_idx = np.unique(index_array, return_inverse=True)
            
                # Fast bulk extraction on sorted unique indices
                bulk_unique_embeddings = emb_ds[unique_h5_indices, :]
            
                # Re-expand back to the original manifest order
                bulk_embeddings = bulk_unique_embeddings[row_to_unique_idx]
                extracted_indices = valid_df.index.tolist()
                extracted_embeddings = bulk_embeddings
            except Exception as bulk_err:
                print(f"Bulk extraction failed ({bulk_err}). Falling back to row-wise extraction for diagnosis...")
                extracted_indices = []
                extracted_embeddings = []
                for row_idx, row in valid_df.iterrows():
                    try:
                        extracted_indices.append(row_idx)
                        extracted_embeddings.append(emb_ds[int(row['h5_index']), :])
                    except Exception as e:
                        df.loc[row_idx, 'status'] = 'failed'
                        df.loc[row_idx, 'fail_reason'] = f"H5 Error: {str(e)}"

    failed_df = df[df['status'] != 'ok'].copy()
    valid_df = df[df['status'] == 'ok'].copy()

    if len(extracted_indices) > 0:
        emb_df = pd.DataFrame(extracted_embeddings, index=extracted_indices)
        full_df = pd.concat([valid_df.loc[extracted_indices], emb_df], axis=1)
    else:
        full_df = valid_df.copy()

    print(f"Successfully extracted embeddings for {len(full_df)} rows.")
    if len(failed_df) > 0:
        print(f"Failed to extract {len(failed_df)} rows.")
        reason_counts = Counter(failed_df['fail_reason'])
        for reason, count in reason_counts.items():
            print(f"  - {reason}: {count}")
        preview = failed_df[['protein', 'h5_index_raw', 'dataset_tag', 'fail_reason']].head(10)
        print("  Example failed rows:")
        print(preview.to_string(index=False))
        # Save the failures to disk for easy auditing
        failed_out_path = args.input_tsv + ".extraction_failures.csv"
        failed_df[['protein', 'h5_index_raw', 'dataset_tag', 'fail_reason']].to_csv(failed_out_path, index=False)
        print(f"  Full list of failures saved to: {failed_out_path}")
    print()

    # 2. MERGE AND LABEL DATA
    def assign_label(tag):
        if tag in pos_set:
            return 1
        if tag in neg_set:
            return 0
        if tag in putative_set:
            return '?'
        return 'ignored'

    full_df['label'] = full_df['dataset_tag'].apply(assign_label)

    # Reorder columns: protein, h5_index, dataset_tag, label, [features...]
    cols = ['protein', 'h5_index', 'dataset_tag', 'label']
    feature_cols = [c for c in full_df.columns if c not in cols + ['h5_index_raw', 'status', 'fail_reason']]
    full_df = full_df[cols + feature_cols]

    # 3. SPLIT DATASETS
    ignored_df = full_df[full_df['label'] == 'ignored'].copy()
    putative_df = full_df[full_df['label'] == '?'].copy()
    train_base_df = full_df[full_df['label'].isin([0, 1])].copy()

    # Ensure label is int for ML downstream in the training sets
    train_base_df['label'] = train_base_df['label'].astype(int)

    if len(ignored_df) > 0:
        print(f"Ignored rows due to dataset tags not selected this run: {len(ignored_df)}")
        ignored_counts = ignored_df['dataset_tag'].value_counts().to_dict()
        print(f"  Ignored tag counts: {ignored_counts}")
        print()

    # 4. NEGATIVE SAMPLING & HOLDOUT (If requested)
    if args.neg_sample_frac is not None:
        print(f"Performing negative sampling (frac={args.neg_sample_frac})...")
        positives = train_base_df[train_base_df['label'] == 1]
        negatives = train_base_df[train_base_df['label'] == 0]

        holdout_df = negatives.sample(frac=args.neg_sample_frac, random_state=args.random_state)
        training_negs = negatives.drop(holdout_df.index)

        # Final training pool is positives + remaining negatives
        final_train_df = pd.concat([positives, training_negs]).sample(frac=1, random_state=args.random_state).reset_index(drop=True)

        print(f"Saving Holdout Negatives (N={len(holdout_df)}) to {args.out_holdout}")
        holdout_df.to_csv(args.out_holdout, index=False)
    else:
        # If no split requested, the final training pool is everything
        final_train_df = train_base_df.sample(frac=1, random_state=args.random_state).reset_index(drop=True)

    # 5. SAVE FINAL OUTPUTS
    print("Final row counts:")
    print(f"  - Input manifest rows: {total_queries}")
    print(f"  - Successfully extracted rows: {len(full_df)}")
    print(f"  - Failed/missing rows: {len(failed_df)}")
    print(f"  - Ignored rows: {len(ignored_df)}")
    print(f"  - Training pool rows: {len(final_train_df)}")
    print(f"  - Putative rows: {len(putative_df)}")
    print()

    print(f"Saving Training Pool (N={len(final_train_df)}) to {args.out_train}")
    final_train_df.to_csv(args.out_train, index=False)

    if len(putative_df) > 0:
        print(f"Saving Putative set (N={len(putative_df)}) to {args.out_putative}")
        putative_df.to_csv(args.out_putative, index=False)
    else:
        print("No putative sequences found.")

    print("Data preparation complete!")


if __name__ == "__main__":
    main()
