def check_source_coverage(
    dataset_filepath, source_filepath, output_filepath="coverage_report.txt"
):

    # 1. Extract unique root IDs from the main dataset
    dataset_root_ids = set()
    with open(dataset_filepath, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) > 0:
                # The first column is the root ID
                dataset_root_ids.add(parts[0])

    # 2. Extract IDs from the source_tweets.tsv file
    source_ids = set()
    with open(source_filepath, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) > 0:
                # Assuming the first column is the ID, based on your prompt
                source_ids.add(parts[0])

    # 3. Calculate intersections and differences
    # intersection() finds IDs present in BOTH sets
    matched_ids = dataset_root_ids.intersection(source_ids)

    # difference() finds IDs in the dataset that are MISSING from the source file
    missing_ids = dataset_root_ids.difference(source_ids)

    # 4. Compute the coverage metrics
    total_dataset_roots = len(dataset_root_ids)
    total_matched = len(matched_ids)
    total_missing = len(missing_ids)

    if total_dataset_roots == 0:
        print("Error: No IDs found in the dataset file.")
        return

    coverage_percentage = (total_matched / total_dataset_roots) * 100

    # 5. Print results to the console
    print("--- Coverage Summary ---")
    print(f"Total Unique Cascades in Dataset: {total_dataset_roots}")
    print(f"Matched in source_tweets.tsv:     {total_matched}")
    print(f"Missing from source_tweets.tsv:   {total_missing}")
    print(f"Overall Coverage:                 {coverage_percentage:.2f}%")

    # 6. Write the results to a new file
    with open(output_filepath, "w", encoding="utf-8") as out_f:
        out_f.write("=== COVERAGE REPORT ===\n")
        out_f.write(f"Total Unique Cascades: {total_dataset_roots}\n")
        out_f.write(f"Matched: {total_matched}\n")
        out_f.write(f"Missing: {total_missing}\n")
        out_f.write(f"Coverage: {coverage_percentage:.2f}%\n\n")

        out_f.write("=== MISSING IDs ===\n")
        if total_missing == 0:
            out_f.write("None! 100% Coverage.\n")
        else:
            for missing_id in missing_ids:
                out_f.write(f"{missing_id}\n")


if __name__ == "__main__":
    # Ensure your filenames match what is on your system
    DATASET_FILE = "BiGCN/data/Twitter15/data.TD_RvNN.vol_5000.txt"
    SOURCE_FILE = "source_tweets_t15.tsv"

    check_source_coverage(DATASET_FILE, SOURCE_FILE)
