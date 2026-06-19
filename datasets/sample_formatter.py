import csv

sample_filename = "paired_data_sample.csv"
output_tsv_filename = "beautiful_git_sample.tsv"

cleaned_rows = []

with open(sample_filename, "r", encoding="utf-8", errors="ignore") as f:
    reader = csv.reader(f)
    for row in reader:
        # 1. Flatten the text block completely
        cleaned_row = [cell.replace("\n", " ").replace("\r", " ").strip() for cell in row]
        cleaned_rows.append(cleaned_row)

# 2. Write it out using TABS as the separator
with open(output_tsv_filename, "w", encoding="utf-8", newline="") as f:
    writer = csv.writer(f, delimiter="\t")
    writer.writerows(cleaned_rows)

print("Done! Push this .tsv file to GitHub instead.")
