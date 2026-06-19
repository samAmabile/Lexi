import pandas as pd
import csv

sample_filename = "paired_data_sample.csv"
clean_filename = "paired_data_sample_git.csv"

cleaned_rows = []

# Open your current sample file
with open(sample_filename, "r", encoding="utf-8", errors="ignore") as f:
    # csv.reader handles the remaining valid quotes correctly
    reader = csv.reader(f)
    
    for row in reader:
        # Strip internal newlines from every cell so GitHub doesn't panic
        cleaned_row = [cell.replace("\n", " ").replace("\r", " ").strip() for cell in row]
        cleaned_rows.append(cleaned_row)

# Write it back out as a perfectly uniform CSV
with open(clean_filename, "w", encoding="utf-8", newline="") as f:
    writer = csv.writer(f)
    writer.writerows(cleaned_rows)

print("Done! Check beautiful_git_sample.csv") 
