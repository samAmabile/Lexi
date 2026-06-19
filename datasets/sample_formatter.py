import csv
import pandas as pd

sample_filename = "paired_data_sample.csv"
output_tsv_filename = "beautiful_git_sample.tsv"

df = pd.read_csv("paired_data_master.csv")

sample = df.head(50)

sample.to_csv(sample_filename, sep='|', index=False)


