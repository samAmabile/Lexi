import pandas as pd

df_full = pd.read_csv("paired_data_master.csv")

#TODO: parse the sample dataset into an html table all pretty like 
df = df_full.head(50).copy()

html = df.to_html(index=False, classes='table table-striped')

with open("data_sample_table.html", 'w', encoding='utf-8') as f:
    f.write(html)


