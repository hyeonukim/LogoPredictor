import pandas as pd

input_file = "logos_rewritten.csv"
output_file = "top10k_logos.csv"
rank_column = "rank"                      
chunksize = 100_000                  

top10k = pd.DataFrame()

for chunk in pd.read_csv(input_file, chunksize=chunksize):
    chunk[rank_column] = pd.to_numeric(chunk[rank_column], errors='coerce')
    chunk = chunk.dropna(subset=[rank_column])
    chunk_top = chunk.nsmallest(10_000, rank_column)
    top10k = pd.concat([top10k, chunk_top])
    top10k = top10k.nsmallest(10_000, rank_column)

top10k.to_csv(output_file, index=False)

print(f"Saved top 10,000 rows (lowest {rank_column}) to {output_file}")
print(top10k[rank_column].describe())
