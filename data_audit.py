import pandas as pd

df = pd.read_csv("my_own_data.csv", on_bad_lines='skip')

# Normalize text for analysis
df["query_lower"] = df["query"].str.lower()

# 1. Label distribution per concept keyword
keywords = ["bfs", "dfs", "quicksort", "dijkstra", "recursion", "binary search"]

for k in keywords:
    subset = df[df["query_lower"].str.contains(k)]
    print("\n====================")
    print("Concept:", k)
    print(subset["label"].value_counts())
