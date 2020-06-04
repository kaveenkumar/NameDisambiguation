import json
import pandas as pd
import numpy as np

with open("result.txt", "rb") as file:
    results = pd.read_table(file, sep=",", header=None)

    targets = results[results.columns[16]]

    posCounter = targets.sum()
    negCounter = len(targets) - posCounter

    results.sort_values(by=results.columns[16], inplace=True, ascending = False)

    sampled = results.head(posCounter*2)

    targets = sampled[sampled.columns[16]]

    posCounter = targets.sum()
    negCounter = len(targets) - posCounter

    sampled.replace({r'[^\x00-\x7F]+': ''}, regex=True, inplace=True)
    sampled.sort_values(by=sampled.columns[17], inplace=True)


with open("resultsSampled.txt", "w", newline = '\n') as file:
    sampled.to_csv(file, sep=',', index=False, header=False)
