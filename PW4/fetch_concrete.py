import os
import pandas as pd
from ucimlrepo import fetch_ucirepo

concrete = fetch_ucirepo(id=165)
X = concrete.data.features.copy()
y = concrete.data.targets

X.columns = [
    "cement", "slag", "ash",
    "coarse_agg", "fine_agg",
    "water", "superplastic",
    "age"
]
y = y.iloc[:, 0].rename("strength")

os.makedirs("data", exist_ok=True)
X.assign(strength=y).to_csv("data/concrete.csv", index=False)

print("--- Metadata ---")
print(concrete.metadata)
print("\n--- Variables ---")
print(concrete.variables)
