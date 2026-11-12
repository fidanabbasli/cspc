import os
import pandas as pd
import numpy as np
from sklearn.datasets import load_wine
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

os.makedirs("results", exist_ok=True)

print("=== SESSION B: WINE CHEMISTRY ANALYSIS ===")
wine = load_wine(as_frame=True)
X = wine.data
cultivar = wine.target

# B3: Scale and PCA
scaler = StandardScaler()
Z = scaler.fit_transform(X)

pca = PCA()
pca.fit(Z)
print("Explained variance ratio (PC1 & PC2):", pca.explained_variance_ratio_[:2])

# B4 & B5: K-Means Clustering (k=3)
km = KMeans(n_clusters=3, init="k-means++", n_init=10, random_state=0)
km.fit(Z)

labels = km.labels_

# B6: The Reveal & Agreement calculation
contingency = pd.crosstab(cultivar, labels)
agreement = contingency.max(axis=0).sum() / len(cultivar)
print(f"K-Means Agreement with Cultivars: {agreement:.4f}")

# Save agreement table to results/B_agreement.csv
agreement_df = pd.DataFrame({
    "Method": ["Scaled PCA + K-Means"],
    "Agreement": [agreement]
})
agreement_df.to_csv("results/B_agreement.csv", index=False)
print("Agreement table successfully saved to results/B_agreement.csv")
