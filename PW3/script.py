import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import glob

print("=== DATASET 1: HEART DISEASE (CLEVELAND) ===")

cols = ['age','sex','cp','trestbps','chol','fbs','restecg','thalach','exang','oldpeak','slope','ca','thal','target']

csv_files = glob.glob("*.csv")
heart_csv = [f for f in csv_files if "heart" in f.lower() or "cleveland" in f.lower()]

if heart_csv:
    df_heart = pd.read_csv(heart_csv[0])
else:
    data_files = glob.glob("*.data")
    df_heart = pd.read_csv(data_files[0], header=None, names=cols, sep=r'\s+', na_values='?')

for col in df_heart.columns:
    df_heart[col] = pd.to_numeric(df_heart[col], errors='coerce')

if 'target' in df_heart.columns:
    df_heart['target'] = df_heart['target'].apply(lambda x: 1 if x > 0 else 0)

print("Dataset Shape:", df_heart.shape)

cont_cols = ["age", "chol", "trestbps", "thalach"]

fig, axes = plt.subplots(2, 2, figsize=(10, 8))
axes = axes.flatten()
for i, col in enumerate(cont_cols):
    axes[i].hist(df_heart[col].dropna(), bins=20, edgecolor='black', color='steelblue')
    axes[i].set_title(f"Distribution: {col}")
plt.tight_layout()
plt.savefig("heart_distributions.png")

print("\n--- Normality Test (Shapiro-Wilk) ---")
for col in cont_cols:
    data_clean = df_heart[col].dropna()
    stat, p_val = stats.shapiro(data_clean)
    print(f"{col}: W={stat:.4f}, p={p_val:.5f} -> Normal? {'Yes' if p_val > 0.05 else 'No'}")

disease_thalach = df_heart[df_heart['target'] == 1]['thalach'].dropna()
healthy_thalach = df_heart[df_heart['target'] == 0]['thalach'].dropna()

t_stat, p_val_tt = stats.ttest_ind(disease_thalach, healthy_thalach)
print(f"\nThalach T-test p-value = {p_val_tt:.5e}")

mean_dis, se_dis = disease_thalach.mean(), stats.sem(disease_thalach)
mean_hlth, se_hlth = healthy_thalach.mean(), stats.sem(healthy_thalach)
print(f"Disease Group: Mean = {mean_dis:.2f} ± {se_dis:.2f} (SE)")
print(f"Healthy Group: Mean = {mean_hlth:.2f} ± {se_hlth:.2f} (SE)")

plt.figure(figsize=(6, 4))
plt.errorbar(["Disease", "Healthy"], [mean_dis, mean_hlth], yerr=[se_dis, se_hlth], fmt='o', capsize=6, color='crimson', markersize=8)
plt.title("Mean Max Heart Rate (thalach) ± SE")
plt.ylabel("thalach (bpm)")
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.savefig("thalach_means.png")

valid_age_thalach = df_heart[['age', 'thalach']].dropna()
r_pe, p_pe = stats.pearsonr(valid_age_thalach['age'], valid_age_thalach['thalach'])
print(f"\nAge vs Thalach Correlation -> Pearson r={r_pe:.3f} (p={p_pe:.5e})")

print("\n=== DATASET 2: CHEMICAL EXPOSURE MYSTERY ===")
df_chem = pd.read_csv("chemicals_cancer.csv")

r_benz, p_benz = stats.pearsonr(df_chem['benzene'], df_chem['malignancy'])
r_cadm, p_cadm = stats.pearsonr(df_chem['cadmium'], df_chem['malignancy'])
print(f"Naive Benzene vs Malignancy: r={r_benz:.3f}")
print(f"Naive Cadmium vs Malignancy: r={r_cadm:.3f}")

df_res = df_chem[(df_chem['pollution_index'] > 40) & (df_chem['pollution_index'] < 60)]
r_benz_res, _ = stats.pearsonr(df_res['benzene'], df_res['malignancy'])
r_cadm_res, _ = stats.pearsonr(df_res['cadmium'], df_res['malignancy'])
print(f"Controlled Benzene correlation: r={r_benz_res:.3f}")
print(f"Controlled Cadmium correlation: r={r_cadm_res:.3f}")

print("\n=== BONUS: TARGET BALANCE ===")
probs = df_heart['target'].value_counts(normalize=True)
entropy = -np.sum(probs * np.log2(probs))
print(probs)
print(f"Shannon Entropy: {entropy:.4f} bits")
