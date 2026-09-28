## PW1 --- Lab B

### Data vs Analytical Model Report
The observed decay data follows an exponential decay trend. When compared side-by-side with the analytical law $N(t) = N_0 e^{-\lambda t}$ using $\lambda = 0.3$ and $N_0 = 5000$, the shapes of both curves match closely, demonstrating that the real observation closely aligns with theoretical predictions.

### Snakemake Automation
The Snakemake pipeline automatically tracks the timestamps of `decay_observed.csv` and `plot.py` to regenerate `figure.png` only when the source data or script changes, ensuring smooth and automated reproducibility.
