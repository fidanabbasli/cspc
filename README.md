## PW2 Lab A

* **Mean Acceleration:** The measured mean acceleration is -8.58 m/s² with a standard deviation of 28.72 m/s².
* **Noise Observation:** The acceleration is extremely noisy because differentiation amplifies the small measurement errors from the position data. Differentiating twice magnifies this noise even further.
* **Integration Result:** Integrating the noisy acceleration back to position recovers the original path with a maximum difference of only 0.7846 m, demonstrating that integration suppresses random noise.
## PW2 --- Lab B

### Part 2: Optimization Methods Comparison
- **Convex function f(x):** All three methods (Gradient Descent, Newton, SLSQP) converged to $x \approx 3$.
- **Complex landscape g(x):** The starting point $x_0$ significantly affected the results. From $x_0 = 0$, Newton's method landed on a local maximum ($g'' < 0$), while SLSQP found the minimum.

### Part 3: Reaction Rate Fitting
- **Fitted Rate Constant (k):** ~0.252
- Saved plot to `PW2/Lab B/kinetics.png`.

### Part 4: Chemical Equilibrium
- **Equilibrium Extent (x):** ~0.664 (both Newton and SLSQP agreed).
- **Equilibrium Composition:** $H_2 \approx 0.336$ mol, $I_2 \approx 0.336$ mol, $HI \approx 1.328$ mol.
- Saved plot to `PW2/Lab B/equilibrium.png`.

### Part 5: Titration Equivalence Point
- **Equivalence Volume:** 50.0 mL (maximum slope of pH curve).
- Saved plot to `PW2/Lab B/titration.png`.