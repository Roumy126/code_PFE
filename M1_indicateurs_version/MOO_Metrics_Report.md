# 📊 Multi-Objective Optimization (MOO) Metrics Report

This document provides a detailed explanation and interpretation guide for the quality indicators implemented in the Quantum Circuit Optimization pipeline.

## 1. Why Multi-Objective Metrics?
Quantum optimization isn't just about **Fidelity**. We also care about **Circuit Depth** (how "tall" it is) and **Gate Cost** (how expensive it is). 

Since these goals often conflict (e.g., a more accurate circuit might be deeper), we use **Pareto Optimization**. These metrics tell us how "healthy" our set of trade-offs is.

---

## 2. The Metrics Explained

### 🟢 Hypervolume (HV)
*   **What it is**: The most used metric in MOO. It calculates the volume of the space "covered" by your solutions relative to a fixed worst-case reference point.
*   **Interpretation**: 
    *   **Increasing HV** = The algorithm is getting better (it's pushing toward higher fidelity and lower depth/cost simultaneously).
    *   **Higher is better.**

### 🔵 Spread ($\Delta$)
*   **What it is**: Measures the distribution of solutions on the Pareto front.
*   **Interpretation**: 
    *   **$\Delta \approx 0$**: Excellent. Your solutions are evenly spaced and cover the extremes (e.g., you have the absolute best fidelity version AND the absolute simplest version).
    *   **High $\Delta$**: Bad. Your solutions are clustered in one spot, or you have large gaps.
    *   **Lower is better.**

### 🟡 Inverted Generational Distance (IGD)
*   **What it is**: The average distance from the "Perfect Pareto Front" to your actual solutions.
*   **Interpretation**: 
    *   **Decreasing IGD** = You are converging toward the mathematical limit of what's possible for that block.
    *   **Lower is better.**

### 🟣 Epsilon Indicator ($\epsilon$)
*   **What it is**: Tells you the factor by which you'd need to "stretch" your solutions to cover the perfect front.
*   **Interpretation**: 
    *   **$\epsilon \approx 1.0$**: Your solutions are very close to optimal.
    *   **Lower is better.**

---

## 3. How to Read Your Results

When you look at the `Moo Metrics Per Block` summary in your notebook output:

1.  **Check `n_pareto`**: If it's `1`, you only have one optimal solution. If it's higher (e.g., `4` or `5`), you have multiple choices (some high-fidelity, some low-depth).
2.  **Compare HV across blocks**: Larger blocks (more qubits) usually have lower HV because the optimization problem is much harder.
3.  **The "Convergence de la fidélité" Plot**: This shows the single best individual. If it goes down, your code is working. If it's flat, you might need more generations or a larger population.

---

## 4. Technical Reference
*   **Implementation**: See **Partie 5.5** in the notebook.
*   **Data Access**: The metrics are returned in the `info` object of the pipeline.
*   **LaTeX Source**: I have also provided a formal `.tex` file if you wish to include these definitions in a research paper or report.
