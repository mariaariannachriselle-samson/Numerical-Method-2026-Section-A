"""
===============================================================================
EXERCISE 3 - The series for e^x :  e^x = sum x^n / n!
===============================================================================
Laboratory Series1 (Series1.pdf), Exercise 3.

Mathematical background
-----------------------
The exponential function has the power-series representation

             inf
             ---     x^n
    e^x  =   >      ----
             ---     n!
             n=0

For x = 1 this gives the number e = 2.718281828459045...:

             N        1
    S_N  =  sum     ----    ->   e      as N -> infinity
             n=0     n!

The PDF asks for the partial sums up to N = 10,000 terms.

Numerical stability
-------------------
math.factorial(10000) would create astronomically large integers and then
divide them, which is both wasteful and unnecessary.  Instead every term is
obtained from the previous one with the recurrence

    term_n = term_(n-1) * x / n          (with term_0 = x^0 / 0! = 1)

so no factorial is ever computed.  The terms become smaller than 10^-300
already around n = 170, and the partial sum stabilises at e long before
10,000 terms.

WHAT THE SCRIPT PRODUCES
------------------------
* terminal table: N / partial sum / |S_N - e| / correct digits
* figures/exercise3_series.png      (bar chart of partial sums + log-log error)
* results/exercise3_results.json    (read by the HTML dashboard and the PPTX)

Run with:
    python exercise3.py
===============================================================================
"""

import json
import math
import os
import sys

import matplotlib

matplotlib.use("Agg")            # non-interactive backend (only savefig)
import matplotlib.pyplot as plt
import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# ---------------------------------------------------------------------------
# 0. Configuration
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(BASE_DIR, "figures")
DATA_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

X = 1.0                       # the PDF uses x = 1  (so the series sums to e)
N_MAX = 10000                 # partial sums up to 10,000 terms (PDF)
E = math.e                    # reference value

# the N values highlighted in the PDF figure 3 (number of terms in the sum)
PLOT_N = [1, 10, 100, 1000, 10000]

# N values printed in the terminal table and stored in the JSON
TARGET_N = [1, 2, 3, 4, 5, 10, 20, 50, 100, 1000, 10000]

# ---------------------------------------------------------------------------
# 1. Numerically stable computation of the partial sums
# ---------------------------------------------------------------------------
def partial_sums_stable(x: float, n_max: int) -> list:
    """Return the list S where S[N-1] is the partial sum after N terms:

        S[N-1] = sum_{n=0}^{N-1} x^n / n!          (N terms)

    Terms are updated recursively (no factorial):  t_n = t_(n-1) * x / n
    """
    partial = []
    acc = 0.0
    term = 1.0                       # t_0 = x^0 / 0! = 1
    for n in range(n_max):           # n = 0 .. n_max-1  ->  n_max terms
        acc += term
        partial.append(acc)
        term *= x / (n + 1)          # t_{n+1} = t_n * x / (n+1)
    return partial


partial = partial_sums_stable(X, N_MAX)
final_sum = partial[N_MAX - 1]       # partial sum after 10,000 terms

# how many terms are needed until the running sum stops changing at all in
# double precision (i.e. the added terms are below the floating point floor)
stable_after = next((n + 1 for n in range(1, len(partial))
                     if partial[n] == partial[n - 1]), None)

# first N such that the RELATIVE error drops to machine precision (2e-16)
machine_n = next((n + 1 for n in range(len(partial))
                  if abs(partial[n] - E) / E <= 2e-16), None)

# ---------------------------------------------------------------------------
# 2. Print the numerical results
# ---------------------------------------------------------------------------
print("=" * 82)
print("EXERCISE 3 - Series for e^x at x = 1, up to 10,000 terms")
print("=" * 82)
print("Series:   e^x = sum (n=0..inf) x^n / n!   with x = " + str(X))
print(f"Reference value:  e = {E:.15f}")
print(f"Number of terms computed: {N_MAX:,}   (PDF requirement)")
print("-" * 82)
print(f"{'N (terms)':>12}{'S_N':>22}{'|S_N - e|':>16}{'correct digits':>16}")
print("-" * 82)
for n_terms in TARGET_N:
    s = partial[n_terms - 1]
    err = abs(s - E)
    digits = -math.log10(err / E) if err > 0 else 15.0
    print(f"{n_terms:>12,}{s:>22.12f}{err:>16.3e}{digits:>16.1f}")
print("-" * 82)
print(f"Partial sum after 10,000 terms:  S = {final_sum:.15f}")
print(f"math.e                            = {E:.15f}")
print(f"Absolute error: |S - e|           = {abs(final_sum - E):.3e}")
print(f"Relative error: |S - e| / e       = {abs(final_sum - E) / E:.3e}")
print(f"The partial sum stops changing (double precision) after ~{stable_after} terms.")
print(f"Machine precision |S_N - e| <= 2e-16 reached after ~{machine_n} terms.")
print(f"Tail terms beyond n ~ 170 are < 1e-300 (term recurrence, no factorial).")
print("=" * 82)
# ---------------------------------------------------------------------------
# 3. Matplotlib graph (mirrors PDF figure 3)
#    left  : bar chart of the partial sums S_N for N = 1, 10, 100, 1000, 10000
#    right : |S_N - e| vs N on a log-log scale with the accuracy bands of the PDF
# ---------------------------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 6.0),
                               constrained_layout=True)

# --- left panel: histogram / bar chart of the partial sums ----------------
n_plot = [n for n in PLOT_N if n <= len(partial)]
s_plot = [partial[n - 1] for n in n_plot]

ax1.bar([str(n) for n in n_plot], s_plot, color="#4C72B0",
        edgecolor="black", linewidth=0.6, zorder=3, label="partial sum S_N")
ax1.axhline(E, color="red", linestyle="--", linewidth=1.8, zorder=4,
            label=f"e = {E:.6f}")
for i, sv in enumerate(s_plot):
    ax1.text(i, sv - 0.010, f"{sv:.6f}", ha="center", va="top", fontsize=8)
ax1.set_xticks(range(len(n_plot)))
ax1.set_xticklabels([f"{n:,}" for n in n_plot])
ax1.set_xlabel("Number of terms N in the summation", fontsize=10)
ax1.set_ylabel("Partial sum  S_N", fontsize=10)
ax1.set_title("Exercise 3: e^x = sum x^n/n!  (x = 1)\nHistogram of the partial sums",
              fontsize=11, fontweight="bold")
ax1.grid(axis="y", linestyle=":", alpha=0.6)
ax1.legend(loc="lower right", fontsize=9)

# --- right panel: error N -> |S_N - e| on a log-log scale ----------------
nn = np.arange(1, len(partial) + 1)
errs = np.abs(np.asarray(partial) - E)
mask = errs > 1e-320              # only positive errors (S_N != e exactly)
ax2.loglog(nn[mask], errs[mask], color="#4C72B0", linewidth=1.6,
           label="|S_N - e|")

# accuracy bands used in the PDF figure 3
ax2.axhspan(1e-2, 10, color="#FFB366", alpha=0.25, label="rough  (error > 1e-2)")
ax2.axhspan(1e-6, 1e-2, color="#FFE082", alpha=0.35, label="engineering  (1e-6 .. 1e-2)")
ax2.axhspan(2e-16, 1e-6, color="#7FC97F", alpha=0.30, label="high precision  (< 1e-6)")
ax2.axhline(2e-16, color="k", linestyle=":", linewidth=1.0, label="machine precision  (2e-16)")

ax2.set_xlabel("Number of terms N in the summation (log scale)", fontsize=10)
ax2.set_ylabel("Absolute error  |S_N - e|  (log scale)", fontsize=10)
ax2.set_title("How many correct digits each N buys (log-log)",
              fontsize=11, fontweight="bold")
ax2.grid(which="major", linestyle=":", alpha=0.5)
ax2.grid(which="minor", linestyle=":", alpha=0.25)
ax2.legend(loc="lower left", fontsize=7)

fig_path = os.path.join(FIG_DIR, "exercise3_series.png")
fig.savefig(fig_path, dpi=160, bbox_inches="tight")
plt.close(fig)
print(f"Graph saved as: {fig_path}")

# ---------------------------------------------------------------------------
# 4. Save the numerical results in JSON (used by dashboard + PowerPoint)
# ---------------------------------------------------------------------------
def correct_digits(s):
    err = abs(s - E)
    return None if err == 0 else -math.log10(err / E)

results = {
    "title": "Exercise 3 - Series for e^x (x = 1)",
    "equation": "e^x = sum_{n=0..inf} x^n / n!",
    "x": X,
    "N_max": N_MAX,
    "e": E,
    "e_str": f"{E:.15f}",
    "final_sum": final_sum,
    "final_sum_str": f"{final_sum:.15f}",
    "abs_error": abs(final_sum - E),
    "abs_error_str": f"{abs(final_sum - E):.3e}",
    "rel_error_str": f"{abs(final_sum - E) / E:.3e}",
    "stable_after_terms": stable_after,
    "machine_precision_terms": machine_n,
    "partial_sums": [
        {
            "N": nt,
            "S": partial[nt - 1],
            "S_str": f"{partial[nt - 1]:.12f}",
            "error": abs(partial[nt - 1] - E),
            "error_str": f"{abs(partial[nt - 1] - E):.3e}",
            "digits_str": f"{correct_digits(partial[nt - 1]):.1f}",
        }
        for nt in TARGET_N
    ],
    "conclusion": (
        "The stable recursive series evaluation needs no factorial: with the "
        "recurrence term_n = term_(n-1) * x / n the partial sums converge to "
        "e.  After about 20 terms the sum agrees with e to machine precision, "
        "and the full 10,000-term sum reproduces e = 2.718281828459045 with "
        "relative error below 1e-15."
    ),
}

json_path = os.path.join(DATA_DIR, "exercise3_results.json")
with open(json_path, "w", encoding="utf-8") as fh:
    json.dump(results, fh, indent=2)
print(f"Results saved as: {json_path}")
print("EXERCISE 3 finished successfully.")