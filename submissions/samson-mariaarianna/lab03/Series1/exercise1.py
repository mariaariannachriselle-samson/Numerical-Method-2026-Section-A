"""
===============================================================================
EXERCISE 1 - Convergence of the sequence  (1 + 1/n)^n  towards  e
===============================================================================
Laboratory Series1 (Series1.pdf), Exercise 1.

Mathematical background
-----------------------
The sequence

    a_n = (1 + 1/n)^n

converges to Euler's number  e = 2.718281828459045...  as n grows:

    lim    (1 + 1/n)^n  =  e
    n -> inf

In finance this sequence models an interest rate of 100 % compounded
n times per year.  The PDF gives the classical frequencies:

    yearly          n =   1   ->   2.000000
    twice a year    n =   2   ->   2.250000
    quarterly       n =   4   ->   2.441406

This script reproduces exactly those three values, then extends n over the
larger frequencies shown in the PDF (Figure 1) so that the convergence
toward e becomes visible.  A Matplotlib bar graph is drawn (title, axis
labels, e reference line, annotations, grid), the numerical results are
printed, and the graph is saved as a PNG file.

WHAT THE SCRIPT PRODUCES
------------------------
* terminal table: frequency / n / value / |e - value|
* figures/exercise1_convergence.png
* results/exercise1_results.json   (read by the HTML dashboard and the PPTX)

Run with:
    python exercise1.py
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

# Make sure unicode symbols print correctly in the terminal
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

E = math.e                       # reference value the sequence converges to
TOLERANCE = 1e-6                 # convergence tolerance: |value - e| < 1e-6

# Compounding frequencies, exactly as used in Series1.pdf, Figure 1.
# Each entry is (label, number of compounding periods per year n).
FREQUENCIES = [
    ("yearly",             1),
    ("twice a year",       2),
    ("quarterly",          4),
    ("monthly",            12),
    ("weekly",             52),
    ("daily",              365),
    ("hourly",             8760),
    ("every minute",       525600),
    ("every second",       31536000),
    ("every millisecond",  31_536_000_000),
    ("every microsecond",  31_536_000_000_000),
    ("every nanosecond",   31_536_000_000_000_000),
]

# ---------------------------------------------------------------------------
# 1. Computation of (1 + 1/n)^n
# ---------------------------------------------------------------------------
def compound_value(n: int) -> float:
    """Evaluate a_n = (1 + 1/n)^n.

    The direct expression ``(1.0 + 1.0/n) ** n`` is used for moderate n.
    For larger n the direct formula becomes inaccurate: 1/n falls towards the
    floating point resolution and the internal logarithm in the power
    function is amplified by n, so the result can be wrong in the 3rd digit
    (for n = 10^13 the direct formula returns 2.7219 instead of 2.71828).
    For n >= 10^6 the mathematically identical form
    exp(n * ln(1 + 1/n)) is evaluated with math.log1p, which is exact for
    tiny arguments.  This mirrors the "numerically stable" requirement of
    the laboratory: both forms give the same true value, the stable one is
    not destroyed by rounding.
    """
    if n >= 1_000_000:
        # stable path - correct even when 1 + 1/n rounds to 1.0
        return math.exp(n * math.log1p(1.0 / n))
    return (1.0 + 1.0 / n) ** n


# ---------------------------------------------------------------------------
# 2. Build the results table
# ---------------------------------------------------------------------------
rows = []                        # (label, n, value, error, converged)
for label, n in FREQUENCIES:
    value = compound_value(n)
    error = E - value
    converged = abs(error) < TOLERANCE
    rows.append((label, n, value, error, converged))

# ---------------------------------------------------------------------------
# 3. Print the numerical results
# ---------------------------------------------------------------------------
print("=" * 78)
print("EXERCISE 1 - Convergence of (1 + 1/n)^n towards e")
print("=" * 78)
print(f"Reference value:          e  = {E:.15f}")
print(f"Convergence tolerance:    |value - e| < {TOLERANCE:.0e}")
print("-" * 78)
print(f"{'frequency':<20}{'n':>18}{'(1 + 1/n)^n':>20}{'|e - value|':>16}  converged")
print("-" * 78)
for label, n, value, error, converged in rows:
    print(f"{label:<20}{n:>18,}{value:>20.6f}{abs(error):>16.3e}  {'yes' if converged else 'no'}")
print("-" * 78)

# A quick sanity check with the numbers of the PDF table
print("Values from the PDF table:")
print(f"  yearly        n =  1  -> {rows[0][2]:.6f}   (expected 2.000000)")
print(f"  twice a year  n =  2  -> {rows[1][2]:.6f}   (expected 2.250000)")
print(f"  quarterly     n =  4  -> {rows[2][2]:.6f}   (expected 2.441406)")

last_label, last_n, last_value, last_error, _ = rows[-1]
print("-" * 78)
print(f"Largest frequency tested: {last_label}  (n = {last_n:,})")
print(f"  (1 + 1/n)^n = {last_value:.15f}")
print(f"  |e - value| = {abs(last_error):.3e}")
print(f"  The sequence approaches e = {E:.6f} as n grows  ->  convergence demonstrated.")
print("=" * 78)
# ---------------------------------------------------------------------------
# 4. Bar graph (histogram) showing the values and the convergence towards e
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(12.5, 6.5))

labels = [f"{label}\n(n = {n:,})" for label, n, _, _, _ in rows]
values = [value for _, _, value, _, _ in rows]

x = np.arange(len(rows))
colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(rows)))
bars = ax.bar(x, values, color=colors, edgecolor="black", linewidth=0.6,
              zorder=3, label="(1 + 1/n)^n")

# reference line for e
ax.axhline(E, color="red", linestyle="--", linewidth=1.8, zorder=4,
           label=f"e = {E:.6f}")

# value annotations on top of every bar
for xi, v in zip(x, values):
    ax.text(xi, v + 0.012, f"{v:.6f}", ha="center", va="bottom",
            fontsize=7.5, rotation=0, zorder=5)

# readable n / frequency labels
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=7)
plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
ax.tick_params(axis="x", labelsize=7)

ax.set_xlabel("Compounding frequency  (n periods per year)", fontsize=10)
ax.set_ylabel("Value of  (1 + 1/n)^n", fontsize=10)
ax.set_title("Exercise 1: Convergence of (1 + 1/n)^n to e",
             fontsize=13, fontweight="bold")
ax.grid(axis="y", linestyle=":", alpha=0.6, zorder=0)

# annotation of the error behaviour, as in the PDF (error ~ e / (2n))
ax.annotate(f"error shrinks like e/(2n)  ->  e - value = {abs(rows[-1][3]):.2e}",
            xy=(len(rows) - 0.6, E - abs(rows[-1][3])),
            xytext=(len(rows) * 0.35, max(values) + 0.06),
            fontsize=9, arrowprops=dict(arrowstyle="->", color="gray"))

ax.legend(loc="lower right", fontsize=9)
ax.set_ylim(1.95, max(values) + 0.10)

fig.tight_layout()
fig_path = os.path.join(FIG_DIR, "exercise1_convergence.png")
fig.savefig(fig_path, dpi=160, bbox_inches="tight")
plt.close(fig)
print(f"Graph saved as: {fig_path}")

# ---------------------------------------------------------------------------
# 5. Save the numerical results in JSON (used by dashboard + PowerPoint)
# ---------------------------------------------------------------------------
results = {
    "title": "Exercise 1 - Convergence of (1 + 1/n)^n to e",
    "equation": "(1 + 1/n)^n",
    "e": E,
    "e_str": f"{E:.6f}",
    "tolerance": TOLERANCE,
    "frequencies": [
        {
            "label": label,
            "n": n,
            "value": value,
            "value_str": f"{value:.6f}",
            "error": E - value,
            "error_str": f"{E - value:.3e}",
            "converged": bool(converged),
        }
        for label, n, value, _err, converged in rows
    ],
    "largest_n": last_n,
    "last_value_str": f"{last_value:.12f}",
    "last_error_str": f"{abs(last_error):.3e}",
    "conclusion": (
        "The values of (1 + 1/n)^n for n = 1, 2, 4 are exactly the PDF values "
        "2.000000, 2.250000 and 2.441406.  Extending n towards larger "
        "compounding frequencies the sequence approaches e; already for "
        "season-level frequencies (minute, second, ...) it agrees with "
        "e = 2.718282 to six decimals."
    ),
}

json_path = os.path.join(DATA_DIR, "exercise1_results.json")
with open(json_path, "w", encoding="utf-8") as fh:
    json.dump(results, fh, indent=2)
print(f"Results saved as: {json_path}")
print("EXERCISE 1 finished successfully.")