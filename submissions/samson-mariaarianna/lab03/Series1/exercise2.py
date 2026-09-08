"""
===============================================================================
EXERCISE 2 - Limit of the difference quotient  (a^h - 1) / h
===============================================================================
Laboratory Series1 (Series1.pdf), Exercise 2.

Mathematical background
-----------------------
For a fixed base a > 0 and a shrinking step size h the difference quotient

          a^h - 1
    q(h) = -------
              h

converges to the natural logarithm of a:

    lim    (a^h - 1) / h  =  ln(a)
    h -> 0

The PDF table is built for three different bases:

    a = 2                ->  ln(2)        = 0.6931471806
    a = 2.71828... (math.e) -> ln(e)      = 1.0000000000
    a = 3                ->  ln(3)        = 1.0986122887

with the h values of the PDF

    h = 0.1, 0.01, 0.001, 0.0001, 1e-5, 1e-6, 1e-7

and a tolerance of 10^-6:  the quotient "settles at" ln(a) once
|(a^h - 1)/h - ln(a)| < 10^-6.

WHAT THE SCRIPT PRODUCES
------------------------
* terminal table: h / (a^h-1)/h per base / |value - ln(a)| / converged
* figures/exercise2_limits.png
* results/exercise2_results.json   (read by the HTML dashboard and the PPTX)

Run with:
    python exercise2.py
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
# 0. Configuration (exactly the values used in Series1.pdf)
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(BASE_DIR, "figures")
DATA_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

# (name, a)
BASES = [
    ("a = 2",                2.0),
    ("a = e (2.71828...)",   math.e),
    ("a = 3",                3.0),
]

H_VALUES = [0.1, 0.01, 0.001, 0.0001, 1e-5, 1e-6, 1e-7]   # PDF table rows
TOLERANCE = 1e-6                                            # PDF tolerance

# defensive check: the PDF table uses exactly these seven step sizes
assert len(H_VALUES) == 7, (
    "The laboratory PDF requires all seven h values 0.1, 0.01, 0.001, "
    "0.0001, 0.00001, 0.000001 and 0.0000001"
)

# ---------------------------------------------------------------------------
# 1. Computation
# ---------------------------------------------------------------------------
def difference_quotient(a: float, h: float) -> float:
    """Return q(h) = (a^h - 1) / h for the given base a.

    Numerically stable evaluation:  a^h = exp(h * ln(a)), hence

        (a^h - 1) / h  =  expm1(h * ln(a)) / h

    math.expm1(x) computes exp(x) - 1 WITHOUT subtractive cancellation.
    The naive form (a ** h - 1.0) / h subtracts two almost equal numbers
    (a^h -> 1.0 as h shrinks) and then divides by h, which amplifies every
    rounding error by 1/h and makes the values move AWAY from ln(a) at
    h = 10^-7.  The expm1 form keeps full precision down to h = 10^-7.
    """
    return math.expm1(h * math.log(a)) / h


# table[base_index][h_index] = (h, quotient, limit, diff, converged)
table = []
for name, a in BASES:
    limit = math.log(a)
    col = []
    for h in H_VALUES:
        q = difference_quotient(a, h)
        diff = abs(q - limit)
        col.append((h, q, limit, diff, diff < TOLERANCE))
    table.append(col)

# the first h (scanning from the top of the PDF table downwards, i.e. from
# large h towards small h) for which the tolerance is reached
converged_info = []
for name, a in BASES:
    col = table[BASES.index((name, a))]
    first = next((h for h, _, _, _, yes in col if yes), None)
    converged_info.append((name, a, first))

# ---------------------------------------------------------------------------
# 2. Print the numerical results
# ---------------------------------------------------------------------------
print("=" * 88)
print("EXERCISE 2 - Limit of (a^h - 1) / h as h -> 0")
print("=" * 88)
print(f"Tolerance used:  |(a^h - 1)/h - ln(a)| < {TOLERANCE:.0e}   (from the PDF)")
print("-" * 88)
for (name, a), col in zip(BASES, table):
    print(f"Base {name}:  theoretical limit ln({a:.6g}) = {math.log(a):.10f}")
    print(f"  {'h':>9} {'(a^h-1)/h':>16} {'|q - ln(a)|':>14}  converged (tol 1e-6)")
    for h, q, limit, diff, conv in col:
        print(f"  {h:>9.1e} {q:>16.7f} {diff:>14.3e}  {'yes' if conv else 'no'}")
    print()

print("-" * 88)
print("Convergence summary (first h where the tolerance 1e-6 is met):")
for name, a, first_h in converged_info:
    print(f"  {name:<22} limit ln(a) = {math.log(a):>10.7f}   tolerance met for h <= {first_h:.0e}" if first_h is not None
          else f"  {name:<22} tolerance never met in the tested h range")
print("=" * 88)
# ---------------------------------------------------------------------------
# 3. Matplotlib graph (as in the PDF: bars per h, dashed limit lines ln(a))
# ---------------------------------------------------------------------------
def h_tick_label(h: float) -> str:
    """Readable x-tick label for one step size of the PDF table.

    Decimal notation for 0.1 .. 0.0001 and compact exponent notation for
    1e-5 .. 1e-7, so that ALL seven h values are clearly displayed.
    """
    if h >= 1e-4:
        return f"h = {h:g}"                       # 0.1, 0.01, 0.001, 0.0001
    return f"h = {h:.0e}".replace("e-0", "e-")    # h = 1e-5, 1e-6, 1e-7


fig, ax = plt.subplots(figsize=(11.5, 6.5))

# x positions are log10(h), so the seven h values are spaced logarithmically
x_pos = np.log10(H_VALUES)                        # -> [-1, -2, ..., -7]
bar_colors = ["#1f77b4", "#2ca02c", "#d62728"]    # blue / green / red
offsets = [-0.12, 0.0, 0.12]                      # group the three bases
width = 0.10

for idx, ((name, a), col) in enumerate(zip(BASES, table)):
    qs = [q for _h, q, _l, _d, _c in col]
    ax.bar(x_pos + offsets[idx], qs, width=width, color=bar_colors[idx],
           alpha=0.85, edgecolor="black", linewidth=0.4, zorder=3,
           label=name)

# dashed limit lines ln(a) for the three bases
for idx, ((name, a), col) in enumerate(zip(BASES, table)):
    limit = math.log(a)
    ax.axhline(limit, color=bar_colors[idx], linestyle="--", linewidth=1.6,
               zorder=2, label=f"limit  ln(a) = {limit:.4f}")

# x-axis: one clearly labelled tick for EVERY h value (0.1 ... 1e-7).
# The axis runs from log10(h) = -7  (h = 1e-7, left edge)  to
# log10(h) = -1  (h = 0.1, right edge); the explicit limits below keep
# BOTH extreme groups fully on screen (the old limits cut h = 1e-7 off).
ax.set_xticks(x_pos)
ax.set_xticklabels([h_tick_label(h) for h in H_VALUES], fontsize=9)
ax.set_xlim(x_pos[-1] - 0.55, x_pos[0] + 0.55)   # -> (-7.55, -0.45)

ax.set_xlabel("Step size h (logarithmic scale)  [h = 10^-1 down to 10^-7]",
              fontsize=10)
ax.set_ylabel("Value of  (a^h - 1) / h", fontsize=10)
ax.set_title("Exercise 2: (a^h - 1) / h settles at ln(a) as h shrinks",
             fontsize=13, fontweight="bold")
# explicit vertical range: tall bar at h = 0.1 / a = 3 is 1.1612, all
# limit lines are <= 1.0986, so [0, 1.35] keeps everything with headroom
ax.set_ylim(0.0, 1.35)
ax.grid(which="major", linestyle=":", alpha=0.6, zorder=0)
ax.grid(which="minor", axis="x", linestyle=":", alpha=0.3, zorder=0)
ax.legend(loc="lower right", fontsize=8)

# note on the tolerance, positioned with AXES-FRACTION coordinates in the
# empty top-left band of the plotting area and anchored bottom-side-down
# (va="top"), so it can never reach the figure title above the axes and it
# never covers the h = 0.1 or h = 1e-7 bars (all bars on the left half are
# <= 1.099, the box sits at y ~ 1.20 .. 1.30).
first_met = max(fh for _n, _a, fh in converged_info if fh is not None)
ax.text(0.02, 0.965,
        f"|q - ln(a)| < 1e-6 (PDF tolerance) met for h <= {first_met:.0e}\n"
        "-> the quotient settles at ln(a) for all three bases",
        transform=ax.transAxes, fontsize=9, color="#223344",
        va="top", ha="left",
        bbox=dict(boxstyle="round,pad=0.35", fc="#fdfbe9",
                  ec="#2e8b57", lw=1.2))

fig.tight_layout()
fig_path = os.path.join(FIG_DIR, "exercise2_limits.png")
fig.savefig(fig_path, dpi=160, bbox_inches="tight")
plt.close(fig)
print(f"Graph saved as: {fig_path}")

# ---------------------------------------------------------------------------
# 4. Save the numerical results in JSON (used by dashboard + PowerPoint)
# ---------------------------------------------------------------------------
results = {
    "title": "Exercise 2 - Limit of (a^h - 1)/h",
    "equation": "(a^h - 1) / h  ->  ln(a)",
    "tolerance": TOLERANCE,
    "h_values": H_VALUES,
    "bases": [
        {
            "name": name,
            "a": a,
            "ln_a": math.log(a),
            "ln_a_str": f"{math.log(a):.10f}",
            "first_converged_h": first_h,
            "first_converged_h_str": (f"{first_h:.0e}" if first_h is not None else "never"),
            "rows": [
                {
                    "h": h,
                    "quotient": q,
                    "quotient_str": f"{q:.7f}",
                    "diff": diff,
                    "diff_str": f"{diff:.3e}",
                    "converged": bool(conv),
                }
                for h, q, _l, diff, conv in col
            ],
        }
        for (name, a), col, (_, _, first_h) in zip(BASES, table, converged_info)
    ],
    "conclusion": (
        "For every base the difference quotient (a^h - 1)/h approaches "
        "ln(a) as h shrinks.  With the PDF tolerance 10^-6 the values have "
        "converged for h <= 10^-6: for a = e the quotient is 1.000000, "
        "for a = 2 it is 0.6931 and for a = 3 it is 1.0986, exactly the "
        "'settles at' row of the PDF table."
    ),
}

json_path = os.path.join(DATA_DIR, "exercise2_results.json")
with open(json_path, "w", encoding="utf-8") as fh:
    json.dump(results, fh, indent=2)
print(f"Results saved as: {json_path}")
print("EXERCISE 2 finished successfully.")