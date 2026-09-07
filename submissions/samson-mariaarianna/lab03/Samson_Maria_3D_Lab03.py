# ============================================================
# LAB 03 - REAL-WORLD DATA LINEAR REGRESSION
# Numerical Methods - Section 3D
#
# Dataset:
# Philippines Population vs GDP (2011-2025)
# Source: World Bank
# ============================================================

import numpy as np
import matplotlib.pyplot as plt


# ------------------------------------------------------------
# 1. DATA
# ------------------------------------------------------------
# x = Population (people)
# y = GDP (current US$)

years = np.array([
    2011, 2012, 2013, 2014, 2015,
    2016, 2017, 2018, 2019, 2020,
    2021, 2022, 2023, 2024, 2025
])

population = np.array([
     98248614,
    100175512,
    102076336,
    103767130,
    105312992,
    106735719,
    108119693,
    109465287,
    110804683,
    112081264,
    113100950,
    113964338,
    114891199,
    115843670,
    116786962
], dtype=float)

gdp = np.array([
    234216872955,
    261920481725,
    283902818577,
    297483582902,
    306445117723,
    318626957647,
    328480739673,
    346841689750,
    376823091728,
    361750638591,
    394087419150,
    404353550708,
    437055646226,
    461671537274,
    487086070094
], dtype=float)


# ------------------------------------------------------------
# 2. LEAST-SQUARES LINEAR REGRESSION
# ------------------------------------------------------------
# Regression model:
#
# y = a0 + a1*x
#
# Slope:
# a1 = [n Σ(xy) - Σx Σy] / [n Σ(x²) - (Σx)²]
#
# Intercept:
# a0 = y_mean - a1*x_mean

n = len(x := population)
y = gdp

sum_x = np.sum(x)
sum_y = np.sum(y)
sum_x2 = np.sum(x ** 2)
sum_xy = np.sum(x * y)

a1 = (n * sum_xy - sum_x * sum_y) / (
    n * sum_x2 - sum_x ** 2
)

x_mean = np.mean(x)
y_mean = np.mean(y)

a0 = y_mean - a1 * x_mean


# ------------------------------------------------------------
# 3. CALCULATE PREDICTED VALUES AND RESIDUALS
# ------------------------------------------------------------

y_pred = a0 + a1 * x

residuals = y - y_pred


# ------------------------------------------------------------
# 4. SUM OF SQUARED ERRORS (SSE)
# ------------------------------------------------------------

SSE = np.sum(residuals ** 2)


# ------------------------------------------------------------
# 5. R-SQUARED
# ------------------------------------------------------------

SST = np.sum((y - y_mean) ** 2)

r_squared = 1 - (SSE / SST)


# ------------------------------------------------------------
# 6. STANDARD ERROR OF ESTIMATE
# ------------------------------------------------------------

standard_error = np.sqrt(SSE / (n - 2))


# ------------------------------------------------------------
# 7. PREDICTION
# ------------------------------------------------------------
# We choose 120,000,000 people.
# This value is NOT included in our 2011-2025 dataset.

prediction_population = 120_000_000

predicted_gdp = a0 + a1 * prediction_population


# ------------------------------------------------------------
# 8. DISPLAY RESULTS
# ------------------------------------------------------------

print("=" * 65)
print("LAB 03 - REAL-WORLD DATA LINEAR REGRESSION")
print("=" * 65)

print("\nDATASET")
print("-" * 65)
print("Source: World Bank")
print("Country: Philippines")
print("Years: 2011-2025")
print("Number of observations:", n)

print("\nVARIABLES")
print("-" * 65)
print("Independent variable (x): Population (people)")
print("Dependent variable (y): GDP (current US$)")

print("\nREGRESSION RESULTS")
print("-" * 65)

print(f"Intercept (a0): {a0:,.6f}")
print(f"Slope (a1):     {a1:,.6f}")

print("\nRegression equation:")
print(f"y = {a0:,.2f} + {a1:,.6f}x")

print("\nERROR AND FIT STATISTICS")
print("-" * 65)
print(f"SSE (Sr):          {SSE:,.2f}")
print(f"R-squared (r²):    {r_squared:.6f}")
print(f"R-squared (%):     {r_squared * 100:.2f}%")
print(f"Standard error:    {standard_error:,.2f}")

print("\nPREDICTION")
print("-" * 65)
print(f"Population (x):    {prediction_population:,.0f} people")
print(f"Predicted GDP:     ${predicted_gdp:,.2f}")

print("\nINTERPRETATION")
print("-" * 65)
print(
    "The positive slope indicates that population and GDP have "
    "a positive linear relationship in this dataset."
)

print(
    f"The R-squared value of {r_squared:.4f} means that approximately "
    f"{r_squared * 100:.2f}% of the variation in GDP is explained "
    "by the linear regression model."
)

print(
    "The residuals represent the difference between the actual GDP "
    "and the GDP predicted by the regression line."
)

print("=" * 65)


# ------------------------------------------------------------
# 9. PRINT DATA TABLE
# ------------------------------------------------------------

print("\nDATA USED")
print("-" * 65)
print(f"{'Year':<8}{'Population':>18}{'GDP (US$)':>25}")

for i in range(n):
    print(
        f"{years[i]:<8}"
        f"{population[i]:>18,.0f}"
        f"{gdp[i]:>25,.2f}"
    )


# ------------------------------------------------------------
# 10. GRAPH: DATA AND FITTED LINE
# ------------------------------------------------------------

# Sort x values so the fitted line is displayed smoothly.
sort_index = np.argsort(x)

x_sorted = x[sort_index]
y_pred_sorted = y_pred[sort_index]

plt.figure(figsize=(10, 6))

plt.scatter(
    x,
    y,
    label="Actual data"
)

plt.plot(
    x_sorted,
    y_pred_sorted,
    linewidth=2,
    label="Least-squares regression line"
)

plt.xlabel("Population (people)")
plt.ylabel("GDP (current US$)")
plt.title("Philippines Population vs GDP (2011-2025)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.savefig(
    "lab03_regression_graph.png",
    dpi=300
)

plt.show()


# ------------------------------------------------------------
# 11. RESIDUAL PLOT
# ------------------------------------------------------------

plt.figure(figsize=(10, 6))

plt.scatter(
    x,
    residuals
)

plt.axhline(
    0,
    linewidth=1.5
)

plt.xlabel("Population (people)")
plt.ylabel("Residual (US$)")
plt.title("Residual Plot - Philippines GDP Regression")
plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.savefig(
    "lab03_residual_plot.png",
    dpi=300
)

plt.show()


# ------------------------------------------------------------
# END OF PROGRAM
# ------------------------------------------------------------