import math
import os
import matplotlib.pyplot as plt


# ============================================================
# CIVIL ENGINEERING SERIES EXERCISE
# HOW ACCURATE IS GOOD ENOUGH?
# ============================================================

# ------------------------------------------------------------
# GIVEN DATA
# ------------------------------------------------------------

L = 20.0  # structural length, m

angles_deg = [1, 2, 5, 10, 15, 20, 30]

term_counts = [1, 2, 3, 4]

ERROR_TOLERANCE = 0.1  # percent


# ------------------------------------------------------------
# OUTPUT FOLDER
# Saves everything directly on the Desktop
# ------------------------------------------------------------

OUTPUT_FOLDER = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "series_outputs"
)

os.makedirs(OUTPUT_FOLDER, exist_ok=True)


# ============================================================
# PART 1 - GEOMETRIC SERIES
# ============================================================

def geometric_sum(x, N):
    """
    Calculate the partial geometric series:

    S_N = 1 + x + x^2 + ... + x^N

    using a loop.
    """

    total = 0.0

    for k in range(N + 1):
        total += x ** k

    return total


def geometric_series_analysis():

    print("\n" + "=" * 80)
    print("PART 1 - GEOMETRIC SERIES")
    print("=" * 80)

    x_values = [0.5, 0.8, 0.9]

    for x in x_values:

        exact = 1.0 / (1.0 - x)

        print(f"\nx = {x}")
        print(f"Exact value = {exact:.10f}")

        print(
            f"{'N':>5}"
            f"{'Partial Sum':>20}"
            f"{'Absolute Error':>20}"
        )

        print("-" * 50)

        for N in [1, 2, 3, 5, 10, 20]:

            approximation = geometric_sum(x, N)

            error = abs(exact - approximation)

            print(
                f"{N:>5}"
                f"{approximation:>20.10f}"
                f"{error:>20.10f}"
            )


# ============================================================
# PART 2 - POWER SERIES
# ============================================================

def power_series(x, coefficients):
    """
    Evaluate a finite power series:

    P_N(x) = a0 + a1*x + a2*x^2 + ... + aN*x^N
    """

    result = 0.0

    for k, a_k in enumerate(coefficients):
        result += a_k * (x ** k)

    return result


def power_series_demo():

    print("\n" + "=" * 80)
    print("PART 2 - POWER SERIES")
    print("=" * 80)

    coefficients = [2, 3, 4, 5]

    x = 2

    result = power_series(x, coefficients)

    print("\nExample:")
    print("P(x) = 2 + 3x + 4x^2 + 5x^3")
    print(f"x = {x}")
    print(f"P({x}) = {result:.6f}")


# ============================================================
# PART 3 - MACLAURIN SERIES
# ============================================================

def sin_maclaurin(theta, N):
    """
    Approximate sin(theta) using N Maclaurin terms.

    sin(theta) =
        theta
        - theta^3/3!
        + theta^5/5!
        - theta^7/7!
        + ...
    """

    result = 0.0

    for n in range(N):

        sign = (-1) ** n

        exponent = 2 * n + 1

        factorial = math.factorial(exponent)

        term = (
            sign
            * theta ** exponent
            / factorial
        )

        result += term

    return result


# ============================================================
# PART 4 - TAYLOR SERIES CENTERED AT 10 DEGREES
# ============================================================

def sin_taylor(theta, a, N):
    """
    Approximate sin(theta) using Taylor series centered
    at a.

    Derivative cycle:

        sin
        cos
        -sin
        -cos
    """

    result = 0.0

    sin_a = math.sin(a)
    cos_a = math.cos(a)

    for n in range(N):

        derivative_pattern = n % 4

        if derivative_pattern == 0:
            f_deriv = sin_a

        elif derivative_pattern == 1:
            f_deriv = cos_a

        elif derivative_pattern == 2:
            f_deriv = -sin_a

        else:
            f_deriv = -cos_a

        term = (
            f_deriv
            * (theta - a) ** n
            / math.factorial(n)
        )

        result += term

    return result


# ============================================================
# ERROR CALCULATIONS
# ============================================================

def absolute_error(exact, approximation):

    return abs(exact - approximation)


def percentage_error(exact, approximation):

    if exact == 0:
        return 0.0

    return (
        abs(exact - approximation)
        / abs(exact)
        * 100.0
    )


# ============================================================
# CREATE MACLAURIN RESULTS
# ============================================================

def create_maclaurin_results():

    results = []

    for angle in angles_deg:

        theta = math.radians(angle)

        exact_y = L * math.sin(theta)

        for N in term_counts:

            approx_sin = sin_maclaurin(
                theta,
                N
            )

            approx_y = L * approx_sin

            abs_error = absolute_error(
                exact_y,
                approx_y
            )

            pct_error = percentage_error(
                exact_y,
                approx_y
            )

            results.append({
                "angle": angle,
                "terms": N,
                "exact_y": exact_y,
                "approx_y": approx_y,
                "absolute_error": abs_error,
                "percentage_error": pct_error
            })

    return results


# ============================================================
# CREATE TAYLOR RESULTS
# ============================================================

def create_taylor_results():

    results = []

    # Center at 10 degrees
    a = math.radians(10)

    for angle in angles_deg:

        theta = math.radians(angle)

        exact_y = L * math.sin(theta)

        for N in term_counts:

            approx_sin = sin_taylor(
                theta,
                a,
                N
            )

            approx_y = L * approx_sin

            abs_error = absolute_error(
                exact_y,
                approx_y
            )

            pct_error = percentage_error(
                exact_y,
                approx_y
            )

            results.append({
                "angle": angle,
                "terms": N,
                "exact_y": exact_y,
                "approx_y": approx_y,
                "absolute_error": abs_error,
                "percentage_error": pct_error
            })

    return results


# ============================================================
# PRINT TABLE
# ============================================================

def print_results_table(results, title):

    print("\n" + "=" * 115)
    print(title)
    print("=" * 115)

    print(
        f"{'Angle':>7}"
        f"{'Terms':>7}"
        f"{'Exact y (m)':>18}"
        f"{'Approx. y (m)':>18}"
        f"{'Abs. Error':>18}"
        f"{'% Error':>16}"
    )

    print("-" * 115)

    for row in results:

        print(
            f"{row['angle']:>7}"
            f"{row['terms']:>7}"
            f"{row['exact_y']:>18.8f}"
            f"{row['approx_y']:>18.8f}"
            f"{row['absolute_error']:>18.8f}"
            f"{row['percentage_error']:>16.8f}"
        )


# ============================================================
# SAVE COMPLETE RESULTS
# ============================================================

def save_results_to_text(
    maclaurin_results,
    taylor_results
):

    filepath = os.path.join(
        OUTPUT_FOLDER,
        "series_results.txt"
    )

    with open(
        filepath,
        "w",
        encoding="utf-8"
    ) as file:

        file.write(
            "CIVIL ENGINEERING SERIES EXERCISE\n"
        )

        file.write(
            "HOW ACCURATE IS GOOD ENOUGH?\n\n"
        )

        file.write(
            "Engineering relationship:\n"
        )

        file.write(
            "y = L sin(theta)\n"
        )

        file.write(
            f"L = {L:.1f} m\n\n"
        )

        # ----------------------------------------------------
        # MACLAURIN
        # ----------------------------------------------------

        file.write(
            "MACLAURIN SERIES RESULTS\n"
        )

        file.write("-" * 115 + "\n")

        file.write(
            f"{'Angle':>7}"
            f"{'Terms':>7}"
            f"{'Exact y (m)':>18}"
            f"{'Approx. y (m)':>18}"
            f"{'Abs. Error':>18}"
            f"{'% Error':>16}\n"
        )

        file.write("-" * 115 + "\n")

        for row in maclaurin_results:

            file.write(
                f"{row['angle']:>7}"
                f"{row['terms']:>7}"
                f"{row['exact_y']:>18.8f}"
                f"{row['approx_y']:>18.8f}"
                f"{row['absolute_error']:>18.8f}"
                f"{row['percentage_error']:>16.8f}\n"
            )

        file.write("\n\n")

        # ----------------------------------------------------
        # TAYLOR
        # ----------------------------------------------------

        file.write(
            "TAYLOR SERIES RESULTS - CENTERED AT 10 DEGREES\n"
        )

        file.write("-" * 115 + "\n")

        file.write(
            f"{'Angle':>7}"
            f"{'Terms':>7}"
            f"{'Exact y (m)':>18}"
            f"{'Approx. y (m)':>18}"
            f"{'Abs. Error':>18}"
            f"{'% Error':>16}\n"
        )

        file.write("-" * 115 + "\n")

        for row in taylor_results:

            file.write(
                f"{row['angle']:>7}"
                f"{row['terms']:>7}"
                f"{row['exact_y']:>18.8f}"
                f"{row['approx_y']:>18.8f}"
                f"{row['absolute_error']:>18.8f}"
                f"{row['percentage_error']:>16.8f}\n"
            )

    print(
        f"\nSaved results to:\n{filepath}"
    )


# ============================================================
# PART 6 - 0.1% ERROR TOLERANCE
# ============================================================

def find_minimum_terms_maclaurin(angle):

    theta = math.radians(angle)

    exact = math.sin(theta)

    for N in range(1, 21):

        approximation = sin_maclaurin(
            theta,
            N
        )

        error = percentage_error(
            exact,
            approximation
        )

        if error < ERROR_TOLERANCE:

            return N, error

    return None, None


def find_minimum_terms_taylor(angle):

    theta = math.radians(angle)

    a = math.radians(10)

    exact = math.sin(theta)

    for N in range(1, 21):

        approximation = sin_taylor(
            theta,
            a,
            N
        )

        error = percentage_error(
            exact,
            approximation
        )

        if error < ERROR_TOLERANCE:

            return N, error

    return None, None


def tolerance_analysis():

    print("\n" + "=" * 90)
    print("0.1% ERROR TOLERANCE ANALYSIS")
    print("=" * 90)

    print(
        f"\nRequirement: Percentage error < "
        f"{ERROR_TOLERANCE}%"
    )

    print()

    print(
        f"{'Angle':>8}"
        f"{'Maclaurin Terms':>20}"
        f"{'Maclaurin Error':>20}"
        f"{'Taylor Terms':>18}"
        f"{'Taylor Error':>18}"
    )

    print("-" * 90)

    results = []

    for angle in angles_deg:

        m_terms, m_error = (
            find_minimum_terms_maclaurin(angle)
        )

        t_terms, t_error = (
            find_minimum_terms_taylor(angle)
        )

        results.append({
            "angle": angle,
            "maclaurin_terms": m_terms,
            "maclaurin_error": m_error,
            "taylor_terms": t_terms,
            "taylor_error": t_error
        })

        print(
            f"{angle:>8}"
            f"{m_terms:>20}"
            f"{m_error:>20.8f}%"
            f"{t_terms:>18}"
            f"{t_error:>18.8f}%"
        )

    return results


# ============================================================
# SMALL-ANGLE APPROXIMATION
# ============================================================

def small_angle_analysis():

    print("\n" + "=" * 80)
    print("SMALL-ANGLE APPROXIMATION")
    print("=" * 80)

    print("\nsin(theta) ≈ theta")

    print(
        f"{'Angle':>8}"
        f"{'Exact sin':>18}"
        f"{'Approximation':>18}"
        f"{'% Error':>18}"
    )

    print("-" * 65)

    critical_angle = None

    for angle in range(1, 31):

        theta = math.radians(angle)

        exact = math.sin(theta)

        approximation = theta

        error = percentage_error(
            exact,
            approximation
        )

        print(
            f"{angle:>8}"
            f"{exact:>18.10f}"
            f"{approximation:>18.10f}"
            f"{error:>18.10f}"
        )

        if (
            critical_angle is None
            and error >= ERROR_TOLERANCE
        ):

            critical_angle = angle

    print()

    print(
        f"The approximation first reaches/exceeds "
        f"{ERROR_TOLERANCE}% error at approximately "
        f"{critical_angle} degrees."
    )

    print(
        f"At {critical_angle - 1} degrees, it is still "
        f"within the tolerance."
    )

    return critical_angle


# ============================================================
# PLOT 1 - MACLAURIN CONVERGENCE
# ============================================================

def convergence_plot():

    plt.figure(figsize=(10, 6))

    for angle in angles_deg:

        theta = math.radians(angle)

        exact = math.sin(theta)

        terms = list(range(1, 11))

        errors = []

        for N in terms:

            approximation = sin_maclaurin(
                theta,
                N
            )

            error = percentage_error(
                exact,
                approximation
            )

            # Prevent zero from breaking log scale
            if error == 0:
                error = 1e-15

            errors.append(error)

        plt.plot(
            terms,
            errors,
            marker="o",
            label=f"{angle}°"
        )

    plt.yscale("log")

    plt.xlabel("Number of Terms")

    plt.ylabel("Percentage Error (%)")

    plt.title(
        "Maclaurin Series Convergence"
    )

    plt.grid(True)

    plt.legend()

    plt.tight_layout()

    filepath = os.path.join(
        OUTPUT_FOLDER,
        "convergence_plot.png"
    )

    plt.savefig(
        filepath,
        dpi=300
    )

    plt.close()

    print(
        f"Created: {filepath}"
    )


# ============================================================
# PLOT 2 - FUNCTION COMPARISON
# Side-by-side plots
# ============================================================

def function_comparison_plot():

    theta_deg = [
        x / 10
        for x in range(0, 301)
    ]

    theta_rad = [
        math.radians(x)
        for x in theta_deg
    ]

    exact_values = [
        math.sin(theta)
        for theta in theta_rad
    ]

    # Create side-by-side plots
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(15, 6)
    )

    # --------------------------------------------------------
    # LEFT - MACLAURIN
    # --------------------------------------------------------

    axes[0].plot(
        theta_deg,
        exact_values,
        linewidth=2,
        label="Exact sin(theta)"
    )

    for N in term_counts:

        values = [
            sin_maclaurin(theta, N)
            for theta in theta_rad
        ]

        axes[0].plot(
            theta_deg,
            values,
            linestyle="--",
            label=f"Maclaurin - {N} terms"
        )

    axes[0].set_xlabel("Angle (degrees)")

    axes[0].set_ylabel("sin(theta)")

    axes[0].set_title(
        "Maclaurin vs Exact"
    )

    axes[0].grid(True)

    axes[0].legend()

    # --------------------------------------------------------
    # RIGHT - TAYLOR
    # --------------------------------------------------------

    a = math.radians(10)

    axes[1].plot(
        theta_deg,
        exact_values,
        linewidth=2,
        label="Exact sin(theta)"
    )

    for N in term_counts:

        values = [
            sin_taylor(
                theta,
                a,
                N
            )
            for theta in theta_rad
        ]

        axes[1].plot(
            theta_deg,
            values,
            linestyle="--",
            label=f"Taylor - {N} terms"
        )

    axes[1].set_xlabel("Angle (degrees)")

    axes[1].set_ylabel("sin(theta)")

    axes[1].set_title(
        "Taylor vs Exact (center = 10 degrees)"
    )

    axes[1].grid(True)

    axes[1].legend()

    fig.suptitle(
        "Exact Sine vs Series Approximations",
        fontsize=16
    )

    fig.tight_layout()

    filepath = os.path.join(
        OUTPUT_FOLDER,
        "function_comparison_plot.png"
    )

    fig.savefig(
        filepath,
        dpi=300
    )

    plt.close(fig)

    print(
        f"Created: {filepath}"
    )


# ============================================================
# PLOT 3 - ABSOLUTE ERROR COMPARISON
# ============================================================

def absolute_error_comparison_plot():

    maclaurin_results = create_maclaurin_results()

    taylor_results = create_taylor_results()

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(14, 10)
    )

    for index, N in enumerate(term_counts):

        row = index // 2

        col = index % 2

        mac_errors = []

        taylor_errors = []

        for angle in angles_deg:

            for result in maclaurin_results:

                if (
                    result["angle"] == angle
                    and result["terms"] == N
                ):

                    mac_errors.append(
                        result["absolute_error"]
                    )

                    break

            for result in taylor_results:

                if (
                    result["angle"] == angle
                    and result["terms"] == N
                ):

                    taylor_errors.append(
                        result["absolute_error"]
                    )

                    break

        axes[row, col].plot(
            angles_deg,
            mac_errors,
            marker="o",
            label=f"Maclaurin - {N} terms"
        )

        axes[row, col].plot(
            angles_deg,
            taylor_errors,
            marker="s",
            label=f"Taylor - {N} terms"
        )

        axes[row, col].set_xlabel(
            "Angle (degrees)"
        )

        axes[row, col].set_ylabel(
            "Absolute Error (m)"
        )

        axes[row, col].set_title(
            f"{N} Terms"
        )

        axes[row, col].grid(True)

        axes[row, col].legend()

    fig.suptitle(
        "Absolute Error Comparison",
        fontsize=16
    )

    fig.tight_layout()

    filepath = os.path.join(
        OUTPUT_FOLDER,
        "absolute_error_comparison.png"
    )

    fig.savefig(
        filepath,
        dpi=300
    )

    plt.close(fig)

    print(
        f"Created: {filepath}"
    )


# ============================================================
# PLOT 4 - PERCENTAGE ERROR COMPARISON
# ============================================================

def percentage_error_comparison_plot():

    maclaurin_results = create_maclaurin_results()

    taylor_results = create_taylor_results()

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(14, 10)
    )

    for index, N in enumerate(term_counts):

        row = index // 2

        col = index % 2

        mac_errors = []

        taylor_errors = []

        for angle in angles_deg:

            for result in maclaurin_results:

                if (
                    result["angle"] == angle
                    and result["terms"] == N
                ):

                    mac_errors.append(
                        result["percentage_error"]
                    )

                    break

            for result in taylor_results:

                if (
                    result["angle"] == angle
                    and result["terms"] == N
                ):

                    taylor_errors.append(
                        result["percentage_error"]
                    )

                    break

        axes[row, col].plot(
            angles_deg,
            mac_errors,
            marker="o",
            label=f"Maclaurin - {N} terms"
        )

        axes[row, col].plot(
            angles_deg,
            taylor_errors,
            marker="s",
            label=f"Taylor - {N} terms"
        )

        axes[row, col].axhline(
            ERROR_TOLERANCE,
            linestyle="--",
            label="0.1% tolerance"
        )

        axes[row, col].set_xlabel(
            "Angle (degrees)"
        )

        axes[row, col].set_ylabel(
            "Percentage Error (%)"
        )

        axes[row, col].set_title(
            f"{N} Terms"
        )

        axes[row, col].grid(True)

        axes[row, col].legend()

    fig.suptitle(
        "Percentage Error Comparison",
        fontsize=16
    )

    fig.tight_layout()

    filepath = os.path.join(
        OUTPUT_FOLDER,
        "percentage_error_comparison.png"
    )

    fig.savefig(
        filepath,
        dpi=300
    )

    plt.close(fig)

    print(
        f"Created: {filepath}"
    )


# ============================================================
# ENGINEERING RECOMMENDATION
# ============================================================

def generate_recommendation(
    tolerance_results,
    critical_angle
):

    max_mac_terms = max(
        result["maclaurin_terms"]
        for result in tolerance_results
    )

    max_taylor_terms = max(
        result["taylor_terms"]
        for result in tolerance_results
    )

    # Find the error for Maclaurin at 30 degrees
    mac_30_error = None

    for result in create_maclaurin_results():

        if (
            result["angle"] == 30
            and result["terms"] == 2
        ):

            mac_30_error = result["percentage_error"]

    # Find Taylor error at 10 degrees with 1 term
    taylor_10_error = None

    for result in create_taylor_results():

        if (
            result["angle"] == 10
            and result["terms"] == 1
        ):

            taylor_10_error = result["percentage_error"]

    recommendation = f"""
ENGINEERING RECOMMENDATION
===========================

Engineering relationship:

    y = L sin(theta)

Given:

    L = {L:.1f} m

Required accuracy:

    Percentage error < {ERROR_TOLERANCE}%


1. MACLAURIN SERIES
-------------------

The Maclaurin series is centered at theta = 0 degrees.

For the tested angle range of 1 to 30 degrees, increasing
the number of terms improves the approximation.

The results show that a maximum of {max_mac_terms} Maclaurin
terms is sufficient to meet the less-than-{ERROR_TOLERANCE}%
error requirement for all tested angles.

At 30 degrees, the 2-term Maclaurin approximation has an
error of approximately {mac_30_error:.6f}%.


2. TAYLOR SERIES
----------------

The Taylor series is centered at 10 degrees.

Because the expansion point is 10 degrees, the approximation
is especially accurate near 10 degrees.

At 10 degrees, the 1-term Taylor approximation gives the
exact value, with an error of {taylor_10_error:.6f}%.

For the complete 1 to 30 degree range, a maximum of
{max_taylor_terms} Taylor terms is required to meet the
less-than-{ERROR_TOLERANCE}% error requirement.


3. SMALL-ANGLE APPROXIMATION
----------------------------

The approximation

    sin(theta) approximately equal to theta

remains within the {ERROR_TOLERANCE}% tolerance through
approximately {critical_angle - 1} degrees.

At {critical_angle} degrees, the error reaches or exceeds
the required tolerance.

Therefore, the critical angle is approximately
{critical_angle} degrees.


4. ENGINEERING DECISION
-----------------------

For the complete angle range from 1 to 30 degrees, the
Maclaurin series is the more computationally efficient
series approximation because at most {max_mac_terms} terms
are required to satisfy the specified accuracy.

The Taylor series centered at 10 degrees is advantageous
when calculations are concentrated near 10 degrees because
the expansion point is close to the angle of interest.

For this particular problem, the recommended approximation
is therefore the Maclaurin series with up to {max_mac_terms}
terms when the complete 1 to 30 degree range must be covered.

If the calculation is focused specifically around 10 degrees,
a Taylor series centered at 10 degrees can be advantageous.

The exact sine function is the simplest method when direct
evaluation is allowed and an approximation is not required.


5. CONCLUSION
-------------

The results demonstrate that adding terms generally reduces
the approximation error. The rate of convergence depends on
the distance between the angle and the expansion point.

Maclaurin is naturally suited to angles near zero, while a
Taylor series can improve accuracy around its selected
expansion point.

For the specified 0.1% accuracy requirement, the choice of
method should consider the angle range, number of terms,
computational simplicity, and required accuracy.
"""

    filepath = os.path.join(
        OUTPUT_FOLDER,
        "engineering_recommendation.txt"
    )

    with open(
        filepath,
        "w",
        encoding="utf-8"
    ) as file:

        file.write(recommendation)

    print("\n" + "=" * 80)
    print("ENGINEERING RECOMMENDATION")
    print("=" * 80)

    print(recommendation)

    print(
        f"\nSaved recommendation to:\n{filepath}"
    )


# ============================================================
# MAIN PROGRAM
# ============================================================

def main():

    print("\n")
    print("=" * 80)
    print("CIVIL ENGINEERING SERIES EXERCISE")
    print("HOW ACCURATE IS GOOD ENOUGH?")
    print("=" * 80)

    print("\nEngineering application:")
    print("y = L sin(theta)")
    print(f"L = {L} m")
    print(f"Angles = {angles_deg}")
    print(
        f"Required accuracy = less than "
        f"{ERROR_TOLERANCE}% error"
    )

    # --------------------------------------------------------
    # PART 1
    # --------------------------------------------------------

    geometric_series_analysis()

    # --------------------------------------------------------
    # PART 2
    # --------------------------------------------------------

    power_series_demo()

    # --------------------------------------------------------
    # PART 3 / 4
    # --------------------------------------------------------

    maclaurin_results = create_maclaurin_results()

    taylor_results = create_taylor_results()

    print_results_table(
        maclaurin_results,
        "MACLAURIN SERIES RESULTS"
    )

    print_results_table(
        taylor_results,
        "TAYLOR SERIES RESULTS - CENTERED AT 10 DEGREES"
    )

    # --------------------------------------------------------
    # SAVE NUMERICAL TABLES
    # --------------------------------------------------------

    save_results_to_text(
        maclaurin_results,
        taylor_results
    )

    # --------------------------------------------------------
    # PART 6 - TOLERANCE
    # --------------------------------------------------------

    tolerance_results = tolerance_analysis()

    # --------------------------------------------------------
    # SMALL ANGLE
    # --------------------------------------------------------

    critical_angle = small_angle_analysis()

    # --------------------------------------------------------
    # PLOTS
    # --------------------------------------------------------

    print("\n" + "=" * 80)
    print("GENERATING REQUIRED PLOTS")
    print("=" * 80)

    convergence_plot()

    function_comparison_plot()

    absolute_error_comparison_plot()

    percentage_error_comparison_plot()

    # --------------------------------------------------------
    # RECOMMENDATION
    # --------------------------------------------------------

    generate_recommendation(
        tolerance_results,
        critical_angle
    )

    # --------------------------------------------------------
    # COMPLETE
    # --------------------------------------------------------

    print("\n" + "=" * 80)
    print("PROGRAM COMPLETE")
    print("=" * 80)

    print("\nALL OUTPUTS SAVED TO:")

    print(OUTPUT_FOLDER)

    print("\nGenerated files:")

    print("1. series_results.txt")
    print("2. engineering_recommendation.txt")
    print("3. convergence_plot.png")
    print("4. function_comparison_plot.png")
    print("5. absolute_error_comparison.png")
    print("6. percentage_error_comparison.png")

    print("\nYou can now open the 'series_outputs' folder on your Desktop.")


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()