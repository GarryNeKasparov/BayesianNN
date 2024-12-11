import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import quad
from scipy.stats import norm

alpha = 3
beta = 7


def f(x: np.ndarray) -> float:
    global alpha, beta
    return x ** (alpha - 1) * (1 - x) ** (beta - 1)


def estimate_integral(
    f: callable, a: float, b: float, confidence: float = 0.9, tol: float = 1e-3
):
    """
    Estimate an integral using Monte Carlo integration.

    Parameters
    ----------
    f : callable
        The function to integrate.
    a : float
        The lower bound of the integral.
    b : float
        The upper bound of the integral.
    confidence : float, optional
        The desired confidence level of the estimate.
        Must be in the range [0, 1).
    tol : float, optional
        The desired tolerance of the estimate.

    Returns
    -------
    estimate : float
        The estimated integral.
    """

    volume = b - a
    q = norm.ppf((1 + confidence) / 2)
    print(q)
    integral = quad(f, a, b)[0]

    mean_estimate = 0
    variance_estimate = 0
    n = 1
    estimates = []
    errors = []
    error = float("inf")
    while error > tol or n < 100:
        x = np.random.uniform(a, b)
        mean_estimate_next = mean_estimate + (f(x) - mean_estimate) / (n + 1)
        variance_estimate = (1 - 1 / n) * variance_estimate + (n + 1) * (
            mean_estimate_next - mean_estimate
        ) ** 2
        mean_estimate = mean_estimate_next
        estimate = volume * mean_estimate
        error = q * volume * np.sqrt(variance_estimate / n)
        estimates.append(estimate)
        errors.append(error)
        n += 1
    print(variance_estimate)
    estimates = np.asarray(estimates)
    errors = np.asarray(errors)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(range(1, n), estimates, label="Оценка интеграла")
    ax.fill_between(
        range(1, n),
        estimates + errors,
        estimates - errors,
        alpha=0.3,
        label="Доверительный интервал",
    )
    ax.hlines(integral, 1, n, color="red", label="Истинное значение")
    # plt.xscale("log")
    # plt.yscale("log")
    plt.xlabel("Номер шага")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.show()
    fig.savefig("./src/int_run.png", bbox_inches="tight")


if __name__ == "__main__":
    estimate_integral(f, 0, 1, confidence=0.95, tol=1e-4)
