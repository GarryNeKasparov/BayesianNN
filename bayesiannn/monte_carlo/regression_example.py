import random
from concurrent.futures import ThreadPoolExecutor

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from bayesiannn.constaints import SEED

np.random.seed(SEED)
random.seed(SEED)
H = 20

# let theta be the vector of weights we want to find
# we need to define [prior] p(theta)
# it's our asumption about theta's distribution
# for example, p(theta) = N(0, I) pdf


def prior(thetas: np.ndarray):
    S = np.eye(thetas.shape[0])
    log_normalization_constant = -0.5 * (
        H * np.log(2 * np.pi) + np.log(np.linalg.det(S))
    )
    exponent = -0.5 * (thetas.T @ np.linalg.inv(S) @ thetas)
    return exponent - log_normalization_constant


def functional_model(x: np.ndarray, thetas: np.ndarray):
    W1 = thetas[:H].reshape(1, H)
    b1 = thetas[H]
    W2 = thetas[H + 1 : -1].reshape(H, 1)
    b2 = thetas[-1]
    z = x @ W1 + b1
    z = np.maximum(0, z)
    return z @ W2 + b2


# also we need to define [predictive power/confidence/likelihood]
# p(y | x, theta)
# this is how we get know we are right
# lets say p(y | x, theta) = epx(-MSE(y, f(x, theta)))


def likelihood(y, pred, sigma=1.0):
    n = len(y)
    residuals = y - pred

    # Логарифм функции правдоподобия
    log_likelihood = (
        -0.5 * n * np.log(2 * np.pi)
        - n * np.log(sigma)
        - 0.5 * np.sum((residuals / sigma) ** 2)
    )
    return log_likelihood


def get_model(thetas: np.ndarray):
    return lambda x: functional_model(x, thetas)


# we want to find p(theta | data) but it is a scary integral ;)
# so we use MH algorithm

# we will sample from unknown distribution
# called [posterior] p(theta | D)
# but we know it with accuracy up to a constant (which is an scary integral)
# so p(theta | D) = p(y | x, theta) * p(theta) * const
# where p(theta) is prior (gaussian in this example)

# define random initial guess - theta0
# define [proposal distribution] Q(theta' | theta)
# if theta' is more likely than theta0 is more likely than θ according to
# the target distribution, it is accepted.


# what is proposal distibution?
# it's how we move from one point to another (to proposal point)
# for example, we could use normal proposal distribution
# Q(theta' | theta) ~ N(theta, sigma^2)
# so we'll appear somewhere near theta
# also common choice is U(theta - eps, theta + eps)


# also we need likelihood of this proposal step
# this is also Q(theta' | theta) in literature
# but actually they are not the same...
def proposal_likelihood(theta_p: np.ndarray, theta_q: np.ndarray):
    S = np.eye(theta_p.shape[0])
    dist = theta_p - theta_q
    log_normalization_constant = -0.5 * (
        H * np.log(2 * np.pi) + np.log(np.linalg.det(S))
    )
    exponent = -0.5 * dist.T @ np.linalg.inv(S) @ dist
    return exponent - log_normalization_constant


def proposal(theta0: np.ndarray):
    return np.random.normal(theta0, 0.05)


def parallel_metropolis_hastings(
    x: np.ndarray, y: np.ndarray, num_chains=4, iterations=10000
):
    results = []

    def run_chain():
        return metropolis_hastings(x, y, iterations)

    with ThreadPoolExecutor(max_workers=num_chains) as executor:
        futures = [executor.submit(run_chain) for _ in range(num_chains)]
        print(len(futures))
        for future in futures:
            results.append(future.result())

    # Объединение цепочек
    all_thetas = np.asarray([res[0] for res in results])
    total_accept = np.asarray([res[1] for res in results])
    total_reject = np.asarray([res[2] for res in results])
    return all_thetas, total_accept, total_reject


def metropolis_hastings(
    x: np.ndarray, y: np.ndarray, iterations: int, verbose: bool = False
):
    """
    Perform Metropolis Hastings algorithm to sample from the posterior
    distribution of model parameters.

    Parameters
    ----------
    x : np.ndarray
        Input data.
    y : np.ndarray
        Output data.
    iterations : int
        Number of iterations to perform.
    verbose : bool, optional
        Whether to print progress information. Defaults to False.

    Returns
    -------
    thetas : np.ndarray
        Sampled model parameters.
    accept : int
        Number of accepted proposals.
    reject : int
        Number of rejected proposals.
    """
    theta_n = np.random.randn(H * 2 + 2)
    thetas = []
    reject = 0
    accept = 0
    if verbose:
        loop = tqdm(True)
    while reject + accept <= iterations:
        if verbose:
            loop.set_description(
                (
                    f"accept: {accept}, reject: {reject}, "
                    f"rate: {round(100*accept / (reject + accept+1e-6), 2)}%"
                )
            )
        theta_p = proposal(theta_n)
        m_p = get_model(theta_p)
        m_n = get_model(theta_n)

        # our posterior f = p(y | x, theta) = m(x) * prior
        posterior_p = (
            likelihood(y, m_p(x))
            + proposal_likelihood(theta_p, theta_n)
            + prior(theta_p)
        )
        posterior_q = (
            likelihood(y, m_n(x))
            + proposal_likelihood(theta_n, theta_p)
            + prior(theta_n)
        )
        if posterior_p - posterior_q >= 0:
            p = 1
        else:
            p = min(1, np.exp(posterior_p - posterior_q))
        if np.random.choice([True, False], p=[p, 1 - p]):
            theta_n = theta_p
            accept += 1
        else:
            reject += 1
        thetas.append(theta_n)
    thetas = np.asarray(thetas[-int(len(thetas) // 2) :])
    return thetas, accept, reject


def plot_chains(x: np.ndarray, y: np.ndarray, chains: list):
    """
    Plot the prediction chains for given data and model parameters.

    Parameters
    ----------
    x : np.ndarray
        Input data.
    y : np.ndarray
        True output data.
    chains : list
        List of parameter chains to generate predictions.

    Returns
    -------
    None
    """
    ncols = 2
    nrows = len(chains) // ncols
    fig, ax = plt.subplots(nrows, ncols, figsize=(8, 7))
    if nrows == 1:
        ax = [ax]

    for i, chain in enumerate(chains):
        row = i // ncols
        col = i % ncols
        predictions = np.asarray([get_model(theta)(x) for theta in chain])
        mean_prediction = np.mean(predictions, axis=0)
        upper_bound = np.percentile(predictions, 97.5, axis=0)
        lower_bound = np.percentile(predictions, 2.5, axis=0)

        ax[row][col].plot(x, y, color="blue", label="Истинные данные")
        ax[row][col].plot(
            x, mean_prediction, label="Средние предсказания", color="orange"
        )
        ax[row][col].plot(x, upper_bound, color="red", linestyle="--")
        ax[row][col].plot(x, lower_bound, color="red", linestyle="--")
        ax[row][col].fill_between(
            x.flatten(),
            lower_bound.flatten(),
            upper_bound.flatten(),
            color="green",
            alpha=0.2,
        )
        ax[row][col].legend()

    plt.show()
    fig.savefig("bayesiannn/monte_carlo/src/chains.png")


def plot_params(chains: list, k: int = 0):
    """
    Plot the evolution of a specific parameter from multiple chains.

    Parameters
    ----------
    chains : list
        A list of chains, where each chain is a sequence of parameter vectors.
    k : int, optional
        The index of the parameter to plot, by default 0.

    Returns
    -------
    None
    """
    ncols = 2
    nrows = len(chains) // ncols
    fig, ax = plt.subplots(nrows, ncols, figsize=(8, 7))

    if nrows == 1:
        ax = [ax]

    for i, chain in enumerate(chains):
        row = i // ncols
        col = i % ncols

        chain = np.asarray([c[k] for c in chain])

        ax[row][col].plot(
            range(len(chain)), chain, label=f"Параметр {k}", color="orange"
        )
        ax[row][col].legend()

    plt.show()
    fig.savefig("bayesiannn/monte_carlo/src/thetas.png")


def main():
    x = np.linspace(1, 5, 100).reshape(-1, 1)
    y = 1 / 3 * (x - 3) ** 2 + 0.1 * np.random.randn(100, 1)
    thetas, _, _ = parallel_metropolis_hastings(x, y, iterations=5000)
    plot_chains(x, y, thetas)
    plot_params(thetas)


if __name__ == "__main__":
    main()
