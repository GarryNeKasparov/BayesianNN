from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

H = 20

# let theta be the vector of weights we want to find
# we need to define [prior] p(theta)
# it's our asumption about theta's distribution
# for example, p(theta) = N(0, I) pdf


def prior(thetas: np.ndarray):
    """
    Априорное распределение p(theta) предполагается нормальным N(0, I).
    """
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
    return np.random.normal(theta0, 0.1)


def main():
    x = np.linspace(1, 5, 100).reshape(-1, 1)
    y = 1 / 3 * (x - 3) ** 2 + 0.1 * np.random.randn(100, 1)
    theta_n = np.random.randn(H * 2 + 2)

    thetas = []
    reject = 0
    accept = 0
    loop = tqdm(True)
    while reject + accept <= 5000:
        loop.set_description(f"accept: {accept}, reject: {reject}")
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

    thetas = np.asarray(thetas[-4000:])
    predictions = [get_model(theta)(x) for theta in thetas]
    mean_prediction = np.mean(predictions, axis=0)
    upper_bound = np.percentile(predictions, 95, axis=0)
    lower_bound = np.percentile(predictions, 5, axis=0)
    plt.plot(x, y, color="blue", label="Истинные данные")
    plt.plot(x, mean_prediction, label="Средние предсказания", color="orange")
    plt.plot(
        x, upper_bound, label="Верхняя граница", color="red", linestyle="--"
    )
    plt.plot(
        x, lower_bound, label="Нижняя граница", color="red", linestyle="--"
    )
    plt.fill_between(
        x.flatten(),
        lower_bound.flatten(),
        upper_bound.flatten(),
        color="green",
        alpha=0.2,
    )
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
