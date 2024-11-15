import numpy as np
import matplotlib.pyplot as plt
from bayesiannn.mh import MetropolisHastings, normal_proposal, normal_trans


def normal_pdf(x: float, mu: float, sigma: float) -> float:
    """
    Simple normal distribution.

    Args:
        x : The value at which to evaluate the probability density function.
        mu : The mean of the distribution.
        sigma: The standard deviation of the distribution.

    Returns
        The probability density function value at x.
    """
    return (
        1
        / (sigma * np.sqrt(2 * np.pi))
        * np.exp(-0.5 * ((x - mu) / sigma) ** 2)
    )


class Distribution:
    def __init__(
        self, pdf: callable, lower_bound: float, upper_bound: float, **kwargs
    ):
        self.pdf = lambda x: pdf(x, **kwargs)
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.kwargs = kwargs

    def sample(self, n_samples: int = 100):
        mh = MetropolisHastings(
            p=self,
            q_proposal=normal_proposal,
            q_trans=normal_trans,
            burn_in=n_samples / 2,
            **self.kwargs
        )
        return mh.process((self.lower_bound + self.upper_bound) / 2, n_samples)

    def __call__(self, x) -> float:
        return self.pdf(x) * int(self.lower_bound <= x <= self.upper_bound)


class Sampler:
    def __init__(self, distribution: Distribution):
        self.distribution = distribution

    def get_samples(self, n_samples: int = 100) -> np.ndarray:
        space = np.linspace(
            self.distribution.lower_bound,
            self.distribution.upper_bound,
            n_samples,
        )
        samples = np.asarray([self.distribution(x) for x in space])
        return samples, space


def main():
    mu = 0
    sigma = 1
    distribution = Distribution(
        normal_pdf, lower_bound=-10, upper_bound=10, mu=mu, sigma=sigma
    )
    sampler = Sampler(distribution)
    samples, space = sampler.get_samples(100)
    plt.plot(space, samples)
    plt.show()


if __name__ == "__main__":
    main()
