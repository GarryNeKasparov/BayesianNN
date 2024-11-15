import numpy as np

SEED = 42

GENERATOR = np.random.default_rng(SEED)


def normal_proposal(x):
    return GENERATOR.normal(x, 1)


def normal_trans(x, x_proposal):
    return 1 / np.sqrt(2 * np.pi) * np.exp(-((x_proposal - x) ** 2) / 2)


class MetropolisHastings:
    def __init__(
        self,
        p: callable,
        q_proposal: callable,
        q_trans: callable,
        burn_in: int = 100,
        **kwargs
    ):
        self.p = p
        self.q_proposal = q_proposal
        self.q_trans = q_trans
        self.burn_in = burn_in

    def step(self, x):
        x_proposal = self.q_proposal(x)
        accept = min(
            1,
            self.p(x_proposal)
            * self.q_trans(x, x_proposal)
            / (self.p(x) * self.q_trans(x_proposal, x) + 1e-6),
        )
        if GENERATOR.choice([True, False], p=[accept, 1 - accept]):
            xt = x_proposal
        else:
            xt = x
        return xt

    def process(self, x0: float, n_steps: int):
        samples = []
        xt = x0
        for _ in range(n_steps):
            xt = self.step(xt)
            samples.append(xt)
        samples = np.asarray(samples[self.burn_in :])
        return samples
