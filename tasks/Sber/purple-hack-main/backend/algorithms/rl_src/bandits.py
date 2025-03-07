import numpy as np


class MultiArmedBandit:
    def __init__(
        self,
        arms_dist: np.ndarray,
    ):
        self.arms_dist = arms_dist

        self.action_space = list(range(len(arms_dist)))
        # the only obs is 0
        self.observation_space = [0]
        self.rng = np.random.default_rng()

    def reset(self, seed=None, options=None):
        self.rng = np.random.default_rng(seed)
        return 0, {}

    def step(self, action: int):
        mean, std = self.arms_dist[action.squeeze()]

        reward = self.rng.normal(loc=mean, scale=std)

        return 0, reward, False, False, {}


class MultiArmedBanditBernoulli:
    def __init__(self, arms_dist: np.ndarray, num_arms: int):
        self.arms_dist = arms_dist
        self.num_arms = num_arms

        self.action_space = list(range(len(arms_dist)))
        # the only obs is 0
        self.observation_space = [0]
        self.rng = np.random.default_rng()

        self.regret = 0

    def reset(self, seed=None, options=None):
        self.rng = np.random.default_rng(seed)
        # we also need to reset regret manually
        self.regret = 0

        return 0, {}

    def step(self, action: int):
        assert action < self.num_arms, (action, self.num_arms)

        # calc reward
        mean = self.arms_dist[action, 0]
        reward = self.rng.binomial(n=1, p=mean)

        # info for calculation of the regret
        opt_mean = self.arms_dist[: self.num_arms, 0].max()
        opt_act = self.arms_dist[: self.num_arms, 0].argmax()

        self.regret += opt_mean - mean
        info = {"regret": self.regret, "opt_act": opt_act}

        return 0, reward, False, False, info


class ContextualBandit:
    def __init__(
        self,
        context_dim: int,
        arm_embeds: np.ndarray,
        num_arms: int,
    ):
        self.arm_embeds = arm_embeds
        self.num_arms = num_arms
        self.context_dim = context_dim

        self.action_space = list(range(len(arm_embeds)))

        # self.observation_space = gym.spaces.Box(
        #     low=-1e20, high=1e20, shape=(context_dim,), dtype=np.float32
        # )
        self.rng = np.random.default_rng()

        self.regret = 0

    def reset(self, seed=None, options=None):
        self.rng = np.random.default_rng(seed)
        self.context = self._get_new_context()
        # we also need to reset regret manually
        self.regret = 0

        return self.context, {}

    def _get_new_context(self):
        return self.rng.normal(size=(self.context_dim,)) / np.sqrt(self.context_dim)

    def step(self, action: int):
        assert action < self.num_arms, (action, self.num_arms)

        all_means = (self.arm_embeds @ self.context)[: self.num_arms]

        # calc reward
        mean = all_means[action]
        reward = self.rng.normal(loc=mean, scale=1)
        # reward = mean

        # info for calculation of the regret
        opt_mean = all_means.max()
        opt_act = all_means.argmax()

        self.regret += opt_mean - mean
        info = {"regret": self.regret, "opt_act": opt_act, "mean": mean}

        self.context = self._get_new_context()

        return self.context, reward, False, False, info