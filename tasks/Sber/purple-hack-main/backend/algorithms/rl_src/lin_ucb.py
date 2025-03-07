import random
from typing import Union
import numpy as np


RewardType = Union[float, np.ndarray]


class DisjointArm:
    def __init__(self,
                 arm_idx: int,
                 dim: int,
                 alpha: float) -> None:
        self.arm_idx = arm_idx
        self.dim = dim
        self.alpha = alpha

        self.A = np.identity(dim)
        self.b = np.zeros([dim, 1])
    
    def calc_ucb(self, features: np.ndarray) -> np.ndarray:
        A_inverted = np.linalg.inv(self.A)
        
        self.theta = np.dot(A_inverted, self.b)

        # [dim, 1]
        features = features.reshape((-1, 1))

        # prob of choosing action with upper confidence bound
        p = np.dot(self.theta.T, features) + self.alpha * np.sqrt(np.dot(features.T, np.dot(A_inverted, features)))

        return p
    
    def reward_update(self,
                      reward: RewardType,
                      features: np.ndarray) -> None:
        # [dim, 1]
        features = features.reshape((-1, 1))

        self.A += np.dot(features, features.T)
        self.b += reward * features


class LinUCB:
    def __init__(self,
                 key,
                 num_arms: int,
                 dim: int,
                 alpha: float) -> None:
        self.key = key
        self.num_arms = num_arms
        
        self.arms = [DisjointArm(idx, dim, alpha) for idx in range(num_arms)]
    
    def select_arm(self, features: np.ndarray) -> int:
        high_ucb = -1
        candidates = np.empty(0)

        for arm_idx in range(self.num_arms):
            arm_ucb = self.arms[arm_idx].calc_ucb(features)

            if arm_ucb > high_ucb:
                high_ucb = arm_ucb
                candidates = np.array([arm_idx])
            
            if arm_ucb == high_ucb:
                np.append(candidates, arm_idx)


        # chosen_arm = np.random.choice(subkey, candidates)
        chosen_arm = random.choice(candidates)

        return chosen_arm

    def reward_update(self,
                      arm_idx: int,
                      reward: RewardType,
                      features: np.ndarray) -> None:
        self.arms[arm_idx].reward_update(reward, features)