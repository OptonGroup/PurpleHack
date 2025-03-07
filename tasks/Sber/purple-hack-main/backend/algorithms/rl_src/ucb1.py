import numpy as np


class UCB:
    def __init__(self, n_arms):
        self.M = n_arms
        self.c = np.zeros(n_arms, dtype=int) # counter (the number of times each acrm was played)
        self.v = np.zeros(n_arms) # average reward obtained from playing the arms
        self.eps = 1e-6
    
    def select_arm(self):
        for arm in range(self.M):
            if not self.c[arm]:
                return arm
        
        u = np.zeros(self.M)
        c = self.c.sum()

        bonus = np.sqrt((2 * np.log(c)) / (self.c + self.eps))
        u = self.v + bonus
        return u.argmax()
    
    def update(self, arm, reward):
        self.c[arm] += 1
        c = self.c[arm]
        v_a = ((c - 1) / c) * v_a + (reward / c)
        
        self.v[arm] = v_a
