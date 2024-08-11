import math
import numpy as np

class Exp4IX:
    def __init__(self, n, k, M, delta):
        """
        Initialize the Exp4-IX algorithm.

        :param n: Number of rounds.
        :param k: Number of actions.
        :param M: Number of experts.
        :param delta: Confidence parameter.
        """
        self.n = n
        self.k = k
        self.M = M
        self.delta = delta
        self.eta = math.sqrt(2 * (math.log(M) + math.log(k + 1) - math.log(delta)) / (n * k))
        self.gamma = self.eta / 2

        self.Q = np.ones(M) / M  # Initialize Q_1 as a uniform distribution over experts
        self.S = np.zeros(M)  # Initialize cumulative rewards S_0

    def get_probs(self, E_t):
        P_t = np.dot(E_t.T, self.Q)
        return P_t

    def reward(self, uav, target):
        difference = target.target_position - uav.uav_position

        if target.is_target_in_view(target, uav):
            return 1 / np.linalg.norm(difference)
        return 0

    def update(self, uav, target, E_t):

        P_t = self.get_probs(E_t)
        A_t = uav.uav_orientation

        Y_t = 1 - self.reward(uav, target)
        reward_vector = np.ones(self.k)
        reward_vector[A_t] = Y_t
        hat_Y_t = reward_vector / (P_t + self.gamma)
        tilde_Y_t = np.dot(E_t, hat_Y_t)
        self.S += tilde_Y_t
        self.Q = np.exp(-self.eta * self.S)
        self.Q /= np.sum(self.Q)
