import math
import numpy as np
from algorithm.particle_filter import is_target_in_view

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
        """
        Get the action probabilities based on current expert advice.

        :param E_t: Expert advice matrix of shape (k, M).
        :return: Action probability distribution P_t of shape (k,).
        """
        P_t = np.dot(E_t.T, self.Q)
        return P_t

    def reward(self, uav_position, uav_orientation, target_position):
        """
        Compute the reward based on the UAV's position, orientation, and the target's position.

        :param uav_position: The current position of the UAV.
        :param uav_orientation: The current orientation of the UAV.
        :param target_position: The position of the target.
        :return: Reward value, where a higher value indicates the target is within view.
        """
        difference = target_position - uav_position

        if is_target_in_view(target_position, uav_position, uav_orientation):
            return 1 / np.linalg.norm(difference)
        return 0

    def update(self, E_t, uav_position, uav_orientation, target_position, A_t):
        """
        Update the Exp4-IX algorithm based on the expert advice, UAV position, orientation, and action taken.

        :param E_t: Expert advice matrix for the current round.
        :param uav_position: The current position of the UAV.
        :param uav_orientation: The current orientation of the UAV.
        :param target_position: The position of the target.
        :param A_t: The action taken by the UAV.
        :return: None. Updates the internal state of the algorithm.
        """

        P_t = self.get_probs(E_t)
        A_t = np.random.choice(self.k, p=P_t)

        Y_t = 1 - self.reward(uav_position, uav_orientation, target_position)
        reward_vector = np.ones(self.k)
        reward_vector[A_t] = Y_t
        hat_Y_t = reward_vector / (P_t + self.gamma)
        tilde_Y_t = np.dot(E_t, hat_Y_t)
        self.S += tilde_Y_t
        self.Q = np.exp(-self.eta * self.S)
        self.Q /= np.sum(self.Q)
