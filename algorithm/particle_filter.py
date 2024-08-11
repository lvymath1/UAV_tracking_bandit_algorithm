import numpy as np


class ParticleFilter:
    def __init__(self, num_particles, uav_position, uav_orientation):
        self.num_particles = num_particles
        self.particles = np.tile(uav_position, (num_particles, 1))
        self.weights = np.ones(num_particles) / num_particles
        self.uav_orientation = uav_orientation
        self.noise = np.random.randn(self.num_particles, 2) * 10

    def predict(self):  # 适当减少噪声
        self.particles += self.noise

    def update(self, target):
        distances = np.linalg.norm(self.particles - target.target_position, axis=1)
        self.weights = np.exp(-distances)
        self.weights += 1.e-300
        self.weights /= np.sum(self.weights)

    def resample(self):
        indices = np.random.choice(range(self.num_particles), self.num_particles, p=self.weights)
        self.particles = self.particles[indices]
        self.weights = self.weights[indices]
        self.weights /= np.sum(self.weights)

    def estimate(self):
        return np.average(self.particles, weights=self.weights, axis=0)

    def particle_filter(self, target, uav):
        self.predict()
        self.update(target)
        self.resample()
        estimated_target_position = self.estimate()

        difference = estimated_target_position - uav.uav_position

        if target.is_target_in_view(target, uav):
            probs = np.array([max(0, difference[1]), max(0, -difference[1]),
                              max(0, -difference[0]), max(0, difference[0])])
            probs = probs / probs.sum()
        else:
            direction_probs = np.array([0.5, 0.5, 0.5, 0.5])
            direction_probs[uav.uav_orientation] += 1.0
            direction_probs /= direction_probs.sum()
            return direction_probs

        return probs