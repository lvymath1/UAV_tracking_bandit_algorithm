import numpy as np
import matplotlib.pyplot as plt


# 粒子滤波器类
class ParticleFilter:
    def __init__(self, num_particles, process_noise, measurement_noise):
        self.num_particles = num_particles
        self.process_noise = process_noise
        self.measurement_noise = measurement_noise
        self.particles = np.random.rand(num_particles, 2) * 100  # 初始化粒子

    def predict(self, control_input):
        # 更新粒子位置
        self.particles += control_input + np.random.randn(self.num_particles, 2) * self.process_noise

    def update(self, measurement):
        # 计算权重
        distances = np.linalg.norm(self.particles - measurement, axis=1)
        weights = np.exp(-distances ** 2 / (2 * self.measurement_noise ** 2))
        weights /= np.sum(weights)  # 归一化权重

        # 重新采样
        indices = np.random.choice(range(self.num_particles), size=self.num_particles, p=weights)
        self.particles = self.particles[indices]

    def get_estimate(self):
        # 返回粒子的位置均值作为估计位置
        return np.mean(self.particles, axis=0)


# 模拟目标轨迹
def generate_trajectory(num_steps):
    trajectory = []
    x, y = 50, 50
    for _ in range(num_steps):
        x += np.random.randn() * 2
        y += np.random.randn() * 2
        trajectory.append((x, y))
    return np.array(trajectory)


# 主函数
def main():
    num_particles = 100
    process_noise = 1.0
    measurement_noise = 5.0
    num_steps = 50

    # 初始化粒子滤波器
    pf = ParticleFilter(num_particles, process_noise, measurement_noise)

    # 生成目标轨迹
    true_trajectory = generate_trajectory(num_steps)

    # 初始化测量噪声
    noisy_measurements = true_trajectory + np.random.randn(num_steps, 2) * measurement_noise

    # 存储估计轨迹
    estimated_trajectory = []

    for measurement in noisy_measurements:
        pf.predict(control_input=np.array([0, 0]))
        pf.update(measurement)
        estimated_trajectory.append(pf.get_estimate())

    # 绘制结果
    true_trajectory = np.array(true_trajectory)
    estimated_trajectory = np.array(estimated_trajectory)

    plt.figure(figsize=(10, 6))
    plt.plot(true_trajectory[:, 0], true_trajectory[:, 1], label='True Trajectory', color='g', marker='o')
    plt.plot(estimated_trajectory[:, 0], estimated_trajectory[:, 1], label='Estimated Trajectory', color='r',
             linestyle='--')
    plt.scatter(noisy_measurements[:, 0], noisy_measurements[:, 1], label='Noisy Measurements', color='b', s=20)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.title('Particle Filter Trajectory Estimation')
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    main()
