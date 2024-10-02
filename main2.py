import numpy as np
import matplotlib.pyplot as plt

from algorithm.average_fusion import average_fusion
from algorithm.particle_filter import ParticleFilter
from algorithm.previous_position import previous_position
from algorithm.trajectory_prediction import trajectory_prediction
from bandit.bandit_algorithm import Exp4IX
from environment.Target_adversarial import Target_adversarial
from environment.Target_tracking import Target_tracking
from environment.UAV import UAV
from params import frame_num

np.random.seed(1)

def run_single_experiment(mode='Smooth Trajectory'):
    algorithms = [
        'Previous Position Algorithm',
        'Particle Filtering Algorithm',
        'Trajectory Fitting Algorithm',
        'Average Fusion Algorithm',
        'Exp4-IX Algorithm'
    ]
    cumulative_distances = {algo: [] for algo in algorithms}

    uav = UAV()
    if mode == 'Smooth Trajectory':
        target = Target_tracking(uav.uav_position)
    else:
        target = Target_adversarial(uav.uav_position)

    for algorithm in algorithms:
        pf = ParticleFilter(num_particles=1000, uav_position=uav.uav_position, uav_orientation=target.target_position)
        epoch = min(frame_num, len(target.target_positions)) if mode == 'Smooth Trajectory' else frame_num
        exp4ix = Exp4IX(n=epoch, k=4, M=3, delta=0.01)
        uav.reset()
        target.reset()

        cumulative_distance = 0

        for i in range(epoch):
            target.update(uav.uav_position)
            uav.update(target)

            if algorithm == 'Previous Position Algorithm':
                probs = previous_position(target, uav)
            elif algorithm == 'Particle Filtering Algorithm':
                probs = pf.particle_filter(target, uav)
            elif algorithm == 'Trajectory Fitting Algorithm':
                probs = trajectory_prediction(target, uav)
            elif algorithm == 'Average Fusion Algorithm':
                probs_previous = previous_position(target, uav)
                probs_particle_filter = pf.particle_filter(target, uav)
                probs_trajectory = trajectory_prediction(target, uav)
                expert_advice = np.vstack((probs_previous, probs_particle_filter, probs_trajectory))
                probs = average_fusion(expert_advice)
            elif algorithm == 'Exp4-IX Algorithm':
                probs_previous = previous_position(target, uav)
                probs_particle_filter = pf.particle_filter(target, uav)
                probs_trajectory = trajectory_prediction(target, uav)
                expert_advice = np.vstack((probs_previous, probs_particle_filter, probs_trajectory))
                probs = exp4ix.get_probs(expert_advice)

            uav.move(probs)
            distance = np.linalg.norm(uav.uav_position - target.target_position)
            cumulative_distance += distance / 100
            cumulative_distances[algorithm].append(cumulative_distance)

            if algorithm == 'Exp4-IX Algorithm':
                exp4ix.update(uav, target, expert_advice)

    return cumulative_distances

def run_monte_carlo_experiments(mode='tracking', num_experiments=50):
    algorithms = [
        'Previous Position Algorithm',
        'Particle Filtering Algorithm',
        'Trajectory Fitting Algorithm',
        'Average Fusion Algorithm',
        'Exp4-IX Algorithm'
    ]
    all_distances = {algo: [] for algo in algorithms}

    for epoch in range(num_experiments):
        print(epoch + 1, "/", num_experiments)
        cumulative_distances = run_single_experiment(mode)
        for algo in algorithms:
            all_distances[algo].append(cumulative_distances[algo])

    # Calculate mean and standard deviation across all experiments
    avg_distances = {algo: np.mean(all_distances[algo], axis=0) for algo in algorithms}
    std_distances = {algo: np.std(all_distances[algo], axis=0) for algo in algorithms}

    return avg_distances, std_distances

def plot_avg_cumulative_distances(avg_distances, std_distances, output_file='experimental_pic/avg_cumulative_distances.jpg'):
    plt.figure(figsize=(14, 10))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

    for i, algo in enumerate(avg_distances):
        avg = avg_distances[algo]
        std = std_distances[algo]
        lower_bound = np.maximum(avg - std, 0)  # Ensure no negative values
        upper_bound = avg + std

        plt.plot(range(len(avg)), avg, label=algo, color=colors[i], linewidth=2.5)
        plt.fill_between(range(len(avg)), lower_bound, upper_bound, color=colors[i], alpha=0.3)

    plt.xlabel('Rounds', fontsize=22, fontweight='bold')
    plt.ylabel('Cumulative Distance (km)', fontsize=22, fontweight='bold')
    plt.title('Cumulative Distance', fontsize=26, fontweight='bold')
    plt.legend(fontsize=15, loc='upper left', frameon=True, shadow=True)
    plt.grid(True, linestyle='--', alpha=0.7)

    plt.xticks(fontsize=18, fontweight='bold')
    plt.yticks(fontsize=18, fontweight='bold')

    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Average cumulative distance plot saved as {output_file}")

    plt.show()

# Run the Monte Carlo simulations and plot results
simulation_mode = 'Adversarial Trajectory'
avg_distances, std_distances = run_monte_carlo_experiments(mode=simulation_mode, num_experiments=100)
plot_avg_cumulative_distances(avg_distances, std_distances)