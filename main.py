import numpy as np

from algorithm.average_fusion import average_fusion
from algorithm.particle_filter import ParticleFilter
from algorithm.previous_position import previous_position
from algorithm.trajectory_prediction import trajectory_prediction
from bandit.bandit_algorithm import Exp4IX
from environment.Target_adversarial import Target_adversarial
from environment.Target_tracking import Target_tracking
from environment.UAV import UAV
from params import frame_num
from plot.animation import animation_video
from plot.combine_images import combine_images
from plot.plot_cumulative_distances import plot_cumulative_distances
from plot.plot_q_weights import plot_q_weights


np.random.seed(0)

def run_all_simulations(mode='Smooth Trajectory', generate_animation='on'):
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
        print("Initialized smooth trajectory target.")
    else:
        target = Target_adversarial(uav.uav_position)
        print("Initialized adversarial target.")

    for algorithm in algorithms:
        print(f"Starting simulation in \"{mode}\" mode using \"{algorithm}\" ...")

        pf = ParticleFilter(num_particles=1000, uav_position=uav.uav_position, uav_orientation=target.target_position)
        epoch = min(frame_num, len(target.target_positions)) if mode == 'tracking' else frame_num
        exp4ix = Exp4IX(n=epoch, k=4, M=3, delta=0.01)

        cumulative_distance = 0
        distances_over_time = []

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

            if algorithm == 'Exp4-IX Algorithm':
                exp4ix.update(uav, target, expert_advice)

            # Calculate distance and accumulate
            distance = np.linalg.norm(target.target_position - uav.uav_position)
            cumulative_distance += distance / 100
            distances_over_time.append(cumulative_distance)

        cumulative_distances[algorithm] = distances_over_time

        if generate_animation == 'on':
            print("Simulation complete. Generating animation...")
            animation_video(target, uav, mode, algorithm)
            print("Animation video generated.")

        if algorithm == 'Exp4-IX Algorithm':
            plot_q_weights(exp4ix.Q_history, algorithms[:3])

        uav.reset()
        target.reset()

    # Plot cumulative distances
    plot_cumulative_distances(cumulative_distances)

# Run all simulations
# 'Smooth Trajectory', 'Adversarial Trajectory'
run_all_simulations(mode='Smooth Trajectory', generate_animation='on')
image_files = [
    'experimental_pic/final_frame_Smooth Trajectory_Previous Position Algorithm_setting.jpg',
    'experimental_pic/final_frame_Smooth Trajectory_Particle Filtering Algorithm_setting.jpg',
    'experimental_pic/final_frame_Smooth Trajectory_Average Fusion Algorithm_setting.jpg',
    'experimental_pic/final_frame_Smooth Trajectory_Exp4-IX Algorithm_setting.jpg',
]
combine_images(image_files)