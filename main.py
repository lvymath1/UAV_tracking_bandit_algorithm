import numpy as np

from algorithm.particle_filter import ParticleFilter
from algorithm.previous_position import previous_position
from algorithm.trajectory_prediction import trajectory_prediction
from bandit.bandit_algorithm import Exp4IX
from environment.Target_adversarial import Target_adversarial
from environment.Target_tracking import Target_tracking
from environment.UAV import UAV
from params import frame_num
from plot.animation import animation_video

np.random.seed(0)

def simulation(mode='tracking', algorithm='fusion algorithm'):
    uav = UAV()
    if mode == 'tracking':
        target = Target_tracking(uav.uav_position)
    else:
        target = Target_adversarial(uav.uav_position)

    pf = ParticleFilter(num_particles=1000, uav_position=uav.uav_position, uav_orientation=target.target_position)
    epoch = min(frame_num, len(target.target_positions)) if mode == 'tracking' else frame_num
    exp4ix = Exp4IX(n=epoch, k=4, M=3, delta=0.01)

    for i in range(epoch):
        target.update(uav.uav_position)
        uav.update(target)
        if algorithm == 'previous position algorithm':
            probs = previous_position(target, uav)
        if algorithm == 'particle filter algorithm':
            probs = pf.update(target, uav)
        if algorithm == 'trajectory_prediction':
            probs = trajectory_prediction(target, uav)
        if algorithm == 'fusion algorithm':
            probs_previous = previous_position(target, uav)
            probs_particle_filter = pf.particle_filter(target, uav)
            probs_trajectory = trajectory_prediction(target, uav)
            # Combine both results into one list
            expert_advice = np.vstack((probs_previous, probs_particle_filter, probs_trajectory))
            probs = exp4ix.get_probs(expert_advice)
        uav.move(probs)
        if algorithm == 'fusion algorithm':
            exp4ix.update(uav, target, expert_advice)
    animation_video(target, uav, mode)


simulation_mode = 'adversarial'  # 'tracking', 'adversarial'
algorithm = 'fusion algorithm' # 'fusion algorithm', 'particle filter algorithm', 'previous position algorithm', 'trajectory_prediction'
simulation(mode=simulation_mode, algorithm=algorithm)
