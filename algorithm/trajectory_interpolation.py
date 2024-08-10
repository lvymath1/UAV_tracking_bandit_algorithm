import numpy as np


def is_target_in_view(target_position, uav_position, uav_orientation):
    difference = target_position - uav_position

    if uav_orientation == 0:  # Up
        return (difference[1] > 0) and (abs(difference[0]) <= 800) and (difference[1] <= 1000)
    elif uav_orientation == 1:  # Right
        return (difference[0] > 0) and (abs(difference[1]) <= 800) and (difference[0] <= 1000)
    elif uav_orientation == 2:  # Down
        return (difference[1] < 0) and (abs(difference[0]) <= 800) and (abs(difference[1]) <= 1000)
    elif uav_orientation == 3:  # Left
        return (difference[0] < 0) and (abs(difference[1]) <= 800) and (abs(difference[0]) <= 1000)
    else:
        return False


def trajectory_interpolation(target_position, uav_position, uav_orientation):
    difference = target_position - uav_position

    if is_target_in_view(target_position, uav_position, uav_orientation):
        # Calculate probabilities based on the target's position relative to the drone
        probs = np.array([max(0, difference[1]), max(0, -difference[1]),
                          max(0, -difference[0]), max(0, difference[0])])
        probs = probs / probs.sum()  # Normalize to probabilities
    else:
        # Return equal probabilities if the target is not in view
        direction_probs = np.array([0.5, 0.5, 0.5, 0.5])
        direction_probs[uav_orientation] += 1.0
        direction_probs /= direction_probs.sum()  # Normalize to probabilities
        return direction_probs

    return probs
