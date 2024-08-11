import numpy as np


def previous_position(target, uav):
    difference = target.target_position - uav.uav_position

    if target.is_target_in_view(target, uav):
        # Calculate probabilities based on the target's position relative to the drone
        probs = np.array([max(0, difference[1]), max(0, -difference[1]),
                          max(0, -difference[0]), max(0, difference[0])])
        probs = probs / probs.sum()  # Normalize to probabilities
    else:
        # Return equal probabilities if the target is not in view
        direction_probs = np.array([0.5, 0.5, 0.5, 0.5])
        direction_probs[uav.uav_orientation] += 1.0
        direction_probs /= direction_probs.sum()  # Normalize to probabilities
        return direction_probs

    return probs
