import numpy as np
from scipy.interpolate import splprep, splev
from scipy.spatial.distance import euclidean

from params import min_dist, max_dist

def generate_smooth_trajectory(drone_position, num_control_points=20, num_points=1000, max_frame_distance=30, max_attempts=100):
    attempt = 0
    while attempt < max_attempts:
        try:
            # Generate the first control point near the drone_position
            control_points = [generate_first_control_point_near(drone_position)]

            while len(control_points) < num_control_points:
                new_point = generate_random_point_near_last(control_points[-1])
                # Ensure that the new point is within the specified range and distance constraints
                if (2000 <= new_point[0] <= 8000 and
                        2000 <= new_point[1] <= 8000 and
                        euclidean(new_point, control_points[-1]) <= max_dist):
                    control_points.append(new_point)

            control_points = np.array(control_points)

            if len(control_points) < 4:
                raise ValueError("Insufficient control points for spline fitting. Need at least 4 control points.")

            # Create a smooth trajectory using splines
            tck, u = splprep([control_points[:, 0], control_points[:, 1]], s=0)

            u_new = np.linspace(u.min(), u.max(), num_points)
            x_new, y_new = splev(u_new, tck, der=0)
            trajectory = np.vstack((x_new, y_new)).T

            # Check distances between consecutive points
            refined_trajectory = [trajectory[0]]
            for i in range(1, len(trajectory)):
                dist = euclidean(trajectory[i], trajectory[i - 1])
                if dist > max_frame_distance:
                    # If the distance is too large, add intermediate points
                    num_extra_points = int(np.ceil(dist / max_frame_distance))
                    for j in range(1, num_extra_points):
                        intermediate_point = trajectory[i - 1] + j * (trajectory[i] - trajectory[i - 1]) / num_extra_points
                        refined_trajectory.append(intermediate_point)
                refined_trajectory.append(trajectory[i])

            return np.array(refined_trajectory)

        except Exception as e:
            print(f"Error in trajectory generation: {e}")
            attempt += 1
            print(f"Retrying ({attempt}/{max_attempts})...")

    raise RuntimeError("Failed to generate a valid trajectory after several attempts.")

def generate_first_control_point_near(uav_position):
    point = np.random.uniform(
        low=np.array(uav_position) - [200, 200],
        high=np.array(uav_position) + [200, 200]
    )
    return np.clip(point, [2000, 2000], [8000, 8000])

def generate_random_point_near_last(last_point):
    angle = np.random.uniform(0, 2 * np.pi)
    radius = np.random.uniform(min_dist, max_dist)
    x = last_point[0] + radius * np.cos(angle)
    y = last_point[1] + radius * np.sin(angle)
    point = np.array([x, y])
    return np.clip(point, [2000, 2000], [8000, 8000])
