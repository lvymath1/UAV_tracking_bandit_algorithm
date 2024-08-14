import numpy as np
from matplotlib import pyplot as plt, animation, rcParams
import matplotlib.patches as patches

from params import animation_dpi

# Set font to an English font (e.g., Arial)
rcParams['font.family'] = 'Arial'

def animation_video(target, uav, mode, algorithm, dpi=animation_dpi):
    uav_positions = np.array(uav.uav_positions) / 100  # Convert to km
    target_positions = np.array(target.target_positions) / 100  # Convert to km

    # Calculate the minimum number of frames
    num_frames = min(len(target_positions), len(uav_positions))

    fig, ax = plt.subplots(figsize=(10, 8), dpi=dpi)  # Use dpi parameter for resolution
    ax.set_xlim(0, 100)  # Adjusted for km
    ax.set_ylim(0, 100)  # Adjusted for km
    target_line, = ax.plot([], [], 'r-', label='Target Trajectory')
    uav_line, = ax.plot([], [], 'b--', label='UAV Trajectory')
    uav_dot, = ax.plot([], [], 'bo', label='UAV Position')
    target_dot, = ax.plot([], [], 'ro', label='Target Position')

    def init():
        target_line.set_data([], [])
        uav_line.set_data([], [])
        uav_dot.set_data([], [])
        target_dot.set_data([], [])

        # Set the legend in the init function with larger font size
        ax.legend(fontsize=14)

        return target_line, uav_line, uav_dot, target_dot

    def update(frame):
        target_line.set_data(target_positions[:frame, 0], target_positions[:frame, 1])
        uav_line.set_data(uav_positions[:frame, 0], uav_positions[:frame, 1])
        uav_dot.set_data(uav_positions[frame, 0], uav_positions[frame, 1])
        target_dot.set_data(target_positions[frame, 0], target_positions[frame, 1])

        # Clear previous orientation arrows
        for artist in reversed(ax.patches):
            artist.remove()

        return target_line, uav_line, uav_dot, target_dot

    ani = animation.FuncAnimation(fig, update, frames=num_frames, init_func=init, blit=True, interval=20)

    # Save animation as video file
    ani.save(f'experimental_video/UAV_tracking_{mode}_{algorithm}_setting.mp4', writer='ffmpeg', fps=20)
    print(f'experimental_video/UAV_tracking_{mode}_{algorithm}_setting.mp4')

    # Show final result plot with larger font sizes
    plt.xlabel('X Axis (km)', fontsize=24)
    plt.ylabel('Y Axis (km)', fontsize=24)
    plt.title(f'UAV Tracking Target Trajectory', fontsize=28)
    plt.grid(True)

    # Adjust tick label sizes
    ax.tick_params(axis='both', which='major', labelsize=20)

    final_frame_image = f'experimental_pic/final_frame_{mode}_{algorithm}_setting.jpg'
    plt.savefig(final_frame_image, dpi=dpi)  # Save with the specified dpi
    print(f"Final frame image saved as {final_frame_image}")
