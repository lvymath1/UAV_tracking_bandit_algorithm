from matplotlib import pyplot as plt, animation, rcParams
import matplotlib.patches as patches

# 设置字体为英文字体（如 Arial）
rcParams['font.family'] = 'Arial'

def animation_video(target_positions, uav_positions, uav_orientations):
    # 计算最小帧数
    num_frames = min(len(target_positions), len(uav_positions))

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_xlim(0, 10000)
    ax.set_ylim(0, 10000)
    target_line, = ax.plot([], [], 'r-', label='Target Trajectory')
    drone_line, = ax.plot([], [], 'b--', label='Drone Trajectory')
    drone_dot, = ax.plot([], [], 'bo', label='Drone Position')
    target_dot, = ax.plot([], [], 'ro', label='Target Position')

    # Create orientation arrows
    orientation_arrows = {
        0: ('green', 0, 1),  # Up
        1: ('blue', 1, 0),  # Right
        2: ('red', 0, -1),  # Down
        3: ('magenta', -1, 0)  # Left
    }

    arrow_size = 10

    def init():
        target_line.set_data([], [])
        drone_line.set_data([], [])
        drone_dot.set_data([], [])
        target_dot.set_data([], [])

        # Set the legend in the init function
        ax.legend()

        return target_line, drone_line, drone_dot, target_dot

    def update(frame):
        target_line.set_data(target_positions[:frame, 0], target_positions[:frame, 1])
        drone_line.set_data(uav_positions[:frame, 0], uav_positions[:frame, 1])
        drone_dot.set_data(uav_positions[frame, 0], uav_positions[frame, 1])
        target_dot.set_data(target_positions[frame, 0], target_positions[frame, 1])

        # Clear previous orientation arrows
        for artist in reversed(ax.patches):
            artist.remove()

        # Add new orientation arrow
        x0, y0 = uav_positions[frame]
        color, dx, dy = orientation_arrows[uav_orientations[frame]]
        arrow = patches.FancyArrow(x0, y0, dx * arrow_size, dy * arrow_size,
                                   color=color, width=2, head_width=5, head_length=8)
        ax.add_patch(arrow)

        return target_line, drone_line, drone_dot, target_dot

    ani = animation.FuncAnimation(fig, update, frames=num_frames, init_func=init, blit=True, interval=20)

    # Save animation as video file
    ani.save('experimental_video/UAV_tracking.mp4', writer='ffmpeg', fps=20)

    # Show final result plot
    plt.xlabel('X Axis')
    plt.ylabel('Y Axis')
    plt.title('UAV Tracking Target Trajectory')
    plt.grid(True)

    # Save the final frame as an image
    final_frame_image = 'experimental_video/final_frame.jpg'
    plt.savefig(final_frame_image)

