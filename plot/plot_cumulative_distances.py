from matplotlib import pyplot as plt

def plot_cumulative_distances(cumulative_distances, output_file='experimental_pic/cumulative_distances.jpg', max_distance=8000):
    plt.figure(figsize=(10, 10))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

    for i, (algo, distances) in enumerate(cumulative_distances.items()):
        # Filter distances to exclude values beyond max_distance
        filtered_distances = [d for d in distances if d <= max_distance]
        plt.plot(range(len(filtered_distances)), filtered_distances, label=algo, color=colors[i], linewidth=2.5)

    plt.xlabel('Rounds', fontsize=25, fontweight='bold')
    plt.ylabel('Cumulative Distance (km)', fontsize=25, fontweight='bold')
    plt.title('Cumulative Distance Difference Over Time', fontsize=30, fontweight='bold')
    plt.legend(fontsize=20, loc='upper left', frameon=True, shadow=True)
    plt.grid(True, linestyle='--', alpha=0.7)

    # Increase the font size of the tick labels
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)

    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Cumulative distance plot saved as {output_file}")