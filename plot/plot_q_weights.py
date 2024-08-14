import matplotlib.pyplot as plt
import numpy as np

def plot_q_weights(Q_histories, algorithms, output_file='experimental_pic\q_weights_comparison.jpg'):
    plt.figure(figsize=(14, 10))

    # Define a color palette
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # Blue, Orange, Green

    # Extract the number of algorithms
    num_algorithms = len(algorithms)

    # Transpose Q_histories so that each row corresponds to a single algorithm over all rounds
    Q_histories = np.array(Q_histories)
    rounds = Q_histories.shape[0]

    # Iterate over each algorithm and plot its Q weight over time
    for i in range(num_algorithms):
        weights = Q_histories[:, i]  # Extract the weights for algorithm i across all rounds
        plt.plot(range(rounds), weights, label=algorithms[i], color=colors[i], linewidth=2.5)

    plt.xlabel('Rounds', fontsize=22, fontweight='bold')
    plt.ylabel('Q values', fontsize=22, fontweight='bold')
    plt.title('Q Weight Changes Over Rounds', fontsize=26, fontweight='bold')
    plt.legend(fontsize=15, loc='upper left', frameon=True, shadow=True)
    plt.grid(True, linestyle='--', alpha=0.7)

    # Set font size for tick labels on both axes
    plt.xticks(fontsize=18, fontweight='bold')
    plt.yticks(fontsize=18, fontweight='bold')

    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Q weights plot saved as {output_file}")
