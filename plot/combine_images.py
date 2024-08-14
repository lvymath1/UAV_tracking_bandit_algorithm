import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.gridspec import GridSpec

def combine_images(image_files, output_file='combined_image.jpg'):
    # Create a figure with a custom grid layout
    fig = plt.figure(figsize=(16, 8))  # Adjust figure size to make room for large image
    gs = GridSpec(2, 3, figure=fig, width_ratios=[1, 1, 2])  # Last column is twice as wide

    # Titles for each subplot
    titles = ['Previous Position Algorithm', 'Particle Filtering Algorithm',
              'Trajectory Fitting Algorithm', 'Exp4-IX Algorithm']

    # Place the first four images in a 2x2 grid on the left
    for i, img_file in enumerate(image_files[:4]):
        ax = fig.add_subplot(gs[i // 2, i % 2])
        image = mpimg.imread(img_file)
        ax.imshow(image)
        ax.set_title(titles[i], fontsize=22)  # Adjust font size for better fit
        ax.axis('off')  # Hide axes for a cleaner look

    # Place the large image on the right, occupying two rows
    ax_final = fig.add_subplot(gs[:, 2])  # This takes both rows in the third column
    final_image = mpimg.imread(image_files[-1])
    ax_final.imshow(final_image)
    ax_final.axis('off')  # Hide axes for a cleaner look

    # Adjust layout to avoid overlapping
    plt.subplots_adjust(wspace=0, hspace=0, left=0, right=1, top=1, bottom=0)

    # Save the combined image with higher resolution
    output_path = f'experimental_pic/{output_file}'
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0, dpi=300)  # Increase dpi for higher quality
    print(f"Combined image saved as {output_path}")