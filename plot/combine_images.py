import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.gridspec import GridSpec

def combine_images(image_files, output_file='combined_image.jpg'):
    # Create a figure with a custom grid layout
    fig = plt.figure(figsize=(12, 10))  # Adjust figure size to a rectangular shape
    gs = GridSpec(2, 2, figure=fig)  # Create a 2x2 grid for the images

    # Titles for each subplot
    titles = ['Previous Position Algorithm', 'Particle Filtering Algorithm',
              'Average Fusion Algorithm', 'Exp4-IX Algorithm']

    # Place the four images in a 2x2 grid
    for i, img_file in enumerate(image_files[:4]):
        ax = fig.add_subplot(gs[i // 2, i % 2])
        image = mpimg.imread(img_file)
        ax.imshow(image)
        ax.set_title(titles[i], fontsize=22, pad=10)  # Adjust padding to move the title down
        ax.axis('off')  # Hide axes for a cleaner look

    # Adjust layout to reduce the space between images
    plt.subplots_adjust(wspace=0.05, hspace=0.1, left=0.02, right=0.98, top=0.95, bottom=0.02)

    # Save the combined image with higher resolution
    output_path = f'experimental_pic/{output_file}'
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0, dpi=300)  # Increase dpi for higher quality
    print(f"Combined image saved as {output_path}")