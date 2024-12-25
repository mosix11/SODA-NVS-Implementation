import numpy as np
import matplotlib.pyplot as plt


def plot_multiple_views(views, label):
    # Plot all 24 views in a grid
    num_views = views.shape[0]
    cols = 6  # Set the number of columns
    rows = int(np.ceil(num_views / cols))

    fig, axs = plt.subplots(rows, cols, figsize=(12, rows * 2))
    axs = axs.flatten()

    for i in range(num_views):
        axs[i].imshow(views[i])
        axs[i].axis('off')
        axs[i].set_title(f'View {i+1}')

    # Remove any unused subplots
    for j in range(num_views, len(axs)):
        axs[j].axis('off')

    plt.tight_layout()
    plt.suptitle(f'Label: {label}', fontsize=16, y=1.02)
    plt.show()