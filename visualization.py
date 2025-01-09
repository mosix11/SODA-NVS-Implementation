import numpy as np
import matplotlib.pyplot as plt

import imageio.v2 as imageio
from PIL import Image


def plot_multiple_views(views, label):
    # Plot all 24 views in a grid
    num_views = len(views)
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
    
    
def plot_sinusoidal_encoding(encodings):
    # Visualize encodings
    plt.figure(figsize=(6, 10))
    plt.imshow(encodings, cmap='inferno', aspect='auto', extent=[-1, 1, 0, encodings.shape[0]])
    # plt.imshow(encodings_np, cmap='inferno', aspect='auto')
    plt.gca().invert_yaxis()
    plt.colorbar(label='Encoding Value')
    plt.title("Complex Sinusoidal Positional Encodings")
    plt.xlabel("Normalized Position")
    plt.ylabel("Encoding Dimensions")
    plt.show()
    
def plot_noise_schedules(cosine_schedule, inverted_cosine_schedule, T):
    t = np.linspace(0, 1, T+1)
    
    cosine_alphabars = cosine_schedule['alphabar_t']
    inverted_alphabars = inverted_cosine_schedule['alphabar_t']

    # Plot the schedules
    plt.figure(figsize=(8, 6))
    plt.plot(t, cosine_alphabars, label='Cosine Schedule', color='red')
    plt.plot(t, inverted_alphabars, label='Inverse Schedule', color='blue')

    # Highlight noise regions using fill_between
    plt.fill_between(t, 0, 1, where=(t <= 0.4), color='red', alpha=0.1, label='Low Noise')
    plt.fill_between(t, 0, 1, where=((t > 0.4) & (t <= 0.7)), color='cyan', alpha=0.2, label='Medium Noise')
    plt.fill_between(t, 0, 1, where=(t > 0.7), color='blue', alpha=0.1, label='High Noise')

    # Add annotations for noise levels
    plt.text(0.1, 0.9, 'Low Noise', color='red', fontsize=12)
    plt.text(0.5, 0.6, 'Medium Noise', color='blue', fontsize=12)
    plt.text(0.8, 0.2, 'High Noise', color='blue', fontsize=12)

    # Add labels, title, legend, and grid
    plt.xlabel('timestep (t)', fontsize=12)
    plt.ylabel(r'$\bar{\alpha}_t$', fontsize=12)
    plt.title('Noise Levels', fontsize=14)
    plt.legend(loc='lower left')
    plt.grid(alpha=0.3)

    # Display the plot
    plt.tight_layout()
    plt.show()
    
    