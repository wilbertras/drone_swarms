import numpy as np
import matplotlib.pyplot as plt
plt.style.use('matplotlibrc')
import matplotlibcolors


def plot_formation(z, ax, title='Formation'):
    """
    This function plots the formation in graph representation for a formation control project.
    
    Parameters:
        z (numpy.ndarray): N-by-D matrix where N is the number of agents and D is the dimensionality of their positions.
                          Example: z = np.array([[2, 0], [1, 1], [1, -1], [0, 1], [0, -1], [-1, 1], [-1, -1]])
    """
    # Define the topology incidence matrix B
    B = np.array([
        [1, -1, 0, 0, 0, 0, 0, 0, 0, -1, 0, 1],
        [-1, 0, 0, 0, 0, 0, 1, -1, 0, 0, 0, 0],
        [0, 1, -1, 0, 0, 0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, -1, 0, 0, 1, -1, 0],
        [0, 0, 1, -1, 0, 0, 0, 0, 0, 0, 1, -1],
        [0, 0, 0, 0, 1, -1, 0, 0, -1, 0, 0, 0],
        [0, 0, 0, 1, -1, 0, 0, 1, 0, 0, 0, 0]
    ])
    N, M = B.shape
    
    D = z.shape[1]
    if len(z.shape) == 3:
        z_end = z[:, :, -1]
        plot_trajectory = True
    else:
        z_end = z
        plot_trajectory = False

    colors = ['b', 'o', 'g', 'y', 'p', 'lb', 'r']

    for i in range(M):
        ax.plot(z_end[B[:, i]!=0, 0],z_end[B[:, i]!=0, 1], c='k', linewidth=.5, zorder=0)
    for i in range(3,N):
        ax.scatter(z_end[i,0], z_end[i,1], color=colors[i-2], edgecolor='k', s=50, zorder=2, marker='o')
    for i in range(0,3):
        ax.scatter(z_end[i,0], z_end[i,1], color=colors[0], edgecolor='k', s=50, zorder=2, marker='o')

    if plot_trajectory:    
        for i in range(3,N):
            ax.plot(z[i, 0, :], z[i, 1, :], color=colors[i-2], linewidth=1, zorder=1)


def plot_error(z_star, *z, labels):
    fig, axes = plt.subplot_mosaic('a', figsize=(4,4), constrained_layout=True)
    ax = axes['a']
    for i, x in enumerate(z):
        error = np.sum(np.linalg.norm(z-z_star, axis=1), axis=0)
        ax.semilogy(error, labels[i])
    # ax.set_aspect('equal')
    ax.set_xlabel('# timesteps')
    ax.set_ylabel('Error')
    ax.set_title('Error in the formation')
