import numpy as np
import matplotlib.pyplot as plt


def plot_formation(z, goal=None):
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

    z_star = np.array([[ 2,  0],
                        [ 1,  1],
                        [ 1, -1],
                        [ 0,  1],
                        [ 0, -1],
                        [-1,  1],
                        [-1, -1]])
    
    D = z_star.shape[1]
    if len(z.shape) == 3:
        z_end = z[:, :, -1]
        z_star = z_star.reshape(N, D, 1)
        plot_trajectory = True
    else:
        z_end = z
        plot_trajectory = False


    fig, axes = plt.subplot_mosaic('ab', figsize=(8,4), constrained_layout=True)
    ax = axes['a']
    for i in range(M):
        ax.plot(z_end[B[:, i]!=0, 0],z_end[B[:, i]!=0, 1], c='k', linewidth=1, zorder=0)
    for i in range(3,N):
        ax.scatter(z_end[i,0], z_end[i,1], color='tab:green', s=50, zorder=2)
    for i in range(0,3):
        ax.scatter(z_end[i,0], z_end[i,1], color='tab:orange', s=50, zorder=2)
    ax.set_aspect('equal')
    ax.set_xlim([-2, 2])
    ax.set_ylim([-2, 2])
    ax.grid()
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Formation')

    if plot_trajectory:
        ax.plot(z[:, 0, :].T, z[:, 1, :].T, c='limegreen', linewidth=2, zorder=1)

    error = np.sum(np.linalg.norm(z-z_star, axis=1), axis=0)
    ax = axes['b']
    # ax.set_aspect('equal')
    ax.semilogy(error)
    ax.set_xlabel('# timesteps')
    ax.set_ylabel('Error')
    ax.set_title('Error in the formation')
    return fig
