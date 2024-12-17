import numpy as np
import matplotlib.pyplot as plt


def plot_formation(z):
    # topology graph
    B = np.array([[1,-1,0,0,0,0,0,0,0,-1,0,1],
                  [-1,0,0,0,0,0,1,-1,0,0,0,0],
                  [0,1,-1,0,0,0,0,0,1,0,0,0],
                  [0,0,0,0,0,1,-1,0,0,1,-1,0],
                  [0,0,1,-1,0,0,0,0,0,0,1,-1],
                  [0,0,0,0,1,-1,0,0,-1,0,0,0],
                  [0,0,0,1,-1,0,0,1,0,0,0,0]])
    
    # dimensions
    N, M = B.shape
    
    # edge set
    edges = np.where(B.flatten()!=0)[0].reshape((2,M)) % 7
    # edges[edges == 0] = N
    # edges -= 1
    
    # target formation
    plt.figure()
    for i in range(M):
        plt.plot(z[edges[:, i], 0], z[edges[:, i], 1], 'k', linewidth=.5)
    for i in range(N):
        plt.plot(z[i, 0], z[i, 1], 'r.', markersize=5)
    plt.axis([-2, 2, -2, 2])
    plt.show()