import numpy as np
import matplotlib.pyplot as plt


def plot_formation(z):
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
    
    # Number of nodes (N) and edges (M)
    N, M = B.shape

    # Step 1: Find the edge set using np.nonzero
    row_indices, col_indices = np.nonzero(B)  # Get row and column indices where B is non-zero
    
    # Pair up the indices to create edges
    edges = []
    for edge_id in range(M):
        # Find where the column index matches the edge_id
        rows_for_edge = row_indices[col_indices == edge_id]
        if len(rows_for_edge) == 2:  # Each edge should have exactly two nodes connected
            edges.append((rows_for_edge[0], rows_for_edge[1]))
    
    # Step 2: Plot the formation
    plt.figure()
    #plt.gca().set_aspect('equal')  # Set aspect ratio to make the plot square
    #plt.axis([-2, 2, -2, 2])       # Set axis limits
    
    # Plot edges (connections between nodes)
    for edge in edges:
        node1, node2 = edge
        x_values = [z[node1, 0], z[node2, 0]]  # X-coordinates of the two nodes
        y_values = [z[node1, 1], z[node2, 1]]  # Y-coordinates of the two nodes
        plt.plot(x_values, y_values, 'k-', linewidth=1.5)  # Black lines for edges
    
    # Plot nodes (agents)
    for i in range(N):
        plt.plot(z[i, 0], z[i, 1], 'ro', markersize=10)  # Red dots for nodes
    
    # Finalize plot
    plt.title("Formation Plot")
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.grid(True)
    plt.show()

import numpy as np
import matplotlib.pyplot as plt

def plot_formation_trajectories(z):
    """
    This function plots the formation in graph representation along with drone trajectories for a formation control project.
    
    Parameters:
        z (numpy.ndarray): N x 2 x K+1 matrix where N is the number of agents,
                           2 is the dimensionality of their positions, and K+1 is the number of time steps.
        B (numpy.ndarray): N x M incidence matrix representing the graph topology.
                           Each column represents an edge connecting two nodes.
    """

    B = np.array([
        [1, -1, 0, 0, 0, 0, 0, 0, 0, -1, 0, 1],
        [-1, 0, 0, 0, 0, 0, 1, -1, 0, 0, 0, 0],
        [0, 1, -1, 0, 0, 0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, -1, 0, 0, 1, -1, 0],
        [0, 0, 1, -1, 0, 0, 0, 0, 0, 0, 1, -1],
        [0, 0, 0, 0, 1, -1, 0, 0, -1, 0, 0, 0],
        [0, 0, 0, 1, -1, 0, 0, 1, 0, 0, 0, 0]
    ])
    # Number of nodes (N) and edges (M)
    N, M = B.shape

    # Step 1: Find the edge set using np.nonzero
    row_indices, col_indices = np.nonzero(B)  # Get row and column indices where B is non-zero
    
    # Pair up the indices to create edges
    edges = []
    for edge_id in range(M):
        # Find where the column index matches the edge_id
        rows_for_edge = row_indices[col_indices == edge_id]
        if len(rows_for_edge) == 2:  # Each edge should have exactly two nodes connected
            edges.append((rows_for_edge[0], rows_for_edge[1]))
    
    # Step 2: Plot the trajectories of all agents
    plt.figure(figsize=(8, 8))
    for i in range(N):  # Loop through each agent
        plt.plot(z[i, 0, :], z[i, 1, :], linestyle='--', linewidth=1, label=f"Agent {i+1} path")  # Plot trajectory
    
    # Step 3: Plot the formation (edges and nodes) at the final time step
    z_final = z[:, :, -1]  # Final positions of all agents
    
    # Plot edges (connections between nodes)
    for edge in edges:
        node1, node2 = edge
        x_values = [z_final[node1, 0], z_final[node2, 0]]  # X-coordinates of the two nodes
        y_values = [z_final[node1, 1], z_final[node2, 1]]  # Y-coordinates of the two nodes
        plt.plot(x_values, y_values, 'k-', linewidth=1.5)  # Black lines for edges
    
    # Plot nodes (agents) at the final positions
    for i in range(N):
        plt.plot(z_final[i, 0], z_final[i, 1], 'ro', markersize=10, label="Final Position" if i == 0 else "")
    
    # Finalize plot
    plt.title("Drones Trajectories and Formation Plot")
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.grid(True)
    plt.axis([-2, 2, -2, 2])
    plt.legend()
    plt.show()
