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
    plt.gca().set_aspect('equal')  # Set aspect ratio to make the plot square
    plt.axis([-2, 2, -2, 2])       # Set axis limits
    
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

    # # Number of nodes (N) and edges (M)
    # N, M = B.shape

    # # Step 1: Find the edge set using np.nonzero
    # row_indices, col_indices = np.nonzero(B)  # Get row and column indices where B is non-zero
    
    # # Pair up the indices to create edges
    # edges = []
    # for edge_id in range(M):
    #     # Find where the column index matches the edge_id
    #     rows_for_edge = row_indices[col_indices == edge_id]
    #     if len(rows_for_edge) == 2:  # Each edge should have exactly two nodes connected
    #         edges.append((rows_for_edge[0], rows_for_edge[1]))
    
    # # Step 2: Plot the formation
    # plt.figure()
    # plt.gca().set_aspect('equal')  # Set aspect ratio to make the plot square
    # plt.axis([-2, 2, -2, 2])       # Set axis limits
    
    # # Plot edges (connections between nodes)
    # for edge in edges:
    #     node1, node2 = edge
    #     x_values = [z[node1, 0], z[node2, 0]]  # X-coordinates of the two nodes
    #     y_values = [z[node1, 1], z[node2, 1]]  # Y-coordinates of the two nodes
    #     plt.plot(x_values, y_values, 'k-', linewidth=1.5)  # Black lines for edges
    
    # # Plot nodes (agents)
    # for i in range(N):
    #     plt.plot(z[i, 0], z[i, 1], 'ro', markersize=10)  # Red dots for nodes
    
    # # Finalize plot
    # plt.title("Formation Plot")
    # plt.xlabel("X-axis")
    # plt.ylabel("Y-axis")
    # plt.grid(True)
    # plt.show()
