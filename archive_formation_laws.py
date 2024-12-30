# def formation(add_noise=True, estimator='mle'):
#     z = np.ones((N, D, K))*z0.reshape(N, D, 1)
#     k = 0
#     while k < K-1:
#         for i in range(3, N):
#             z_i = z[i, :, k]                    # current position
#             z_j = z[:, :, k]                    # all positions
#             dz = (z_i - z_j).reshape((N,D,1))                    # relative positions

#             if add_noise:
#                 v = np.random.multivariate_normal(np.ones(D), R, (N,T)).reshape(N, D, T)     # measurement noise v ~ N(0, R)
#                 y = dz + v                  # linear Gaussian model
#             else:
#                 y = dz

#             if estimator =='None':           
#                 dz_est = y[:, :, 0]         # no estimator: first measurement is used
#             elif estimator == 'mle':
#                 dz_est = np.mean(y, axis=2)  # estimator 1: mean of measurements
#             elif estimator == 'min':
#                 dz_est = np.amin(y, axis=2)  # estimator 2: minimum of measurements

#             u = np.dot(L[i], dz_est)
#             z[i, :, k+1] = z_i + dt * u     # possibility to add process noise w  ~ squared exponential
#         k += 1
#     return z


# def formation2(add_noise=True, estimator='mle'):
#     z = np.kron(np.ones((K,1,1)), z0.T)
#     H = np.kron(np.ones((T,1)), np.identity(D))
#     R_ij = np.kron(np.identity(T), R)
#     k = 0
#     while k < K-1:
#         for i in range(3, N):
#             z_i = z[k, :, i]            # current position
#             z_i_est  = np.zeros((D, N))     # measurements
#             l_i = L[:, i]               # edge weights
#             for j in range(N):
#                 z_j = z[k, :, j]            # all positions                  
#                 z_ij = z_i - z_j            # relative positions

#                 if add_noise:
#                     v = np.random.multivariate_normal(np.zeros(D*T), R_ij)    # measurement noise v ~ N(0, R)
#                 else:
#                     v = np.zeros(D*T)
#                 y_ij = H @ z_ij + v                  # linear Gaussian model

#                 if estimator =='None':           
#                     z_ij_est = y_ij[:D]         # no estimator: first measurement is used
#                 elif estimator == 'mle':
#                     z_ij_est = 1/T * H.T @ y_ij        # estimator 1
#                 elif estimator=='blue':
#                     z_ij_est = inv(H.T @ inv(R_ij) @ H) @ H.T @ inv(R_ij) @ y_ij    # estimator 2
            
#                 z_i_est[:, j] = z_ij_est

#             u_i = np.sum(l_i * z_i_est, axis=1)
#             z[k+1, :, i] = z_i + dt * u_i     # possibility to add process noise w  ~ squared exponential
#         k += 1
#     return z


# def formation3(add_noise=True, estimator='mle'):
#     z = np.kron(np.ones((K,1,1)), z0.T)
#     H = np.kron(np.ones((T,1)), np.identity(D))
#     R_ij = np.kron(np.identity(T), R)
#     k = 0
#     while k < K-1:
#         for i in range(3, N):
#             z_k = z[k, :, :]            # all positions 
#             z_i = z_k[:, i:i+1]   
#             z_N = np.kron(np.ones((1,N)), z_i)      
#             z_ij = z_N - z_k

#             l_i = L[:, i:i+1]

#             if add_noise:
#                 v = np.random.multivariate_normal(np.zeros((D*T)), R_ij, (N)).T
#             else:
#                 v = np.zeros((D*T, N))
#             y = H @ z_ij + v

#             if estimator =='None':           
#                 z_est = y[:D, :]         # no estimator: first measurement is used
#             elif estimator == 'mle':
#                 z_est = 1/T * H.T @ y

#             u = z_est @ l_i
#             z[k+1, :, i:i+1] = z_i + dt * u
#         k += 1
#     return z


# def formation4(add_noise=True, estimator='mle'):
#     z = np.kron(np.ones((K,1,1)), z0.T)
#     R_ij = np.kron(np.identity(T), R)
#     H = np.kron(np.ones((T,1)), np.identity(D))
#     k = 0
#     while k < K-1:
#         for i in range(3, N):
#             z_k = z[k, :, :]            # all positions 
#             z_i = z_k[:, i:i+1]   
#             z_N = np.kron(np.ones((1,N)), z_i)      
#             z_ij = z_N - z_k

#             l_i = L[i:i+1]

#             if add_noise:
#                 v = np.random.multivariate_normal(np.zeros((D)), R, (N, T)).reshape((D*T,N))
#             else:
#                 v = np.zeros((D*T, N))
#             y = H @ z_ij + v

#             if estimator =='None':           
#                 z_est = y[:D, :]         # no estimator: first measurement is used
#             elif estimator == 'mle':
#                 z_est = 1/T * H.T @ y
#             elif estimator == 'blue':
#                 z_est = inv(H.T @ inv(R_ij) @ H) @ H.T @ inv(R_ij) @ y    # estimator 2

            
#             u = z_est @ l_i.T
#             z[k+1, :, i:i+1] = z_i + dt * u
#         k += 1
#     return z


# def plot_formation(z, ax, title='Formation'):
#     """
#     This function plots the formation in graph representation for a formation control project.
    
#     Parameters:
#         z (numpy.ndarray): N-by-D matrix where N is the number of agents and D is the dimensionality of their positions.
#                           Example: z = np.array([[2, 0], [1, 1], [1, -1], [0, 1], [0, -1], [-1, 1], [-1, -1]])
#     """
#     # Define the topology incidence matrix B
#     B = np.array([
#         [1, -1, 0, 0, 0, 0, 0, 0, 0, -1, 0, 1],
#         [-1, 0, 0, 0, 0, 0, 1, -1, 0, 0, 0, 0],
#         [0, 1, -1, 0, 0, 0, 0, 0, 1, 0, 0, 0],
#         [0, 0, 0, 0, 0, 1, -1, 0, 0, 1, -1, 0],
#         [0, 0, 1, -1, 0, 0, 0, 0, 0, 0, 1, -1],
#         [0, 0, 0, 0, 1, -1, 0, 0, -1, 0, 0, 0],
#         [0, 0, 0, 1, -1, 0, 0, 1, 0, 0, 0, 0]
#     ])
#     N, M = B.shape
    
#     D = z.shape[1]
#     if len(z.shape) == 3:
#         z_end = z[:, :, -1]
#         plot_trajectory = True
#     else:
#         z_end = z
#         plot_trajectory = False

#     colors = ['b', 'o', 'g', 'y', 'p', 'lb', 'r']

#     for i in range(M):
#         ax.plot(z_end[B[:, i]!=0, 0],z_end[B[:, i]!=0, 1], c='k', linewidth=.5, zorder=0)
#     for i in range(3,N):
#         ax.scatter(z_end[i,0], z_end[i,1], color=colors[i-2], edgecolor='k', s=50, zorder=2, marker='o')
#     for i in range(0,3):
#         ax.scatter(z_end[i,0], z_end[i,1], color=colors[0], edgecolor='k', s=50, zorder=2, marker='o')

#     if plot_trajectory:    
#         for i in range(3,N):
#             ax.plot(z[i, 0, :], z[i, 1, :], color=colors[i-2], linewidth=1, zorder=1)