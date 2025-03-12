import numpy as np

# Define sample 2x2 dataset
X = np.array([[4, 2], [2, 4], [3, 3], [6, 2]])  # Features
y = np.array([0, 1, 0, 1])  # Class labels

# Compute class means
mean_0 = np.mean(X[y == 0], axis=0)  # Mean of class 0
mean_1 = np.mean(X[y == 1], axis=0)  # Mean of class 1
overall_mean = np.mean(X, axis=0)    # Mean of all points

# Compute scatter matrices
S_W = np.cov(X.T)  # Within-class scatter matrix (approximated)
mean_diff = (mean_0 - mean_1).reshape(2, 1)  # Reshape for multiplication
S_B = mean_diff @ mean_diff.T  # Between-class scatter matrix

# Compute LDA projection direction
eig_vals, eig_vecs = np.linalg.eig(np.linalg.inv(S_W) @ S_B)
lda_direction = eig_vecs[:, np.argmax(eig_vals)]  # Choose the largest eigenvector

print("LDA Projection Direction (Manual Computation):")
print(lda_direction)
