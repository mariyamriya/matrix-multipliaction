import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# Define sample 2x2 dataset (features)
X = np.array([[4, 2], [2, 4], [3, 3], [6, 2]])  # Features
y = np.array([0, 1, 0, 1])  # Class labels

# Apply LDA
lda = LinearDiscriminantAnalysis(n_components=1)
X_lda = lda.fit_transform(X, y)

print("LDA Projection using Scikit-learn:")
print(X_lda)
