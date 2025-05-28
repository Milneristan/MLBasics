import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage

# Given points
points = np.array([
    [1, 1],    # P1
    [1.5, 1.5],# P2
    [5, 5],    # P3
    [3, 4],    # P4
    [4, 4],    # P5
    [3, 3.5]   # P6
])

# Labels for plotting
labels = ['P1', 'P2', 'P3', 'P4', 'P5', 'P6']

# Create dendrogram with Complete Linkage
linked_complete = linkage(points, method='complete')

plt.figure(figsize=(10, 6))
dendrogram(linked_complete, labels=labels)
plt.title('Dendrogram - Complete Linkage')
plt.xlabel('Points')
plt.ylabel('Euclidean Distance')
plt.grid(True)
plt.show()
