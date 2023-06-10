
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances
from fastdtw import fastdtw

# Assuming you have a numpy array 'time_series_data' with shape (100, sequence_length)
# where each row represents a time series

# Calculate pairwise distances using DTW
pairwise_distances_dtw = np.zeros((len(time_series_data), len(time_series_data)))

for i in range(len(time_series_data)):
    for j in range(i+1, len(time_series_data)):
        distance, _ = fastdtw(time_series_data[i], time_series_data[j])
        pairwise_distances_dtw[i, j] = distance
        pairwise_distances_dtw[j, i] = distance

# Apply K-means clustering
num_clusters = 5  # Adjust the number of clusters as per your requirements
kmeans = KMeans(n_clusters=num_clusters)
cluster_labels = kmeans.fit_predict(pairwise_distances_dtw)

# Print the cluster labels
for i in range(len(cluster_labels)):
    print(f"Time series {i+1} belongs to Cluster {cluster_labels[i]}")
'''

Make sure you have the required dependencies installed, such as `numpy`, `scikit-learn`, and `fastdtw`. You can install them using pip, for example: `pip install numpy scikit-learn fastdtw`.

In this example, the code uses the `fastdtw` function from the `fastdtw` library to calculate the pairwise DTW distances between the time series. Then, it applies the K-means clustering algorithm from scikit-learn to cluster the time series based on these distances.

Remember to adjust the `num_clusters` variable according to the desired number of clusters for your dataset. Also, feel free to modify and extend the code to suit your specific needs, such as adding visualization or additional preprocessing steps.

Note: Depending on the size and complexity of your dataset, the DTW calculation and clustering process may take some time to complete.'''

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances
from fastdtw import fastdtw
import matplotlib.pyplot as plt

# Assuming you have a numpy array 'time_series_data' with shape (100, sequence_length)
# where each row represents a time series

# Calculate pairwise distances using DTW
pairwise_distances_dtw = np.zeros((len(time_series_data), len(time_series_data)))

for i in range(len(time_series_data)):
    for j in range(i + 1, len(time_series_data)):
        distance, _ = fastdtw(time_series_data[i], time_series_data[j])
        pairwise_distances_dtw[i, j] = distance
        pairwise_distances_dtw[j, i] = distance

# Perform K-means clustering for different numbers of clusters
max_clusters = 10  # Adjust the maximum number of clusters to consider
inertia = []

for num_clusters in range(1, max_clusters + 1):
    kmeans = KMeans(n_clusters=num_clusters)
    kmeans.fit(pairwise_distances_dtw)
    inertia.append(kmeans.inertia_)

# Plot the elbow curve
plt.plot(range(1, max_clusters + 1), inertia, marker='o')
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')
plt.title('Elbow Curve')
plt.show()


import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances
from fastdtw import fastdtw
import matplotlib.pyplot as plt

# Assuming you have a numpy array 'time_series_data' with shape (100, sequence_length)
# where each row represents a time series

# Calculate pairwise distances using DTW
pairwise_distances_dtw = np.zeros((len(time_series_data), len(time_series_data)))

for i in range(len(time_series_data)):
    for j in range(i + 1, len(time_series_data)):
        distance, _ = fastdtw(time_series_data[i], time_series_data[j])
        pairwise_distances_dtw[i, j] = distance
        pairwise_distances_dtw[j, i] = distance

# Perform K-means clustering for different numbers of clusters
max_clusters = 10  # Adjust the maximum number of clusters to consider
inertia = []

for num_clusters in range(1, max_clusters + 1):
    kmeans = KMeans(n_clusters=num_clusters)
    kmeans.fit(pairwise_distances_dtw)
    inertia.append(kmeans.inertia_)

# Plot the elbow curve
plt.plot(range(1, max_clusters + 1), inertia, marker='o')
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')
plt.title('Elbow Curve')
plt.show()

# Determine the optimal number of clusters based on the elbow curve
# You can modify this part to use other methods for determining the number of clusters
optimal_num_clusters = 3  # Adjust this based on the elbow curve and your judgment

# Apply K-means clustering with the optimal number of clusters
kmeans = KMeans(n_clusters=optimal_num_clusters)
cluster_labels = kmeans.fit_predict(pairwise_distances_dtw)

# Print the cluster labels
for i in range(len(cluster_labels)):
    print(f"Time series {i+1} belongs to Cluster {cluster_labels[i]}")