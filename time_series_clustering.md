## Exploration of Dynamic Time Warping for Time Series Clustering

### Description

Dynamic Time Warping (DTW) is an effective method for measuring the similarity between temporal sequences that may vary in speed. It is widely used in data mining, pattern recognition, and machine learning, particularly for time series clustering tasks. In this blog post, we delve into the application of DTW for time series clustering and evaluate its performance using Kaggle's "Big Data Biopharmaceutical Manufacturing" dataset.

### Understanding the Challenge

In real-world scenarios, time series data can be challenging to compare directly, especially in the case of fermentation processes where factors like temperature, pH, and dissolved oxygen can vary from batch to batch. Due to these variations, direct comparisons between time series become impractical. DTW comes to the rescue by offering a solution for measuring similarity between temporal sequences with differing speeds, making it an ideal candidate for time series clustering.


### Dataset

The data provided by Kaggle is a collection of time series data from a biopharmaceutical manufacturing process. The data consists of 100 batches. The data is provided in a CSV file, with each row representing a single observation, and each column representing a single time series. The data is provided in a "wide" format, with each time series in a separate column. The data is also provided in a "long" format, with each observation in a separate row. The data is provided in both formats to allow for easy use with different software packages.

### How DTW Works
DTW tackles the challenge of measuring similarity between two time series through a computationally intensive yet powerful algorithm. It constructs a cost matrix by comparing each point in one series with every point in the other series. The matrix helps find the optimal path with the lowest cost, effectively aligning the two sequences to minimize their overall dissimilarity. The resulting optimal path provides the basis for calculating the distance between the two time series.


### Utilizing DTW with Python
To demonstrate the application of DTW for time series clustering, we utilize the [fastDTW](https://github.com/rmaestre/FastDTW) package for efficient computation.

1. **Transform data shape**

The first step involves transforming the Kaggle dataset into a wide format, where each row represents a series and each column represents a time step. This is achieved using the following Python function:



```python

def padded_wide(variable_data, parameter):
    wide = pd.DataFrame()
    for _, temp in variable_data.groupby('Batch reference(Batch_ref:Batch ref)'):
        temp = temp.reset_index(drop=True).reset_index(drop=False)
        temp_wide = temp.pivot(index='Batch reference(Batch_ref:Batch ref)', columns='index', values=parameter)
        wide = pd.concat([wide, temp_wide], axis=0)
    return wide
```

2. **Calculating the distance matrix**

Using the fastdtw function, we calculate pairwise distances between all time series in the dataset. The distances are then stored in a square matrix called pairwise_distances_dtw, where each entry (i, j) represents the DTW distance between time series i and time series j.


```python
# Calculate pairwise distances using DTW
pairwise_distances_dtw = np.zeros((len(time_series_data), len(time_series_data)))

for i in range(len(time_series_data)):
    for j in range(i + 1, len(time_series_data)):
        distance, _ = fastdtw(time_series_data[i], time_series_data[j])
        pairwise_distances_dtw[i, j] = distance
        pairwise_distances_dtw[j, i] = distance
```

3. **Applying K-means Clustering:**

With the distance matrix ready, we can apply any clustering algorithm, such as K-means, to group similar time series together. To determine the optimal number of clusters, we use the elbow method, which helps identify the "elbow point" on the plot of inertia (within-cluster sum of squares) against the number of clusters. In this case, we select the number of clusters with the highest silhouette score as our ideal choice.


```python
# Perform K-means clustering for different numbers of clusters
max_clusters = 10  # Adjust the maximum number of clusters to consider
inertia = []

for num_clusters in range(1, max_clusters + 1):
    kmeans = KMeans(n_clusters=num_clusters)
    kmeans.fit(pairwise_distances_dtw)
    inertia.append(kmeans.inertia_)

```

<img src="images/dtw/scree_fixed.png?raw=true"/>


The elbow method suggests that the optimal number of clusters is either 2 or 3. And the silhouette score suggests that 2 is not a good choice. So we will use 3 clusters.

```python
from sklearn.metrics import silhouette_score


# determine number of clusters

for n_clusters in range(2, 6):
    clusterer = KMeans(n_clusters=n_clusters, random_state=10)
    cluster_labels = clusterer.fit_predict(pairwise_distances_dtw)
    silhouette_avg = silhouette_score(pairwise_distances_dtw, cluster_labels)
    print("For n_clusters =", n_clusters,
          "The average silhouette_score is :", silhouette_avg)
    if silhouette_avg > ideal:
        ideal = silhouette_avg
        ideal_n = n_clusters
        
print("Ideal number of clusters is", ideal_n)

>>>For n_clusters = 2 The average silhouette_score is : 0.9383979222055399
>>>For n_clusters = 3 The average silhouette_score is : 0.49839317175017933
>>>For n_clusters = 4 The average silhouette_score is : 0.5580405338208304
>>>For n_clusters = 5 The average silhouette_score is : 0.5284261919827518
>>>For n_clusters = 6 The average silhouette_score is : 0.5305677072305146
>>>Ideal number of clusters is 2
```
When actually separating the data into 4 clusters, we can see that the data is not well separated into 2 clusters, but rather 4.

| 2 Clusters                    | 4 Clusters                    |
| :-------------------------:  | :-------------------------:  |
<img src="images/dtw/2clust.png?raw=true"/> | <img src="images/dtw/4clust.png?raw=true"/> |



As we see, while not statically shown the 4 clusters are more distinctly seperated into simlar behavior runs than the 2 clusters.

### Conclusion

Dynamic Time Warping is a valuable tool for time series clustering, particularly when dealing with data that is not directly comparable due to varying speeds. Despite its computational intensity, DTW can be applied effectively to cluster time series data. In this project, we used DTW to cluster time series from a biopharmaceutical manufacturing process. The results indicated that the optimal number of clusters was 4, and the clustering successfully revealed distinct behaviors within the fermentation process.




