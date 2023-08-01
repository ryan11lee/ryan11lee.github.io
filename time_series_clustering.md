## Exploration of Dynamic Time Warping for Time Series Clustering

### Description

Dynamic Time Warping (DTW) is a method for measuring similarity between two temporal sequences which may vary in speed. It is a powerful tool for time series clustering, and is often used in the fields of data mining, pattern recognition, and machine learning. This project explores the use of DTW for time series clustering, and assess its performance on [Kaggle's](https://www.kaggle.com/datasets/stephengoldie/big-databiopharmaceutical-manufacturing), "Big Data Biopharmaceutical Manufacturing" dataset.

### Data

The data provided by Kaggle is a collection of time series data from a biopharmaceutical manufacturing process. The data consists of 100 batches. The data is provided in a CSV file, with each row representing a single observation, and each column representing a single time series. The data is provided in a "wide" format, with each time series in a separate column. The data is also provided in a "long" format, with each observation in a separate row. The data is provided in both formats to allow for easy use with different software packages.

### Why DTW?

In an ideal world, time series would be the same and we could directly compare two of them similarities, but given the nature of fermentation time series data this is not the case. The fermentation process is a complex process that is affected by many factors, including temperature, pH, and dissolved oxygen. These factors are not constant, and can vary from batch to batch. This means that the time series data for each batch is not directly comparable. DTW is a method for measuring similarity between two temporal sequences which may vary in speed. It is a powerful tool for time series clustering, and is often used in the fields of data mining, pattern recognition, and machine learning.

### How does it work?

While computationally a very expensive algorithm, it works by comparing to sequences and creating a cost matrix, aligning the two series to minimize the overall cost or measure of dissimilarity. With the smallest cost suggesting the two series are most similar. The cost matrix is created by comparing each point in the series to every other point in the other series. The cost matrix is then used to find the optimal path through the matrix, which is the path with the lowest cost. The optimal path is then used to calculate the distance between the two series.

### How to use it?

Utilizing the [fastDTW](https://github.com/rmaestre/FastDTW) package

#### Transform data shape

Taking the original kaggle dataset, and tranforming into a wide format, with each row representing a series and each column representing a time step.

```python

def padded_wide(variable_data, parameter):
    wide = pd.DataFrame()
    for _, temp in variable_data.groupby('Batch reference(Batch_ref:Batch ref)'):
        temp = temp.reset_index(drop=True).reset_index(drop=False)
        temp_wide = temp.pivot(index='Batch reference(Batch_ref:Batch ref)', columns='index', values=parameter)
        wide = pd.concat([wide, temp_wide], axis=0)
    return wide
```

#### Utilize the dtw package to calculate the distance matrix

```python
# Calculate pairwise distances using DTW
pairwise_distances_dtw = np.zeros((len(time_series_data), len(time_series_data)))

for i in range(len(time_series_data)):
    for j in range(i + 1, len(time_series_data)):
        distance, _ = fastdtw(time_series_data[i], time_series_data[j])
        pairwise_distances_dtw[i, j] = distance
        pairwise_distances_dtw[j, i] = distance
```
The output of this code is a square matrix pairwise_distances_dtw, where each entry at position (i, j) represents the DTW distance between the time series data at index i and index j. Since DTW is symmetric, the distances are calculated only for the upper triangular part of the matrix, and the lower triangular part is filled accordingly to maintain symmetry.

#### Applying clustering algorithms

Now that we have a distance matrix, we can apply any clustering algorithm to the data. In this example, we will use K-means clustering. The K-means algorithm requires us to specify the number of clusters to use. We will use the elbow method to determine the optimal number of clusters.

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

Here is the comparison 2 clusters vs 4 clusters.
<img src="images/dtw/2clust.png?raw=true"/>
vs.
<img src="images/dtw/4clust.png?raw=true"/>

As we see, while not statically shown the 4 clusters are more distinct than the 2 clusters. With each population seemingly matching the other trends in the cluster.


### Conclusion

In conclusion, Dynamic Time Warping is a powerful tool for time series clustering, a It is a computationally expensive algorithm, but can be used to cluster time series data that is not directly comparable. In this example, we used DTW to cluster time series data from a biopharmaceutical manufacturing process. We found that the optimal number of clusters was 4, and that it is clear the differenves in each cluster show different behaviors in the fermentation process.





