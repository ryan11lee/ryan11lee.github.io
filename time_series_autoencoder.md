## Utilizing Autoencoders for Anomaly Detection in Time Series Data

### Project description:

Using data from [Kaggle's](https://www.kaggle.com/datasets/stephengoldie/big-databiopharmaceutical-manufacturing) "Big Data Biopharmaceutical Manufacturing" dataset, I developed an anomaly detection model designed explicitly for fermentation processes. Fermentation data is characterized by being a time series, exhibiting specific trends that may include occasional spikes considered normal behavior during a successful run. However, many automated tools often flag these spikes as errors, leading to false alarms.

The challenges in anomaly detection within fermentation manufacturing are multifaceted. Firstly, fermentation run data is a collection of a time series with inherent trends and occasional spikes considered normal during successful runs. However, automated tools may misinterpret these spikes as errors, leading to unnecessary alarms and disruptions in the production workflow. Hence, distinguishing genuine anomalies from expected variations becomes a critical task.

Secondly, the control logic during fermentation is often difficult to discern, making it challenging to differentiate between a run that deviates from the expected trend and one that is being correctly controlled. Moreover, the length of time points in fermentation runs may vary, introducing complexities in data preprocessing and model training.

The significance of accurate anomaly detection in fermentation manufacturing cannot be overstated, as false positives could cause data to be ignored when it was valuable and expensive to produce. By leveraging autoencoders, we can distinguish between normal variations and genuine anomalies, reducing false alarms and ensuring prompt actions when deviations warrant intervention. Modeling that detects anomalies enhances the manufacturing process's overall efficiency and contributes to maintaining high product quality and safety.

To address these challenges and unlock the full potential of anomaly detection in biopharmaceutical manufacturing, we will explore the application of autoencoders. Autoencoders, a type of neural network, offer a powerful unsupervised learning approach to learn efficient data codings and reconstruct the original trend from the data. By comparing the reconstructed time series with the actual time series, we can identify and quantify discrepancies, providing us with a basis for detecting anomalies.

This anomaly detection model offers an effective solution for identifying anomalies in fermentation processes, enabling better decision-making and reducing false alarms caused by normal variations in the data.


### Exploratory Data Analysis:

Control logic during fermentation is challenging to discern when a run is deviating vs. being controlled correctly, so build a model that can be fit to understand the expected trends vs. deviations. 

After downloading the data, we begin some exploratory data analysis. 

One common thing with fermentation data is a similar but variable length of time for each run, or the length of time points is variable. We will not want to train with any outlier time series data, and first, we will look at the shape of each run.

<img src="images/ae_ferm/hist_rows_by_batch.png?raw=true"/>

As we see some variability in the length of runs, we will ensure that anything that is not within two standard deviations of the mean will be removed from the potential pool of training data, as this method will rely on the training runs being labeled as "good" runs.

Additionally, we will look at the data distribution for each column. The min/max values and ensure visually that the data is not skewed or seems problematic.

Utilizing a quick ipywidget, we can identify trends with each label and ensure each group appears similar.

```python
import ipywidgets as widgets
 
variable_plot_selection = widgets.Dropdown(options=variable_list, value = 'Penicillin concentration(P:g/L)')
variable_plot_selection
```
For this write-up, we will only show Penicillin, as this is the target molecule and will be the most evidence that a deviation was present in the data. 

![Penicillin Plotted by Reference Category](images/ae_ferm/image.png)

As is evident in the plots above, the recipe and operator have run with lower performance that does not follow the average trend. Given a large number of runs and the need to separate runs into "good," we will use an unsupervised learning approach to see what clusters are present in the time series.

### Labeling Runs:
Refer to this post to dive deeper into clustering time series trends [link](/time_series_clustering). Given the listed conditions in the dataset, we will select a k of 4; the scree plot suggests 2 clusters are sufficient, but it only separates one gross outlier. As we see in the image, there is good separation even though clusters 0, 1, 2 are very close.

<img src="images/ae_ferm/Clusters.png?raw=true"/>

After dynamic time-warping K-means clustering, we see that 4 clusters are ideal for the data. Cluster 2 represents the most ideal trend, and the remaining clusters represent the other trends, with reduced performance of penicillin production.

<img src="images/ae_ferm/clusters_penicillin.png?raw=true"/>

### Building an AutoEncoder:

Autoencoders are a type of neural network designed for unsupervised learning, aiming to learn efficient data codings by compressing the input data into a lower-dimensional representation. In the context of anomaly detection in time series data, autoencoders play a pivotal role in capturing the normal patterns and variations present in the data. The encoding process maps the input data into a compressed representation, while the decoding process reconstructs the original input from this representation. During training, the autoencoder learns to minimize the reconstruction error, enabling it to identify anomalies that deviate significantly from the learned patterns. By leveraging autoencoders, we can distinguish between normal trends and genuine anomalies in fermentation processes, offering a robust solution for accurate anomaly detection in the fermentation manufacturing industry.


Now that we have labeled data, we will consider all runs in cluster 2 to be performant and what we want to train the model on. The reason is that while there is valuable data in the other clustered runs, we want to train the model on the ideal runs, as including anomalous runs in the encoder will allow the model to recreate incorrect trends. If an end client identified other runs to be of interest or considered normal, we would add those into the model, but for this write-up, we will only consider the ideal runs, or "Golden" runs.

An example of an autoencoder 

<img src="images/ae_ferm/Autoencoders-graph.png?raw=true"/>

[Source](https://www.compthree.com/blog/autoencoder/)




#### Data Prep for Model:

First, we will build a series of functions to enable easy data prep for the model.

These involve getting the maximum size of the dataset; this way, we can ensure equal lengths for the model to output.
Once the size is determined, we will pad the data with 0's to ensure that the model can learn the trends. The choice of 0's is because if a run is terminated, the probes etc, would no longer be reading data and would be 0.
The demonstration model will be a single variable model **Dissolved oxygen concentration(DO2:mg/L)**. The model will be able to learn the trends of a single variable, and we can see how the model performs. The model can be expanded to include all variables in the future.

 <img src="images/ae_ferm/AE-Control-Run.png?raw=true"/>

Following this initial data prep, we must prepare the data for the model. We will need to normalize the data. We will also need to ensure that the data is in a format the model can understand. We will need to reshape the data to be in a 3D format, where the first dimension is the number of samples, the second is the number of time steps, and the third is the number of features. 

With the transformed data, we will now fit the model. The encoder and decoder will be the same shapes, and the model will be sequential. The model will be trained on the "Golden" runs, and then we will test the model on the other runs to see how the model performs.

To determine an "outlier" run, we will plot the model's error on the training data. After we observe the distribution of the loss function, we can determine a threshold value for the loss function. The model will output a reconstructed error value that we will use to determine if a run is an outlier and account for the noise in the training data.

 <img src="images/ae_ferm/loss_dist.png?raw=true"/>
As we see above, the boxplot we consider runs above the red line 3 standard deviations above the mean to be outliers. This is a conservative approach, and we could adjust this to be more or less conservative depending on the needs of the client.

### Results:


After training the model, we will look at examples of the model's performance and demonstrate the value of visualizing the error. 

<img src="images/ae_ferm/good_mae_01.png?raw=true"/>

As we see and have calculated, the error is 0.02, which is within 3 standard deviations of the mean. Which is classified as a good run, and the model can predict the trend well.

Next, we will observe a run that is not ideal and see how the model performs.

<img src="images/ae_ferm/anomalous_mae.png?raw=true"/>

With this run, there is an MAE score of 1, and thus is anomalous and can be considered an outlier. A subject matter expert would be able to also identify as an outlier, so now the end user can decide to investigate the run, with the visualization of the error to help them decide if the anomaly is worth investigating or a run to be ignored.

Once we have these scores, we can apply a variety of techniques. If the runs are completed, we can automatically auto-classify runs that should be flagged or ignored in our data processing pipeline. Additionally, with some additional work and infrastructure, we can build to model to work in real-time and flag anomalous runs in real-time. This is a valuable tool for the end user to identify and investigate potentially anomalous runs in real-time. Saving thousands of dollars per run.

### Conclusion:

Autoencoders work well with complex time series and provide interpretable and useful outputs for scientists to explore potential reasons a fermentation run was anomalous.

Additionally, the model itself was relatively fast to train, thus providing the Data Scientist with easy-to-optimize models, and therefore, deployment and future API calls against the model can be fast and efficient.

The downsides of this model so far appear to be the specificity of the model to "golden" runs. A client would need to provide a set of labeled runs that a model can be trained on. Additionally, one of the best use cases of a model like this is in a manufacturing or scale-up space where the process is consistent. In a process development environment, there are applications where scientists can observe the differences between process changes and their effects on online trends, but given the limited tank capacity, it could be a potential issue to build a model for every process change.

However, the model can provide a valuable tool for scientists to explore the data and provide a helpful tool for the end user to analyze the data and decide how to proceed with the data. Additionally, if the model is deployed in a production environment, the model can be used to flag anomalous runs and provide a useful tool for the end user to explore the data and make decisions on how to proceed with the data.

As the fermentation industry continues to rely on time series data for critical processes, the utilization of autoencoders holds immense promise for enhancing productivity and product quality. By providing interpretable and actionable insights, autoencoders pave the way for a more efficient and reliable biopharmaceutical manufacturing landscape, driving innovation and advancements in this vital domain.