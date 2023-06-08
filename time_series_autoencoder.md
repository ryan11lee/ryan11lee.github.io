## Utilizing Autoencoders for Anomaly Detection in Time Series Data

**Project description:** With data from 

Using data from [Kaggle](https://www.kaggle.com/datasets/stephengoldie/big-databiopharmaceutical-manufacturing)'s, "Big Data Biopharmaceutical Manufacturing" dataset, I developed an anomaly detection model specifically designed for fermentation processes. Fermentation data is characterized by being a time series, exhibiting specific trends that may include occasional spikes, which are considered normal behavior during a successful run. However, many automated tools often flag these spikes as errors, leading to false alarms.

To address this issue, I adopted an approach that leverages the availability of training data for "good" runs. I constructed an autoencoder, a type of neural network, where the objective was to reconstruct the original trend from the data. By comparing the reconstructed trend to the actual trend, any discrepancies or differences were identified and quantified as errors. By accumulating enough training runs, it became possible to establish a mean or average behavior for the population. Consequently, we could determine which runs deviated significantly from the expected norm, thus identifying statistically significant outliers.

This anomaly detection model offers an effective solution for identifying anomalies in fermentation processes, enabling better decision-making and reducing false alarms caused by normal variations in the data.

 <!-- I created an anomaly detection model that allows for fermentation process. Because fermentation data is a time series, with specific trends having "spikes" as part of a good run, a lot automated tooling will detect errors, when they are actually normal behaviors. The apporach i took was that since there is training data for "good" runs, we will build an autoencoder, whrre the reconstruction will be compared to the original trend. Any difference between actual and predicted will be summarized as error. Thus with enough runs, we can determine a mean for the population and see which runs are statistically an outlier. -->

### 1. Exploratory Data Analysis

Control logic during fermentation is difficult to discern when a run is deviating vs being controlled correctly so build a model that can be fit to understand the normal trends vs deviations. 

After downloading the data we begin some exploratory data analysis 

One thing, that is common with fermentation data is there is similar but variable length of time for each run or the the length of time points is variable. We will not want to train with any outlier time series data and first we will look at the shape of each run.

<img src="images/hist_rows_by_batch.png?raw=true"/>

As we see there is some variability in the length of runs, we will ensure that anything that is not within 2 standard deviations of the mean will be removed from the potential pool of training data as this method will rely on the training runs being labeled as "good" runs.



<!-- 

```javascript
if (isAwesome){
  return true
}
```

### 2. Assess assumptions on which statistical inference will be based

```javascript
if (isAwesome){
  return true
}
```

### 3. Support the selection of appropriate statistical tools and techniques

<img src="images/dummy_thumbnail.jpg?raw=true"/>

### 4. Provide a basis for further data collection through surveys or experiments

Sed ut perspiciatis unde omnis iste natus error sit voluptatem accusantium doloremque laudantium, totam rem aperiam, eaque ipsa quae ab illo inventore veritatis et quasi architecto beatae vitae dicta sunt explicabo. 

For more details see [GitHub Flavored Markdown](https://guides.github.com/features/mastering-markdown/). -->
