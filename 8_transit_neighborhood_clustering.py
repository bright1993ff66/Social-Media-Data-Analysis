import pandas as pd
import numpy as np
import os
import read_data

from scipy.cluster.hierarchy import dendrogram, linkage

from sklearn.cluster import KMeans, MeanShift, estimate_bandwidth, DBSCAN, AgglomerativeClustering
from sklearn.manifold import TSNE
from sklearn import preprocessing
from sklearn.decomposition import PCA

from matplotlib import pyplot as plt

# Set the random seed
random_seed = 777


# Clustering using TSNE
# Get coordinates in the two dimensional space
def get_coordinates(array, number_of_components):
    tsne = TSNE(n_components=number_of_components, random_state=random_seed)
    np.set_printoptions(suppress=True)
    Y = tsne.fit_transform(array)
    x_coords = Y[:, 0]
    y_coords = Y[:, 1]
    return x_coords, y_coords


# Use k means to do the clustering
def k_means(array, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters).fit(array)
    kmeans_clustering_labels = list(kmeans.labels_)
    return kmeans_clustering_labels


# Use mean shift to do the clustering
def mean_shift(array, bandwidth, quantile = 0.2, bandwidth_estimation = True):
    if bandwidth_estimation:
        bandwidth_value = estimate_bandwidth(array, quantile=quantile)
        ms = MeanShift(bandwidth=bandwidth_value)
        ms.fit(array)
        labels = list(ms.labels_)
    else:
        ms = MeanShift(bandwidth=bandwidth)
        ms.fit(array)
        labels = list(ms.labels_)
    return labels


# Use DBSCAN to do the clustering
def dbscan(array, eps=2.0, min_samples=4):
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(array)
    labels = list(clustering.labels_)
    return labels


def draw_dendrogram(array, labels, saved_file_name):
    """
    :param array: the array which recores the sentiment and activity level of each TN
    :param labels: the transit neighborhood names
    :param saved_file_name: the name of the saved figure
    :return: a dendrogram which shows the clustering result
    """
    linked = linkage(array, method='ward')
    figure = plt.figure(figsize=(10, 7))
    dendrogram(linked, orientation='top', labels=labels, distance_sort='descending', show_leaf_counts=True)
    figure.savefig(os.path.join(read_data.plot_path, saved_file_name))
    plt.show()


if __name__ == '__main__':
    # Classification based on the monthly sentiment
    file_clustering_monthly_sentiment = pd.read_csv(os.path.join(read_data.desktop, 'by_month_file.csv'), index_col=0)
    file_clustering_monthly_activity = pd.read_csv(os.path.join(read_data.desktop, 'by_month_activity.csv'),
                                                   index_col=0)
    sentiment_activty_concat_dataframe = pd.concat(
        [file_clustering_monthly_sentiment, file_clustering_monthly_activity],
        axis=1)
    station_names = list(sentiment_activty_concat_dataframe.index.values)
    array_for_clustering_sentiment_activity = np.array(sentiment_activty_concat_dataframe)
    array_for_clustering_sentiment = np.array(file_clustering_monthly_sentiment)
    array_for_clustering_activity = np.array(file_clustering_monthly_activity)

    # Standardize the data
    standardized_activity = preprocessing.scale(array_for_clustering_activity)
    standardized_sentiment = preprocessing.scale(array_for_clustering_sentiment)
    standardized_activity_sentiment = preprocessing.scale(array_for_clustering_sentiment_activity)
	
	# Draw the dengdrogram to get the number of clusters
	draw_dendrogram(standardized_activity_sentiment, labels=station_names,
                    saved_file_name='hierarchical_clustering.png')

	
	# Use the hierarchical clustering to do the cluster analysis
    clustering_agglomerative = AgglomerativeClustering(n_clusters=3,
                                                       affinity='euclidean',
                                                       linkage='ward').fit(standardized_activity_sentiment)
    print(clustering_agglomerative.labels_)
    
