import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import sklearn as sk
from sklearn import datasets
from sklearn.cluster import KMeans
from scipy.stats import mode
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

digits = datasets.load_digits()


def getPredictedValues(num_clusters):

    digits = datasets.load_digits()
    labels = digits.target
    # list of 1700... actual digit values

    k_means = KMeans(n_clusters=num_clusters, random_state=0)
    cluster_groups = k_means.fit_predict(digits.data)
    # list of 1700... cluster group numbers
    # 0, 7, 2, 5, 7, 2, 1, 8, etc...

    mapping = []
    for i in range(num_clusters):  # 0-9
        indices = []
        # for each cluster we store all the index's
        for j in range(len(cluster_groups)):  # index of cluster group item
            if cluster_groups[j] == i:  # compare value to group number
                indices.append(j)

        digits = []  # 4, 4, 4, 4, 4, 4, 4, 4, 3, 4, 4, 4
        # we then get the actual values from this list of index's
        for index in indices:  # 5, 84, 192, 1045
            digits.append(labels[index])
        # compute the mode - and add to mapping
        average_digit = mode(digits)[0]
        # now the mapping will have the most common value for that cluster
        # ie: any item in the cluster_group that is cluster 0 has a value of 4
        mapping.append(average_digit)

    predicted_labels = []
    # 4, 1, 5, 3, 1, 5, 9, 8, etc...
    # we then go through the list of cluster_groups and add their digit values to the predicted_labels list
    for cluster_group in cluster_groups:
        predicted_labels.append(mapping[cluster_group])

    return predicted_labels


def accuracyOfClusters(n_clusters, print_matrix=False):

    # list of samples -> inside that list, cluster that it has been assigned to

    k_means = KMeans(n_clusters=n_clusters, random_state=0)

    clusters = k_means.fit_predict(digits.data)

    labels = getPredictedValues(n_clusters)

    if print_matrix:

        mat = confusion_matrix(digits.target, labels)
        sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False,
                    xticklabels=digits.target_names,
                    yticklabels=digits.target_names)

        plt.title("Confusion matrix showing image accuracy of " +
                  str(round(accuracy_score(digits.target, labels), 2) * 100) + "%")
        plt.xlabel('true label')
        plt.ylabel('predicted label')
        plt.show()

    # returning the accuracy score as a %
    return round(accuracy_score(digits.target, labels), 2) * 100


def k_means_inertia(n_clusters_min, n_clusters_max):

    k_means_interia_values = []
    k_means_cluster_amount = []

    for cluster in range(n_clusters_min, n_clusters_max):

        k_means = KMeans(n_clusters=cluster, random_state=0)
        clusters = k_means.fit(digits.data)

        k_means_interia_values.append(round(k_means.inertia_, 2) / 1000)
        k_means_cluster_amount.append(cluster)

    plt.title("Inertia of K-means Algorithm depending on Cluster amount")
    plt.ylabel("Inertia of Cluster")
    plt.xlabel("Cluster amount")
    plt.plot(k_means_cluster_amount, k_means_interia_values, color="black", marker="o", linewidth='1', markersize=4)
    plt.show()


def get_label(digit_clusters, clusters):
    # get all digits actual value
    labels = np.array(digits.target)

    cluster_labels = [[] for cluster in range(digit_clusters)]

    predicted_classes = clusters

    for c in range(digit_clusters):
        mode_for_that_cluster = mode(labels[np.where(predicted_classes == c)])
        cluster_labels[c].append(mode_for_that_cluster[0][0])

    for cluster in cluster_labels:
        cluster[0] = round(cluster[0], 0)

    return cluster_labels


def print_cluster_image(n_clusters, rows):

    k_means = KMeans(n_clusters=n_clusters, random_state=0)
    # compute the cluster centers and predict the cluster index for each sample
    clusters = k_means.fit_predict(digits.data)

    # subplot(1, 10) means 1 row 10 columns

    fig, ax = plt.subplots(rows, round(n_clusters / rows), figsize=(8, 3))
    # get labels for each cluster
    k_means_labels = get_label(n_clusters, clusters)
    labels_np = np.array(k_means_labels)
    labels_np = labels_np.astype(int)
    counter = 0
    # reshapes the clusters, into an 8 by 8 pixel image
    centers = k_means.cluster_centers_.reshape(n_clusters, 8, 8)
    for axi, cent in zip(ax.flat, centers):
        d = labels_np[counter][0]
        axi.title.set_text(d)
        counter += 1
        # for each center (image) plot it as a subplot
        axi.set(xticks=[], yticks=[])
        axi.imshow(cent, interpolation='nearest', cmap=plt.cm.binary)

    plt.show()


def show_cluster_accuracy_diagram(range_min, range_max, step):
    x = []
    y = []
    for i in range(range_min, range_max, step):
        x.append(accuracyOfClusters(i))
        y.append(i)

    plt.title("Accuracy of K-means Algorithm depending on Cluster amount")
    plt.ylabel("Accuracy of Cluster")
    plt.xlabel("Cluster amount")
    plt.plot(y, x, color="black", marker="o", linewidth='1', markersize=4)
    plt.show()


def cluster_scatter_diagram(n_clusters):

    k_means = KMeans(n_clusters=n_clusters, random_state=0)
    X, y_true = sk.datasets.make_blobs(n_samples=digits.data.shape[0], centers=10, cluster_std=1, random_state=0)
    k_means.fit(X)
    y_kmeans = k_means.predict(X)
    plt.title("Scatter diagram of each cluster (digit)")
    plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, s=2, cmap='viridis')
    c = k_means.cluster_centers_
    plt.scatter(c[:, 0], c[:, 1], c='black', s=20, alpha=0.7)
    plt.show()


def main(k1, k2):

    print_cluster_image(k1, 2)
    accuracyOfClusters(k1, True)
    show_cluster_accuracy_diagram(k1, k2, 1)
    cluster_scatter_diagram(k1)


main(10, 20)
