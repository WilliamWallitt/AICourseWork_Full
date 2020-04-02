import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets

load_digits = datasets.load_digits()


def get_training_and_validation_sets(digits):
    # get list of [label, matrix]
    images_and_labels = list(zip(digits.target, digits.images))

    training_size = int(len(images_and_labels) * 0.5)

    validation_set = images_and_labels[:training_size]
    training_set = images_and_labels[training_size:]

    return training_set, validation_set


training, validation = get_training_and_validation_sets(load_digits)


def starting_centroids(data, cluster_amount):
    starting = []

    for k in range(cluster_amount):
        to_append = random.choice(data)
        starting.append(to_append[1])

    return starting


def mean_of_cluster(cluster):
    init_sum = cluster[0][1].copy()

    for c in cluster[1:]:
        init_sum += c[1]

    mean_of_points = init_sum * (1.0 / len(cluster))

    return mean_of_points


def move_centroids_centers(clusters):
    new_centers = []

    for c in clusters:
        new_centers.append(mean_of_cluster(c))

    return new_centers


def k_means_algorithm(data, centroids):
    len_centriods = range(len(centroids))

    clusters = [[] for c in len_centriods]

    for (label, matrix) in data:

        smallest_distance = float("inf")

        for index in len_centriods:

            current_centroid = centroids[index]
            distance = np.linalg.norm(matrix - current_centroid)

            if distance < smallest_distance:
                closest_centroid_index = index
                smallest_distance = distance

        clusters[closest_centroid_index].append((label, matrix))

    return clusters


def difference_prev_current_centroids(prev_centroid, current_centroid):
    list_prev = []
    list_current = []
    list_diff = []

    for index in range(len(prev_centroid)):
        list_prev.append(np.linalg.norm(prev_centroid[index]))
        list_current.append(np.linalg.norm(current_centroid[index]))

    for index in range(len(list_prev)):
        list_diff.append(abs(list_current[index] - list_prev[index]))

    return sum(list_diff)


def k_means_iterator(data, clusters, centroids):
    while True:

        old_centroids = centroids
        centroids = move_centroids_centers(clusters)
        clusters = k_means_algorithm(data, centroids)
        difference = difference_prev_current_centroids(old_centroids, centroids)

        if difference == 0.0:
            break

    return clusters, centroids


def k_means_start(data, cluster_amount):
    centroids = starting_centroids(data, cluster_amount)
    clusters = k_means_algorithm(training, centroids)
    end_clusters, end_centroids = k_means_iterator(training, clusters, centroids)

    return end_clusters, end_centroids


def look_up_label(data_set, matrix):
    for item in data_set:
        if item[1] == matrix:
            return item[0]


def most_frequent(list):
    counter = 0
    num = list[0]

    for i in list:
        curr_frequency = list.count(i)
        if curr_frequency > counter:
            counter = curr_frequency
            num = i

    return num


def add_labels_to_centroids(clusters):
    centroid_labels = [[] for x in range(len(clusters))]

    for index, c in enumerate(clusters):

        for matrix in c:
            label = matrix[0]
            centroid_labels[index].append(label)

    for index, label in enumerate(centroid_labels):
        centroid_labels[index] = most_frequent(centroid_labels[index])

    return centroid_labels


# goes through list of centroids and list of labels, and prints each centroids matrix
def print_centroids(centroids, labels):
    # resize the image to an 8 by 8 pixel image
    fig = plt.figure(figsize=(8, 8))
    num_clusters = len(centroids)
    cols = int(num_clusters / 2)
    rows = 2

    for i in range(1, cols * rows + 1):
        x = centroids[i - 1]
        x.reshape(8, 8)
        plt.set_cmap('gray_r')
        fig.add_subplot(rows, cols, i)
        plt.title(str(labels[i - 1]))
        plt.axis('off')

        plt.imshow(x, interpolation='nearest')

    plt.show()


def main(data_set, num_clusters):

    # get final clusters and centroids
    test_clust, test_cent = k_means_start(data_set, num_clusters)
    # add labels to each centroid
    centroid_labels = add_labels_to_centroids(test_clust)
    # print centroids (with the respective labels)
    print_centroids(test_cent, centroid_labels)


main(training, 10)