import random
import numpy as np
import matplotlib.pyplot as plt
import csv
from sklearn.cluster import KMeans
import math
from matplotlib import cm
from sklearn.decomposition import PCA

index_of_wanted_attributes = []
NUM_CLUSTERS = 7


# getting the information we want from the pokemon.csv file
def read_in_file_1(file_path):

    list_of_pokemon = []

    with open(file_path, encoding="utf8") as csv_file:

        csv_reader = csv.reader(csv_file, delimiter=',')

        for index, row in enumerate(csv_reader):

            if index == 0:
                for index, i in enumerate(row):
                    if i == 'attack' or i == 'defense':
                        index_of_wanted_attributes.append(index)
            else:

                list_of_pokemon.append(row)

    return list_of_pokemon


def training_and_validation_set(data):

    training_size = int(len(data) * 0.5)
    validation = data
    training = data

    return training, validation


train, valid = training_and_validation_set(read_in_file_1('pokemon.csv'))


def get_data_attributes(data):

    new_data = []

    for index, d in enumerate(data):

        new_data.append((index, d[index_of_wanted_attributes[0]], d[index_of_wanted_attributes[1]]))

    return new_data


# returns tuple of (index, attack, defense) for each data element as a list
training_set = get_data_attributes(train)

# we only want the attack, defense data
attack_def_tuple_list = []
for i in training_set:
    attack_def_tuple_list.append([int(i[1]), int(i[2])])

# lets perform the k-means clustering on this data

k_means = KMeans(n_clusters=NUM_CLUSTERS, init='random')

predicted = k_means.fit_predict(attack_def_tuple_list)

# we now have a list of which cluster a pokemon belongs to
pred_list = np.array(predicted).tolist()

# lets now create a list of lists each list a cluster -> inside each cluster is all the pokemon


def print_clusters(list_clusters, data, labels):

    colors = cm.rainbow(np.linspace(0, 1, NUM_CLUSTERS + 1))

    for index, item in enumerate(data):

        item_cluster = list_clusters[index]
        plt.scatter(item[0], item[1], s=2, color=colors[item_cluster], cmap='viridis')

    # lets plot the cluster centers and add some labels
    centroids = k_means.cluster_centers_

    # plt.scatter(centroids[:, 0], centroids[:, 1],
    #             marker='o', s=15, linewidths=3,
    #             color='k', zorder=10)

    for index, lab in enumerate(labels):

        plt.text(centroids[index][0], centroids[index][1], "Gen: " + str(lab), color='k')
        plt.scatter(centroids[index][0], centroids[index][1],
                    marker='o', s=15, linewidths=3,
                    color='k', zorder=10)

    plt.title("Pokemon Attack vs Defense Clusters")
    plt.xlabel("Attack")
    plt.ylabel("Defense")

    plt.show()


def getting_clusters_of_pokemon(data, num_clusters):

    list_of_cluster = [[] for x in range(num_clusters)]

    for cluster in range(num_clusters):

        for index, item in enumerate(data):

            if item == cluster:

                list_of_cluster[cluster - 1].append(train[index])

    return list_of_cluster


clusters = getting_clusters_of_pokemon(pred_list, NUM_CLUSTERS)

# lets see if for each pokemon generation gen(1-6) there is an increase in average attack/defence


# lets find the most frequent generation of pokemon
def most_frequent(cluster):
    counter = 0
    num = cluster[0]

    for i in cluster:

        curr_frequency = cluster.count(i)

        if curr_frequency > counter:
            counter = curr_frequency
            num = i

    return num


# this function takes in a list of clusters (each list contains the pokemon that belong to that cluster)
# returns the most frequent generation pokemon in each cluster as a list[int]
def get_pokemon_generation_info(clusters):

    cluster_generation = []

    for c in clusters:

        gen = []

        for item in c:

            gen.append(int(item[39]))

        cluster_generation.append(most_frequent(gen))

    for index, i in enumerate(cluster_generation):

        print("Cluster " + str(index + 1) + " generation is :" + str(round(i, 2)))

    return cluster_generation


def main():

    labels = get_pokemon_generation_info(clusters)
    print_clusters(pred_list, attack_def_tuple_list, labels)


main()
