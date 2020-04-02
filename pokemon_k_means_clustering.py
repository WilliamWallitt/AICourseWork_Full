import random
import numpy as np
import matplotlib.pyplot as plt
import csv
from sklearn.cluster import KMeans

# function reads in the pokemon.csv file and returns a list of (attack, defense) tuples
def read_in_file(file_path):
    speed_def_tuples = []
    attributes = []
    indexOfAttribtes = []

    with open(file_path, encoding="utf8") as csv_file:

        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0

        for index, row in enumerate(csv_reader):

            if index == 0:
                for index, i in enumerate(row):
                    if i == 'attack' or i == 'defense':
                        indexOfAttribtes.append(index)
                    attributes.append(i)
            else:

                speed_def_tuples.append([int(row[indexOfAttribtes[0]]), int(row[indexOfAttribtes[1]])])

    return speed_def_tuples


# abilities,against_bug,against_dark,against_dragon,against_electric,against_fairy,against_fight,against_fire,against_flying,against_ghost,against_grass,against_ground,against_ice,against_normal,against_poison,against_psychic,against_rock,against_steel,against_water,attack,base_egg_steps,base_happiness,base_total,capture_rate,classfication,defense,experience_growth,height_m,hp,japanese_name,name,percentage_male,pokedex_number,sp_attack,sp_defense,speed,type1,type2,weight_kg,generation,is_legendary

# splits the list of (attack, defense) tuples into training and validation set
def training_and_validation_set(speed_def):
    training_size = int(len(speed_def) * 0.5)
    validation = speed_def[:training_size]
    training = speed_def[training_size:]

    return training, validation


def init_random_cluster_position(data, k_amount):
    initial_clusters = []

    for i in range(k_amount):

        center = random.choice(data)

        if center not in initial_clusters:

            initial_clusters.append(center)
        else:
            k_amount += 1

    return initial_clusters


train, valid = training_and_validation_set(read_in_file('pokemon.csv'))


def cluster_scatter_diagram(n_clusters, data):

    if n_clusters > 3:
        print("Max 3 clusters")
        return

    k_means = KMeans(n_clusters=n_clusters, random_state=0)
    predicted = k_means.fit_predict(data)

    pred_list = np.array(predicted).tolist()

    for index, item in enumerate(pred_list):
        data[index].append(item)

    colmap = {0: 'r', 1: 'g', 2: 'b'}

    for i in data:
        plt.scatter(i[0], i[1], color=colmap[i[2]], s=3, cmap='viridis')

    c = k_means.cluster_centers_
    plt.scatter(c[:, 0], c[:, 1], c='black', s=20, alpha=0.7)
    plt.title("Pokemon Attack vs Defense vs Attack Clusters")
    plt.xlabel("Attack")
    plt.ylabel("Defense")
    plt.show()


def main():

    cluster_scatter_diagram(3, train)

