import copy
import random
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# k - cluster count
# create some cluster centers
def clasterization(array, k):
    cluster = [[0, 0] for q in range(k)]    # centres of clusters
    cluster_content = [[] for i in range(k)]    # dots in cluster

    # init cluster with random dots
    for i in range(2):
        for q in range(k):
            cluster[q][i] = random.randint(0, max_claster_value)

    # first distribution
    cluster_content = data_distribution(array, cluster, k)

    previous_cluster = copy.deepcopy(cluster)
    #visualisation_2d(cluster_content, cluster)
    while 1:
        cluster = cluster_update(cluster, cluster_content, 2)
        # redistribute
        cluster_content = data_distribution(array, cluster, k)
        if cluster == previous_cluster:
            break
        previous_cluster = copy.deepcopy(cluster)
        #visualisation_2d(cluster_content, cluster)
    #plt.title('Final')
    return cluster_content, cluster
    #visualisation_2d(cluster_content, cluster)

# distribute dots to clusters
def data_distribution(array, cluster, k):
    cluster_content = [[] for i in range(k)]

    for i in range(len(array)):     # for every dot
        min_distance = float('inf')
        situable_cluster = -1
        for j in range(k):  # for every cluster center
            # distance from dot to cluster center
            distance = ((array[i][0]-cluster[j][0])**2+(array[i][1]-cluster[j][1])**2)**(1/2)
            if distance < min_distance:
                min_distance = distance
                situable_cluster = j
        cluster_content[situable_cluster].append(array[i])
    return cluster_content

# update cluster center by average number
def cluster_update(cluster, cluster_content, dim):
    k = len(cluster)    # number of clusters
    for i in range(k):  # by clusters
        for q in range(dim):    # by x and y
            updated_parameter = 0
            for j in range(len(cluster_content[i])):    # for every dot in cluster
                updated_parameter += cluster_content[i][j][q]  # +x or +y
            if len(cluster_content[i]) != 0:    # if not the same center
                updated_parameter = updated_parameter / len(cluster_content[i])
            cluster[i][q] = updated_parameter
    return cluster

def visualisation_2d(cluster_content, cluster):
    k = len(cluster_content)
    plt.grid()
    plt.xlabel("x")
    plt.ylabel("y")

    for i in range(k):
        x_coord = []
        y_coord = []
        for q in range(len(cluster_content[i])):
            x_coord.append(cluster_content[i][q][0])
            y_coord.append(cluster_content[i][q][1])
        plt.scatter(x_coord, y_coord)
    #for j in range(len(cluster)):
    #    plt.scatter(cluster[j][0], cluster[j][1], s=100)

    plt.show()

max_claster_value = 200
random.seed(100)

# load data
data  = pd.read_json("books.json")
# process data
df = data[["Price","Rating"]]
df['Price'] = df['Price'].str.extract(r'(\d+)', expand=False).astype(int)
df['Rating'] = df['Rating'].astype(int)
arr = df.values.tolist()
# k-means
cluster_content = clasterization(arr, 5)
# crop only claster numbers
cluster = pd.DataFrame(cluster_content[0][1])

data = data.assign(cluster = cluster[1])
print(data.keys())
