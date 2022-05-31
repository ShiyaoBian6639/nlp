import numpy as np
from utils.sentence_gen import gen_semantic
from utils.bert_utils import get_sentence_embedding
from utils.cluster import get_kmeans_cluster, get_tsne_cluster, cluster2dict
import seaborn as sns
import matplotlib.pyplot as plt
from utils.dim_red import high_dim_to_plane
import time
import pandas as pd

# get sentence embeddings
positive, negative, label = gen_semantic("./data/animals.txt")
positive_embedding = get_sentence_embedding(positive)
negative_embedding = get_sentence_embedding(negative)

all_sentence = positive + negative
all_embedding = np.concatenate((positive_embedding, negative_embedding))

# kmeans cluster
k = 40
k_means_cluster = get_kmeans_cluster(n_cluster=k, data=all_embedding)
coord = high_dim_to_plane(n_components=2, data=all_embedding)
fig, ax = plt.subplots()
ax.scatter(coord[:, 0], coord[:, 1], c=label)
for i in range(len(all_embedding)):
    if i % 20 == 0:
        ax.annotate(" ".join(all_sentence[i].split(" ")[1:3]), (coord[i, 0], coord[i, 1]))
plt.title("k-means clustering")
plt.show()

# tsne cluster
tsne_cluster = get_tsne_cluster(2, all_embedding)

sns.scatterplot(x=tsne_cluster[:, 0], y=tsne_cluster[:, 1], hue=label, palette=sns.hls_palette(2), legend='full')
plt.show()

# tsne cluster with text
fig, ax = plt.subplots()
ax.scatter(x=tsne_cluster[:, 0], y=tsne_cluster[:, 1], c=label)
for i in range(len(all_embedding)):
    if i % 5 == 0:
        ax.annotate(" ".join(all_sentence[i].split(" ")[1:3]), (tsne_cluster[i, 0], tsne_cluster[i, 1]))
plt.title("tsne clustering")
plt.show()


