import numpy as np

from utils.sentence_gen import gen_semantic
from utils.bert_utils import get_sentence_embedding
from utils.cluster import get_kmeans_cluster, get_tsne_cluster
import seaborn as sns
import matplotlib.pyplot as plt

# get sentence embeddings
positive, negative, label = gen_semantic("./data/animals.txt")
positive_embedding = get_sentence_embedding(positive)
negative_embedding = get_sentence_embedding(negative)

all_embedding = np.concatenate((positive_embedding, negative_embedding))

# kmeans cluster
k_means_cluster = get_kmeans_cluster(n_cluster=2, data=all_embedding)
# tsne cluster
tsne_cluster = get_tsne_cluster(2, all_embedding)

sns.scatterplot(x=tsne_cluster[:, 0], y=tsne_cluster[:, 1], hue=label, palette=sns.hls_palette(2), legend='full')
plt.show()
