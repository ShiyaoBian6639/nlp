import numpy as np
from utils.bert_utils import get_sentence_embedding
from utils.cluster import get_kmeans_cluster, get_tsne_cluster, cluster2dict
import matplotlib.pyplot as plt
import matplotlib
from utils.dim_red import high_dim_to_plane
import pandas as pd

matplotlib.rcParams['interactive'] = True
test_data = pd.read_csv("./data/Single_sentences_filtered.csv")
test_data.replace(r'[<%]', '', regex=True, inplace=True)
data_lines = test_data["0"].to_list()
batch = 100
# for i in range(len(test_data) // batch + 1):
#     emb = get_sentence_embedding(data_lines[100 * i: 100 * i + 100])
#     np.savetxt(f"./data/emb{i}.csv", emb)
# combine all files
arr_list = []
for i in range(len(test_data) // batch + 1):
    emb = np.loadtxt(f"./data/emb{i}.csv")
    arr_list.append(emb)
all_embedding = np.concatenate(arr_list, axis=0)

# perform clustering

k = 10
k_means_cluster = get_kmeans_cluster(n_cluster=k, data=all_embedding)
coord = high_dim_to_plane(n_components=2, data=all_embedding)
fig, ax = plt.subplots()
ax.scatter(coord[:, 0], coord[:, 1], c=k_means_cluster)

# tsne cluster
# tsne cluster
# tsne_cluster = get_tsne_cluster(2, all_embedding)
#
# sc = plt.scatter(tsne_cluster[:, 0], tsne_cluster[:, 1])


def onclick(event):
    print('%s click: button=%d, x=%d, y=%d, xdata=%f, ydata=%f' %
          ('double' if event.dblclick else 'single', event.button,
           event.x, event.y, event.xdata, event.ydata))


cid = fig.canvas.mpl_connect('button_press_event', onclick)
plt.show()
