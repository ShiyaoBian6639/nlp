import numpy as np
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE


def get_kmeans_cluster(n_cluster: int, data: np.ndarray) -> list:
    y_pred = KMeans(n_clusters=n_cluster).fit_predict(data)
    return y_pred


def get_tsne_cluster(n_components: int, data: np.ndarray) -> np.ndarray:
    tsne = TSNE(n_components=n_components, random_state=0)
    tsne_res = tsne.fit_transform(data)
    return tsne_res


def cluster2dict(k_means_cluster: list) -> dict:
    cluster_dict = dict()
    for i in range(len(k_means_cluster)):
        cluster_val = k_means_cluster[i]
        if cluster_val not in cluster_dict:
            cluster_dict[cluster_val] = [i]
        else:
            cluster_dict[cluster_val].append(i)
    return cluster_dict
