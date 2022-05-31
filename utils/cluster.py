from sklearn.cluster import KMeans
from sklearn.manifold import TSNE


def get_kmeans_cluster(n_cluster, data):
    y_pred = KMeans(n_clusters=n_cluster).fit_predict(data)
    return y_pred


def get_tsne_cluster(n_components, data):
    tsne = TSNE(n_components=n_components, random_state=0)
    tsne_res = tsne.fit_transform(data)
    return tsne_res
