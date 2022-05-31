from sklearn.decomposition import PCA


def high_dim_to_plane(n_components, data):
    return PCA(n_components=n_components).fit_transform(data)
