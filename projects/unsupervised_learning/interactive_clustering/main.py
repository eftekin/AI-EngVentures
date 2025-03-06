import numpy as np
from sklearn import datasets
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering

def generate_dataset(dataset_type, n_samples=300, noise=0.1):
    if dataset_type == "Blobs":
        X, _ = datasets.make_blobs(n_samples=n_samples, centers=4, cluster_std=noise)
    elif dataset_type == "Circles":
        X, _ = datasets.make_circles(n_samples=n_samples, noise=noise, factor=0.5)
    else:  # Moons
        X, _ = datasets.make_moons(n_samples=n_samples, noise=noise)
    return X

def perform_clustering(X, algorithm='KMeans', n_clusters=4, eps=0.5, min_samples=5):
    if algorithm == 'KMeans':
        model = KMeans(n_clusters=n_clusters, random_state=42)
    elif algorithm == 'DBSCAN':
        model = DBSCAN(eps=eps, min_samples=min_samples)
    else:
        model = AgglomerativeClustering(n_clusters=n_clusters)
    
    clusters = model.fit_predict(X)
    return clusters, model

if __name__ == '__main__':
    # Example usage
    X = generate_dataset('Blobs', n_samples=300, noise=0.1)
    clusters, model = perform_clustering(X, 'KMeans', n_clusters=4)
    print(f"Number of clusters: {len(np.unique(clusters))}")
