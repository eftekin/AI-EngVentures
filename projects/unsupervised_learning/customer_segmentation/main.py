import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def generate_customer_data(n_samples=500):
    np.random.seed(42)
    age = np.random.normal(40, 15, n_samples)
    income = np.random.normal(60000, 20000, n_samples)
    spending_score = np.random.normal(50, 25, n_samples)
    return pd.DataFrame({"Age": age, "Income": income, "SpendingScore": spending_score})


def perform_clustering(data, algorithm="KMeans", n_clusters=4, eps=0.5, min_samples=5):
    # Standardize the features
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)

    # Select and fit clustering model
    if algorithm == "KMeans":
        model = KMeans(n_clusters=n_clusters, random_state=42)
    elif algorithm == "DBSCAN":
        model = DBSCAN(eps=eps, min_samples=min_samples)
    else:
        model = AgglomerativeClustering(n_clusters=n_clusters)

    clusters = model.fit_predict(data_scaled)

    # Perform PCA for visualization
    pca = PCA(n_components=2)
    data_2d = pca.fit_transform(data_scaled)

    return clusters, data_2d, model


if __name__ == "__main__":
    # Example usage
    data = generate_customer_data()
    clusters, data_2d, model = perform_clustering(data, "KMeans", n_clusters=4)
    print(f"Number of clusters: {len(np.unique(clusters))}")
