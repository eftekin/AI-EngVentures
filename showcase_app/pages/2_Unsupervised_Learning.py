import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn import datasets
from sklearn.cluster import DBSCAN, AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

if "digits_model" not in st.session_state:
    st.session_state.digits_model = None
    st.session_state.digits_data = None

st.title("Unsupervised Learning Projects")

st.markdown("""
### About Unsupervised Learning
This demo showcases different clustering algorithms and their applications:
- **Digit Clustering**: Demonstrates pattern recognition in handwritten digits
- **Customer Segmentation**: Shows how to group customers based on their characteristics
- **Interactive Clustering**: Allows experimentation with different algorithms and parameters
""")

st.markdown("---")

tabs = st.tabs(["Digit Clustering", "Customer Segmentation", "Interactive Clustering"])
tab1, tab2, tab3 = tabs

with tab1:
    st.header("Digit Clustering with K-Means")

    code = """
from sklearn import datasets
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt

# Load digits dataset
digits = datasets.load_digits()

# Create KMeans model
model = KMeans(n_clusters=10, random_state=42)
model.fit(digits.data)

# Visualize cluster centers
fig = plt.figure(figsize=(8, 3))
for i in range(10):
    ax = fig.add_subplot(2, 5, 1 + i)
    ax.imshow(model.cluster_centers_[i].reshape((8, 8)), cmap=plt.cm.binary)
plt.show()
    """

    st.code(code, language="python")

    if st.button("Run Digit Clustering", key="run_clustering"):
        try:
            # Load the digits dataset
            digits = datasets.load_digits()
            st.session_state.digits_data = digits

            # Show sample images
            st.subheader("Sample Digit Images")
            fig1 = plt.figure(figsize=(10, 5))
            for i in range(10):
                ax = fig1.add_subplot(2, 5, i + 1, xticks=[], yticks=[])
                ax.imshow(digits.images[i], cmap=plt.cm.binary)
                ax.set_title(f"Digit: {digits.target[i]}")
            st.pyplot(fig1)
            plt.close(fig1)

            # Create and train KMeans model
            model = KMeans(n_clusters=10, random_state=42)
            model.fit(digits.data)
            st.session_state.digits_model = model

            # Show cluster centers
            st.subheader("Cluster Centers")
            fig2 = plt.figure(figsize=(10, 5))
            for i in range(10):
                ax = fig2.add_subplot(2, 5, i + 1, xticks=[], yticks=[])
                ax.imshow(model.cluster_centers_[i].reshape((8, 8)), cmap=plt.cm.binary)
                ax.set_title(f"Cluster {i}")
            st.pyplot(fig2)
            plt.close(fig2)

            st.success("Digit clustering completed successfully!")

        except Exception as e:
            st.error(f"Error during digit clustering: {str(e)}")

    # Only show prediction section if model exists
    if (
        st.session_state.digits_model is not None
        and st.session_state.digits_data is not None
    ):
        st.subheader("Try Predicting New Digits")

        digits = st.session_state.digits_data
        model = st.session_state.digits_model

        col1, col2 = st.columns(2)
        with col1:
            sample_digit = st.selectbox(
                "Select a sample digit from the dataset",
                range(len(digits.images)),
                key="digit_selector",
            )

            # Original digit
            st.write("Original Digit:")
            fig_orig = plt.figure(figsize=(3, 3))
            plt.imshow(digits.images[sample_digit], cmap=plt.cm.binary)
            plt.title(f"Digit {digits.target[sample_digit]}")
            st.pyplot(fig_orig)
            plt.close(fig_orig)

        with col2:
            # Predicted cluster
            prediction = model.predict([digits.data[sample_digit]])[0]
            st.write("Predicted Cluster:")
            fig_pred = plt.figure(figsize=(3, 3))
            plt.imshow(
                model.cluster_centers_[prediction].reshape(8, 8), cmap=plt.cm.binary
            )
            plt.title(f"Cluster {prediction}")
            st.pyplot(fig_pred)
            plt.close(fig_pred)

    st.markdown("---")
    st.markdown(
        '<a href="https://github.com/eftekin/AI-EngVentures/blob/main/projects/unsupervised_learning/handwriting_recognition_kmeans/main.py" target="_blank"><img src="https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png" width="25" style="vertical-align: middle;"/></a> | <a href="https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html">Dataset: Digits Dataset</a>',
        unsafe_allow_html=True,
    )

with tab2:
    st.header("Customer Segmentation")

    # Generate sample customer data
    @st.cache_data
    def generate_customer_data(n_samples=500):
        np.random.seed(42)
        age = np.random.normal(40, 15, n_samples)
        income = np.random.normal(60000, 20000, n_samples)
        spending_score = np.random.normal(50, 25, n_samples)
        return pd.DataFrame(
            {"Age": age, "Income": income, "SpendingScore": spending_score}
        )

    data = generate_customer_data()

    col1, col2 = st.columns(2)
    with col1:
        n_clusters = st.slider("Number of Clusters", 2, 8, 4, key="customer_n_clusters")
    with col2:
        algorithm = st.selectbox(
            "Clustering Algorithm",
            ["KMeans", "DBSCAN", "Hierarchical"],
            key="customer_algorithm",
        )

    if st.button("Perform Customer Segmentation"):
        # Standardize the features
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(data)

        # Perform clustering
        if algorithm == "KMeans":
            model = KMeans(n_clusters=n_clusters, random_state=42)
        elif algorithm == "DBSCAN":
            model = DBSCAN(eps=0.5, min_samples=5)
        else:
            model = AgglomerativeClustering(n_clusters=n_clusters)

        clusters = model.fit_predict(data_scaled)

        # Visualize results using PCA
        pca = PCA(n_components=2)
        data_2d = pca.fit_transform(data_scaled)

        # Create interactive plot
        df_plot = pd.DataFrame(data_2d, columns=["PC1", "PC2"])
        df_plot["Cluster"] = clusters

        fig = px.scatter(
            df_plot,
            x="PC1",
            y="PC2",
            color="Cluster",
            title=f"Customer Segments using {algorithm}",
        )
        st.plotly_chart(fig)

    st.markdown("---")
    st.markdown(
        '<a href="https://github.com/eftekin/AI-EngVentures/blob/main/projects/unsupervised_learning/customer_segmentation/main.py" target="_blank"><img src="https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png" width="25" style="vertical-align: middle;"/></a>',
        unsafe_allow_html=True,
    )

with tab3:
    st.header("Interactive Clustering Playground")

    # Generate sample datasets
    dataset_type = st.selectbox(
        "Select Dataset Type",
        ["Blobs", "Circles", "Moons"],
        key="interactive_dataset_type",
    )

    col1, col2 = st.columns(2)
    with col1:
        n_samples = st.slider(
            "Number of Samples", 100, 1000, 300, key="interactive_n_samples"
        )
    with col2:
        noise = st.slider("Noise Level", 0.0, 1.0, 0.1, key="interactive_noise")

    # Generate selected dataset
    if dataset_type == "Blobs":
        X, _ = datasets.make_blobs(n_samples=n_samples, centers=4, cluster_std=noise)
    elif dataset_type == "Circles":
        X, _ = datasets.make_circles(n_samples=n_samples, noise=noise, factor=0.5)
    else:
        X, _ = datasets.make_moons(n_samples=n_samples, noise=noise)

    # Clustering parameters
    algorithm = st.selectbox(
        "Select Clustering Algorithm",
        ["KMeans", "DBSCAN", "Hierarchical"],
        key="interactive_algorithm",
    )

    if algorithm == "KMeans":
        n_clusters = st.slider(
            "Number of Clusters", 2, 8, 4, key="interactive_kmeans_n_clusters"
        )
        model = KMeans(n_clusters=n_clusters, random_state=42)
    elif algorithm == "DBSCAN":
        eps = st.slider("Epsilon", 0.1, 2.0, 0.5, key="interactive_dbscan_eps")
        min_samples = st.slider(
            "Min Samples", 2, 10, 5, key="interactive_dbscan_min_samples"
        )
        model = DBSCAN(eps=eps, min_samples=min_samples)
    else:
        n_clusters = st.slider(
            "Number of Clusters", 2, 8, 4, key="interactive_hierarchical_n_clusters"
        )
        model = AgglomerativeClustering(n_clusters=n_clusters)

    if st.button("Run Clustering"):
        clusters = model.fit_predict(X)

        # Plot results
        fig = px.scatter(
            x=X[:, 0],
            y=X[:, 1],
            color=clusters,
            title=f"Clustering Results using {algorithm}",
        )
        st.plotly_chart(fig)

        # Show clustering metrics if applicable
        if hasattr(model, "inertia_"):
            st.info(f"Inertia: {model.inertia_:.2f}")

    st.markdown("---")
    st.markdown(
        '<a href="https://github.com/eftekin/AI-EngVentures/blob/main/projects/unsupervised_learning/interactive_clustering/main.py" target="_blank"><img src="https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png" width="25" style="vertical-align: middle;"/></a>',
        unsafe_allow_html=True,
    )
