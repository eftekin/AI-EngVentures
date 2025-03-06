import streamlit as st

# Page Configuration
st.set_page_config(
    page_title="AI-EngVentures Project Hub", page_icon="🚀", layout="wide"
)

# Header
st.markdown(
    '<h1 style="text-align: center; font-size: 3rem; color: #0D47A1; font-weight: bold;">🚀 AI-EngVentures Project Hub</h1>',
    unsafe_allow_html=True,
)
st.markdown(
    '<p style="text-align: center; font-size: 1.2rem; color: #424242;">Innovating with AI and Building the Future, One Project at a Time</p>',
    unsafe_allow_html=True,
)

# Project Categories Overview
col1, col2 = st.columns(2)

with col1:
    st.subheader("🌟 Project Categories")
    st.markdown(
        """
    - 🎯 **Supervised Learning:** Classification and regression projects <a href="https://github.com/eftekin/AI-EngVentures/tree/main/machine_learning/supervised_learning" target="_blank"><img src="https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png" width="20" style="vertical-align: middle;"/></a>
    - 🔍 **Unsupervised Learning:** Clustering and dimensionality reduction <a href="https://github.com/eftekin/AI-EngVentures/tree/main/machine_learning/unsupervised_learning" target="_blank"><img src="https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png" width="20" style="vertical-align: middle;"/></a>
    - 🧠 **Neural Networks:** Deep learning and computer vision <a href="https://github.com/eftekin/AI-EngVentures/tree/main/neural_networks" target="_blank"><img src="https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png" width="20" style="vertical-align: middle;"/></a>
    - ⚙️ **Feature Engineering:** Data preprocessing and feature creation <a href="https://github.com/eftekin/AI-EngVentures/tree/main/machine_learning/feature_engineering" target="_blank"><img src="https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png" width="20" style="vertical-align: middle;"/></a>
    - 🔄 **ML Pipelines:** End-to-end machine learning workflows <a href="https://github.com/eftekin/AI-EngVentures/tree/main/projects/ml_pipelines" target="_blank"><img src="https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png" width="20" style="vertical-align: middle;"/></a>
    """,
        unsafe_allow_html=True,
    )

with col2:
    st.subheader("🛠️ Core Technologies")
    st.markdown("""
    - Python
    - TensorFlow
    - PyTorch
    - Scikit-learn
    - Pandas
    - NumPy
    - Matplotlib
    """)

# Footer
st.markdown("---")
st.markdown(
    '<p style="text-align: center; font-size: 1rem;">🚀 Crafted with passion by AI-EngVentures | Building AI Solutions for Tomorrow</p>',
    unsafe_allow_html=True,
)
