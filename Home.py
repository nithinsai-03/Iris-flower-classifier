import streamlit as st

st.set_page_config(
    page_title="IRIS CLASSIFICATION WEB APP",
    page_icon="🌸",
    layout="wide"
)

st.title("🌸 IRIS Classification Project")
st.subheader("Welcome to the Interactive Data Analysis and Classification App!")

st.markdown("""
This web application is built using **Streamlit** to explore the famous Iris dataset
and provide a machine learning model for flower classification.

Use the sidebar on the left to navigate:
- **Basic EDA:** Explore the dataset through visual graphs and statistics.
- **Classification:** Predict the species of an Iris flower based on its measurements.
""")

st.header("1. The Iris Dataset Overview")
    
st.subheader("Dataset Details")
st.markdown("""
    The Iris dataset is a classic multivariate data set introduced by R.A. Fisher in 1936.
    It consists of 150 samples from three species of Iris flowers, with 50 samples from each species.
        
    The goal is to classify the species based on four morphological measurements.
""")
    
st.subheader("Anatomy of the Iris Flower ")
st.markdown("""
    The classification is based on four key measurements, all measured in centimeters:

    * **Sepal Length/Width:** The sepal is the outer part of the flower (often green) that encloses the petals in the bud stage.
    * **Petal Length/Width:** The petal is the colorful part of the flower that attracts pollinators.
""")
    
st.subheader("The Three Target Species")
    
st.markdown("""
    The dataset contains equal samples (50 each) of the following three species:

    1.  **Iris setosa:** This species is typically the easiest to identify as it has significantly shorter petals and sepals, making it linearly separable from the others.
    2.  **Iris versicolor:** This species falls between the other two in terms of feature measurements.
    3.  **Iris virginica:** This species generally has the longest petals and sepals and is often the most difficult to distinguish from *Iris versicolor* based solely on simple linear separation.
""")
    
st.subheader("DuckDB Data Sample")
st.markdown("We're using DuckDB to query the dataset loaded in-memory.")