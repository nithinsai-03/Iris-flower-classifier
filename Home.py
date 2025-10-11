import streamlit as st
from sklearn.datasets import load_iris
import duckdb

# Set the title and icon for the app
st.set_page_config(
    page_title="IRIS CLASSIFICATION WEB APP",
    layout="wide"
)

# Home page content
st.title("IRIS Classification Project: An Introduction")
st.subheader("Welcome to the Interactive Data Analysis and Classification App!")

st.markdown("""
This application explores the famous **Iris Dataset** and compares two common machine learning classification algorithms.
        
It utilizes **Streamlit** for the interactive user interface, **DuckDB** for fast, analytical data processing, 
and **Scikit-learn** for model training and evaluation.

The Iris dataset is a classic and widely used dataset in machine learning 
and statistics. It contains 150 samples of Iris flowers, with 50 samples from each of 
three species: **Iris Setosa**, **Iris Versicolor**, and **Iris Virginica**.
""")

st.header("The Iris Dataset Overview")
    
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

st.subheader("The Three Species")
st.markdown("Each species has distinct physical characteristics, which allows the machine learning models to classify them based on their measurements.")

# Image Grid for Species Comparison 
col1, col2, col3 = st.columns(3)

with col1:
    st.image("iris_setosa.webp", caption="Iris Setosa", width='stretch')
    st.markdown("**Iris Setosa:** Characterized by shorter and wider petals and sepals. Generally the easiest species to separate from the other two.")

with col2:
    st.image("iris_versicolor.jpeg", caption="Iris Versicolor", width='stretch')
    st.markdown("**Iris Versicolor:** Has intermediate measurements. Its petal and sepal dimensions often fall between those of Setosa and Virginica.")

with col3:
    st.image("iris_virginica.jpeg", caption="Iris Virginica", width='stretch')
    st.markdown("**Iris Virginica:** Typically has the longest and widest petals and sepals among the three species.")

    

    
st.subheader("DuckDB Data Sample")
st.markdown("We're using DuckDB to query the dataset loaded in-memory.")

@st.cache_resource
def get_duckdb_conn():
    """Initializes and caches the DuckDB connection object (using st.cache_resource for non-serializable objects)."""
    return duckdb.connect(database=':memory:', read_only=False)

@st.cache_data
def load_data_df():
    """Loads Iris data and prepares it as a Pandas DataFrame (using st.cache_data for serializable data)."""
    # Load data from scikit-learn
    iris = load_iris(as_frame=True)
    df = iris.frame
    df.columns = [col.replace(' (cm)', '').replace(' ', '_') for col in df.columns]
    
    # Map target integers to species names
    target_names = {0: 'setosa', 1: 'versicolor', 2: 'virginica'}
    df['species_name'] = df['target'].map(target_names)
    
    return df

# Initialize components and register the data frame with the connection
df_iris = load_data_df()
db_conn = get_duckdb_conn()

# Demonstrating a DuckDB query
query = """
SELECT sepal_length, sepal_width, petal_length, petal_width, species_name
FROM iris_table
ORDER BY RANDOM()
LIMIT 10
"""

# Register the DataFrame with the DuckDB connection
try:
    db_conn.register('iris_table', df_iris)
except duckdb.CatalogException:
    # Table might already be registered across reruns
    pass 
    
try:
    data_sample = db_conn.execute(query).fetchdf()
    st.dataframe(data_sample, use_container_width=True)
    st.code(f"DuckDB Query:\n{query}", language="sql")
except Exception as e:
    st.error(f"Error executing DuckDB query: {e}")


# Navigation Footer 
st.markdown("---")
st.subheader("Ready to Explore?")
st.markdown("Proceed to the **Basic EDA** page in the sidebar to visualize the data!")