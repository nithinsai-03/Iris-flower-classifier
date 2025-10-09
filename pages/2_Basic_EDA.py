import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
import duckdb

# --- Page Configuration ---
st.set_page_config(
    page_title="Basic EDA",
    page_icon="📊",
)
st.title("SQL operation on IRIS Dataset" )

iris = load_iris(as_frame=True)
df = iris.frame

df['species'] = df['target'].apply(lambda x: iris.target_names[x])

con = duckdb.connect(database=':memory:')
con.register('iris_df', df)

a=con.execute("CREATE TABLE iris AS SELECT * FROM iris_df")

query = st.text_area("Enter your SQL query here:", "SELECT * FROM iris LIMIT 5")

result = con.execute(query).fetchdf()
st.write("Query Result:")
st.dataframe(result)



st.title("📊 Basic Exploratory Data Analysis (EDA)")

# 1. Load the Data
# Using scikit-learn's built-in iris dataset for clean loading
@st.cache_data
def load_data():
    """Loads the Iris dataset and returns a Pandas DataFrame."""
    iris = load_iris(as_frame=True)
    df = iris.frame
    # Rename target column for better readability
    df.rename(columns={'target': 'species_id'}, inplace=True)
    # Map species ID to species name for EDA
    df['species'] = df['species_id'].map({
        0: 'Setosa',
        1: 'Versicolor',
        2: 'Virginica'
    })
    return df

iris_df = load_data()

# 2. Display Data Snapshot
st.header("1. Dataset Snapshot")
st.dataframe(iris_df.head())
st.write(f"Dataset Shape: {iris_df.shape[0]} rows, {iris_df.shape[1]} columns")

# 3. Display Basic Statistics
st.header("2. Descriptive Statistics")
st.dataframe(iris_df.describe())

# 4. Simple Visualization (Histogram for a single feature)
st.header("3. Feature Distribution")
feature = st.selectbox(
    "Select a feature to visualize:",
    iris_df.columns[:-2] # Exclude species_id and species name columns
)

fig, ax = plt.subplots()
# Create a histogram of the selected feature, colored by species
sns.histplot(data=iris_df, x=feature, hue='species', kde=True, ax=ax)
ax.set_title(f'Distribution of {feature}')
st.pyplot(fig)

# 5. Correlation Heatmap
st.header("4. Feature Correlation")
fig_corr, ax_corr = plt.subplots(figsize=(8, 6))
correlation_matrix = iris_df.iloc[:, :-2].corr() # Correlate only numeric features
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', ax=ax_corr)
ax_corr.set_title('Correlation Matrix of Iris Features')
st.pyplot(fig_corr)