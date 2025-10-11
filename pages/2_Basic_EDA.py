import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
import duckdb

# Page Configuration 
st.set_page_config(
    page_title="Basic EDA",
)


st.title("Basic Exploratory Data Analysis (EDA)")

# 1. Load the Data
@st.cache_data
def load_data():
    """Loads the Iris dataset and returns a Pandas DataFrame with cleaned columns."""
    iris = load_iris(as_frame=True)
    df = iris.frame
    
    
    df.columns = [col.replace(' (cm)', '').replace(' ', '_') for col in df.columns]
    
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
st.dataframe(iris_df.sample(8))
st.write(f"Dataset Shape: {iris_df.shape[0]} rows, {iris_df.shape[1]} columns")


# 3. Display Basic Statistics
st.header("2. Descriptive Statistics")
st.dataframe(iris_df.describe())

# 4. SQL Query on dataset
st.header("3. SQL operation on IRIS Dataset" )

iris = load_iris(as_frame=True)
df = iris.frame

df['species'] = df['target'].apply(lambda x: iris.target_names[x])
df.drop(columns=['target'], inplace=True)

con = duckdb.connect(database=':memory:')
con.register('iris_df', df)

a=con.execute("CREATE TABLE iris AS SELECT * FROM iris_df")

query = st.text_area("Enter your SQL query here:", "SELECT * FROM iris LIMIT 5")

if st.button('Run Query'):
    try:
        result = con.execute(query).fetchdf()
        st.write("Query Result:")
        st.dataframe(result)
    except Exception as e:
        st.error(f"Error: {str(e)}")

result = con.execute(query).fetchdf()
st.write("Query Result:")
st.dataframe(result)

# 5. Simple Visualization (Histogram for a single feature)
st.header("4. Feature Distribution")
feature = st.selectbox(
    "Select a feature to visualize:",
    iris_df.columns[0:4] # Exclude species_id and species name columns
)

fig, ax = plt.subplots()
# Create a histogram of the selected feature, colored by species
sns.histplot(data=iris_df, x=feature, hue='species', kde=True, ax=ax)
ax.set_title(f'Distribution of {feature}')
st.pyplot(fig)

# 6. Correlation Heatmap
st.header("5. Feature Correlation")
fig_corr, ax_corr = plt.subplots(figsize=(8, 6))
correlation_matrix = iris_df.iloc[:, :-2].corr() # Correlate only numeric features
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', ax=ax_corr)
ax_corr.set_title('Correlation Matrix of Iris Features')
st.pyplot(fig_corr)
import plotly.express as px

# 7. Scatter Matrix Plot 
numeric_features = list(iris_df.columns[:4]) 

st.header("6. Interactive Feature Pair Plots")

# User selection 
selected_feats = st.multiselect(
    "Select features for pair plots:",
    options=numeric_features,
    default=numeric_features
)

# Use selected features, but default to all 4 if the list is empty (can happen during app rerun)
dims = selected_feats if len(selected_feats) >= 2 else list(iris_df.columns[:4]) 

fig_scatter = px.scatter_matrix(
    iris_df,
    dimensions=dims,
    color="species",
    title="Interactive Pairwise Relationships",
    opacity=0.8,
    width=1200,  
    height=900
)
 
fig_scatter.update_traces(diagonal_visible=True, marker=dict(size=6)) 
fig_scatter.update_layout(
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    margin=dict(l=40, r=20, t=60, b=40)
) 


st.plotly_chart(fig_scatter, use_container_width=True) #To expand it across the page