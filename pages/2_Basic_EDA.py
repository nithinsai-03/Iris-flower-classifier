import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
import duckdb
from io import BytesIO
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
import plotly.express as px

# Page Configuration 
st.set_page_config(
    page_title="Basic EDA",
    layout="wide"
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
st.dataframe(iris_df.sample(8), use_container_width=True)
st.write(f"Dataset Shape: {iris_df.shape[0]} rows, {iris_df.shape[1]} columns")

# 3. Display Basic Statistics
st.header("2. Descriptive Statistics")
descriptive_stats = iris_df.describe()
st.dataframe(descriptive_stats, use_container_width=True)

# 4. Simple Visualization (Histogram for a single feature)
st.header("3. Feature Distribution")
feature = st.selectbox(
    "Select a feature to visualize:",
    iris_df.columns[0:4] # Exclude species_id and species name columns
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

# 6. Scatter Matrix Plot 
numeric_features = list(iris_df.columns[:4]) 

st.header("5. Interactive Feature Pair Plots")

# User selection 
selected_feats = st.multiselect(
    "Select features for pair plots:",
    options=numeric_features,
    default=numeric_features
)

# Use selected features, but default to all 4 if the list is empty
dims = selected_feats if len(selected_feats) >= 2 else list(iris_df.columns[:4]) 

fig_scatter = px.scatter_matrix(
    iris_df,
    dimensions=dims,
    color="species",
    title="Interactive Pairwise Relationships",
    opacity=0.8,
    height=900
)
 
fig_scatter.update_traces(diagonal_visible=True, marker=dict(size=6)) 
fig_scatter.update_layout(
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
) 
st.plotly_chart(fig_scatter, use_container_width=True) 

# --- PDF Report Button ---
st.markdown("---")
st.subheader("Generate Report")
st.info("Download a PDF report of the descriptive statistics.")

def create_eda_pdf(stats_df):
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []
    
    story.append(Paragraph("Basic EDA Report", styles['h1']))
    story.append(Paragraph("Descriptive Statistics", styles['h2']))
    
    # Convert DataFrame to a string for the PDF
    try:
        stats_string = stats_df.to_string()
    except:
        stats_string = "Error converting stats to string."
            
    story.append(Paragraph(stats_string.replace("\n", "<br/>"), styles['Code']))

    doc.build(story)
    buffer.seek(0)
    return buffer

pdf_buffer = create_eda_pdf(descriptive_stats)

st.download_button(
    label="Download Stats Report (PDF)",
    data=pdf_buffer,
    file_name="eda_stats_report.pdf",
    mime="application/pdf",
)
