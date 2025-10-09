import streamlit as st

# Set the title and icon for the app
st.set_page_config(
    page_title="IRIS CLASSIFICATION WEB APP",
    page_icon="🌸",
    layout="wide"
)

# --- HOME PAGE CONTENT ---
st.title("🌸 IRIS Classification Project: An Introduction")
st.subheader("Welcome to the Interactive Data Analysis and Classification App!")

st.markdown("""
This web application is built using **Streamlit** to explore the famous **Iris dataset** and provide a machine learning model for flower classification.

The Iris dataset is a classic and widely used dataset in machine learning 
and statistics. It contains 150 samples of Iris flowers, with 50 samples from each of 
three species: **Iris Setosa**, **Iris Versicolor**, and **Iris Virginica**.

The data records four features (measurements in centimeters) for each flower:
- **Sepal Length**
- **Sepal Width**
- **Petal Length**
- **Petal Width**
""")

st.header("The Three Species")
st.markdown("Each species has distinct physical characteristics, which allows the machine learning models to classify them based on their measurements.")

# --- Image Grid for Species Comparison ---
col1, col2, col3 = st.columns(3)

with col1:
    st.image("https://upload.wikimedia.org/wikipedia/commons/5/56/Iris_setosa_2.JPG", caption="Iris Setosa", use_column_width=True)
    st.markdown("**Iris Setosa:** Characterized by shorter and wider petals and sepals. Generally the easiest species to separate from the other two.")

with col2:
    st.image("https://upload.wikimedia.org/wikipedia/commons/4/41/Iris_versicolor_3.jpg", caption="Iris Versicolor", use_column_width=True)
    st.markdown("**Iris Versicolor:** Has intermediate measurements. Its petal and sepal dimensions often fall between those of Setosa and Virginica.")

with col3:
    st.image("https://upload.wikimedia.org/wikipedia/commons/9/9f/Iris_virginica.jpg", caption="Iris Virginica", use_column_width=True)
    st.markdown("**Iris Virginica:** Typically has the longest and widest petals and sepals among the three species.")

st.markdown("""
***
### Navigation Guide

Use the sidebar on the left to navigate the application:
- **Basic EDA:** Dive into the data! Explore the dataset through visual graphs (like histograms and heatmaps) and descriptive statistics.
- **Classification:** Test our machine learning models! Input your own flower measurements and see the predicted species using models like Random Forest, KNN, and SVM.
""")

# Optional: Display the logo if you have it
try:
    st.image('logo.jpg', width=100)
except FileNotFoundError:
    # If the user added a logo.jpg, it will show up here.
    pass