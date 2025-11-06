# Iris-IDEAS-TIH

**Iris-IDEAS-TIH** is an interactive web application for **Iris Flower Classification**, developed using **Streamlit**. This project serves as a practical demonstration of how **data analysis**, **visualization**, and **machine learning** can be combined to build a user-friendly, end-to-end predictive model.

The application leverages the classic **Iris dataset** from the UCI Machine Learning Repository, which contains measurements of sepal length, sepal width, petal length, and petal width for three species of Iris flowers: *Setosa*, *Versicolor*, and *Virginica*. Users can interactively explore the dataset, visualize patterns, train machine learning models, and evaluate their predictions—all within a clean and intuitive web interface.

### Key Features:

- **Interactive EDA**: Explore the dataset using dynamic visualizations and summary statistics to understand feature distributions and relationships.
- **Machine Learning Classification**: Train and test different classification models - **Support Vector Machine (SVM)**,**K-Nearest Neighbor (KNN)** and see real-time predictions for flower species.
- **Visualization of Results**: Displayed model performance metrics, accuracy scores, and graphical outputs for better insights.
- **User-Friendly Interface**: Streamlit provides a smooth and interactive interface, making it accessible to beginners and data science enthusiasts alike.

This project is ideal for anyone looking to learn **Python-based data analysis**, **machine learning workflow**, and how to deploy a model in a **web application** format without heavy backend requirements.

---

## 📂 Project Structure



Iris-IDEAS-TIH/
├── pages/
│   ├── 2_Basic_EDA.py
│   ├── 3_Classification.py
│   ├── 4_SQL.py 
├── Home.py
├── iris_setosa.webp
├── iris_versicolor.jpeg
├── iris_virginica.jpeg
├── README.md
└── requirements.txt


---

## Features

- **Basic EDA** — Visualize and explore the Iris dataset with interactive plots and statistics.  
- **Classification** — Train and evaluate machine learning models - SVM(Support Vector Machine) vs KNN(K-Nearest Neighbours)
- **Image Support** — Visual representation of different Iris flower species.  
- **User-Friendly UI** — Built with Streamlit for an easy and interactive interface.

---

## Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/achal22-coder/Iris-IDEAS-TIH.git
cd Iris-IDEAS-TIH
```
### 2. Create & Activate Virtual Environment (Optional)

```bash
python -m venv venv

### For macOS/Linux:
source venv/bin/activate

###For Windows:
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Application

```bash
streamlit run Home.py
```
---

## Workflow Overview

1. **Home Page** – Introduction to the project and easy navigation to different modules.  
2. **EDA Page** – Explore the dataset with interactive visualizations and summary statistics.  
3. **Classification Page** – Train machine learning models, view predictions, and check accuracy.  
4. **Results** – Display evaluation metrics and visual output for better insights.

---

## Tech Stack

- **Language**: Python  
- **Framework**: Streamlit  
- **Data Analysis**: Pandas, NumPy  
- **Machine Learning**: Scikit-learn  
- **Visualization**: Matplotlib, Seaborn  
- **Dataset**: Iris Dataset (UCI Machine Learning Repository)

---


### This project is made under supervision od Adrija Das ma'am as a part of ISI Kolkata IDEAS-TIH Autumn Internship Program.
