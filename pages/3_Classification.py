import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.datasets import load_iris
from io import BytesIO
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet

# Load iris dataset
iris = load_iris(as_frame=True)
iris_df = iris.frame
iris_df['species_name'] = iris_df['target'].map(dict(enumerate(iris.target_names)))

X = iris_df.drop(['target', 'species_name'], axis=1)
y = iris_df['target']

st.title("Model Comparison and Evaluation")
st.markdown("We will compare two common classification algorithms: *K-Nearest Neighbors (KNN)* and *Support Vector Machine (SVM). **Note:* Since KNN and SVM are sensitive to feature scales, we use a StandardScaler within a Pipeline for best practice.")

# Sidebar for Model Tuning
st.sidebar.header("Model Parameters")
test_size = st.sidebar.slider("Test Set Size Ratio", 0.1, 0.5, 0.3, 0.05)
random_state = st.sidebar.slider("Random State (Seed)", 0, 100, 42)
st.sidebar.subheader("KNN Settings")
n_neighbors = st.sidebar.slider("Number of Neighbors (k)", 1, 20, 5)
st.sidebar.subheader("SVM Settings")
svm_c = st.sidebar.slider("SVM Regularization (C)", 0.1, 10.0, 1.0, 0.1)

# Split Data
st.subheader("Data Split")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
st.info(f"Training set size: {len(X_train)} samples | Test set size: {len(X_test)} samples")


# --- Training and Evaluation ---
st.header("Training and Results")
col_knn, col_svm = st.columns(2)

# Initialize report variables
report_knn = "KNN model not run."
report_svm = "SVM model not run."
acc_knn = 0.0
acc_svm = 0.0

# 1. K-Nearest Neighbors (KNN)
with col_knn:
    st.subheader("K-Nearest Neighbors (KNN) with Scaling Pipeline")
    try:
        knn_pipe = Pipeline([
            ('scaler', StandardScaler()),
            ('knn', KNeighborsClassifier(n_neighbors=n_neighbors))
        ])
        knn_pipe.fit(X_train, y_train)
        y_pred_knn = knn_pipe.predict(X_test)
        
        acc_knn = accuracy_score(y_test, y_pred_knn)
        st.metric(label="Accuracy Score", value=f"{acc_knn:.4f}")
        
        st.markdown("#### Confusion Matrix")
        cm_knn = confusion_matrix(y_test, y_pred_knn)
        fig_knn, ax_knn = plt.subplots(figsize=(6, 5))
        sns.heatmap(cm_knn, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=iris_df['species_name'].unique(), 
                    yticklabels=iris_df['species_name'].unique(), ax=ax_knn)
        ax_knn.set_ylabel('True Species')
        ax_knn.set_xlabel('Predicted Species')
        st.pyplot(fig_knn)
        
        st.markdown("#### Classification Report")
        report_knn = classification_report(y_test, y_pred_knn, target_names=iris_df['species_name'].unique(), output_dict=False)
        st.text(report_knn)
    except Exception as e:
        st.error(f"Error running KNN model: {e}")
        report_knn = f"Error running KNN model: {e}"

# 2. Support Vector Machine (SVM)
with col_svm:
    st.subheader("Support Vector Machine (SVM) with Scaling Pipeline")
    try:
        svm_pipe = Pipeline([
            ('scaler', StandardScaler()),
            ('svm', SVC(C=svm_c, random_state=random_state))
        ])
        svm_pipe.fit(X_train, y_train)
        y_pred_svm = svm_pipe.predict(X_test)
        
        acc_svm = accuracy_score(y_test, y_pred_svm)
        st.metric(label="Accuracy Score", value=f"{acc_svm:.4f}")
        
        st.markdown("#### Confusion Matrix")
        cm_svm = confusion_matrix(y_test, y_pred_svm)
        fig_svm, ax_svm = plt.subplots(figsize=(6, 5))
        sns.heatmap(cm_svm, annot=True, fmt='d', cmap='Reds', 
                    xticklabels=iris_df['species_name'].unique(), 
                    yticklabels=iris_df['species_name'].unique(), ax=ax_svm)
        ax_svm.set_ylabel('True Species')
        ax_svm.set_xlabel('Predicted Species')
        st.pyplot(fig_svm)
        
        st.markdown("#### Classification Report")
        report_svm = classification_report(y_test, y_pred_svm, target_names=iris_df['species_name'].unique(), output_dict=False)
        st.text(report_svm)
    except Exception as e:
        st.error(f"Error running SVM model: {e}")
        report_svm = f"Error running SVM model: {e}"

st.header("Conclusion")
st.markdown("Even with proper scaling, this dataset is often perfectly separable, leading to very high or 100% accuracy. Try adjusting the Test Set Size Ratio or the Random State (Seed) in the sidebar to observe how the model performance might fluctuate with different data splits.")

if acc_knn > acc_svm:
    st.success(f"*KNN* achieved the highest accuracy ({acc_knn:.4f}) with the current parameters.")
elif acc_svm > acc_knn:
    st.success(f"*SVM* achieved the highest accuracy ({acc_svm:.4f}) with the current parameters.")
else:
    st.info(f"Both models achieved the same accuracy ({acc_knn:.4f}).")

# --- PDF Report Button ---
st.markdown("---")
st.subheader("Generate Report")
st.info("Download a PDF report of the model parameters and classification results.")

def create_classification_pdf(params, knn_report, svm_report):
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []
    
    story.append(Paragraph("Model Classification Report", styles['h1']))
    story.append(Paragraph("Parameters", styles['h2']))
    
    param_text = f"""
    Test Set Size Ratio: {params['test_size']}<br/>
    Random State (Seed): {params['random_state']}<br/>
    KNN - Neighbors (k): {params['n_neighbors']}<br/>
    SVM - Regularization (C): {params['svm_c']}
    """
    story.append(Paragraph(param_text, styles['BodyText']))
    story.append(Spacer(1, 12))
    
    story.append(Paragraph("K-Nearest Neighbors (KNN) Report", styles['h2']))
    story.append(Paragraph(knn_report.replace("\n", "<br/>").replace(" ", "&nbsp;"), styles['Code']))
    story.append(Spacer(1, 12))

    story.append(Paragraph("Support Vector Machine (SVM) Report", styles['h2']))
    story.append(Paragraph(svm_report.replace("\n", "<br/>").replace(" ", "&nbsp;"), styles['Code']))

    doc.build(story)
    buffer.seek(0)
    return buffer

current_params = {
    "test_size": test_size,
    "random_state": random_state,
    "n_neighbors": n_neighbors,
    "svm_c": svm_c
}

pdf_buffer = create_classification_pdf(current_params, report_knn, report_svm)

st.download_button(
    label="Download Classification Report (PDF)",
    data=pdf_buffer,
    file_name="classification_report.pdf",
    mime="application/pdf",
)
