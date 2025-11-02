import streamlit as st
from io import BytesIO
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet

# Set the title and icon for the app
st.set_page_config(
    page_title="IRIS CLASSIFICATION WEB APP",
    layout="wide"
)

# Home page content
st.title("IRIS Classification Project: An Introduction")
st.subheader("Welcome to the Interactive Data Analysis and Classification App!")

intro_markdown = """
This application explores the famous *Iris Dataset* and compares two common machine learning classification algorithms.
        
It utilizes *Streamlit* for the interactive user interface, *DuckDB* for fast, analytical data processing, 
and *Scikit-learn* for model training and evaluation.

The Iris dataset is a classic and widely used dataset in machine learning 
and statistics. It contains 150 samples of Iris flowers, with 50 samples from each of 
three species: *Iris Setosa, **Iris Versicolor, and **Iris Virginica*.
"""
st.markdown(intro_markdown)

st.header("The Iris Dataset Overview")
    
st.subheader("Dataset Details")
details_markdown = """
The Iris dataset is a classic multivariate data set introduced by R.A. Fisher in 1936.
It consists of 150 samples from three species of Iris flowers, with 50 samples from each species.
        
The goal is to classify the species based on four morphological measurements.
"""
st.markdown(details_markdown)

st.subheader("Anatomy of the Iris Flower ")
anatomy_markdown = """
    The classification is based on four key measurements, all measured in centimeters:

    * *Sepal Length/Width:* The sepal is the outer part of the flower (often green) that encloses the petals in the bud stage.
    * *Petal Length/Width:* The petal is the colorful part of the flower that attracts pollinators.
"""
st.markdown(anatomy_markdown)

st.subheader("The Three Species")
st.markdown("Each species has distinct physical characteristics, which allows the machine learning models to classify them based on their measurements.")

# Image Grid for Species Comparison 
col1, col2, col3 = st.columns(3)

with col1:
    st.image("iris_setosa.webp", caption="Iris Setosa", use_container_width=True)
    st.markdown("*Iris Setosa:* Characterized by shorter and wider petals and sepals. Generally the easiest species to separate from the other two.")

with col2:
    st.image("iris_versicolor.jpeg", caption="Iris Versicolor", use_container_width=True)
    st.markdown("*Iris Versicolor:* Has intermediate measurements. Its petal and sepal dimensions often fall between those of Setosa and Virginica.")

with col3:
    st.image("iris_virginica.jpeg", caption="Iris Virginica", use_container_width=True)
    st.markdown("*Iris Virginica:* Typically has the longest and widest petals and sepals among the three species.")

# Navigation Footer 
st.markdown("---")
st.subheader("Ready to Explore?")
st.markdown("Proceed to the *Basic EDA* page in the sidebar to visualize the data, or try the *SQL Playground*!")

# --- PDF Report Button ---
st.markdown("---")
st.subheader("Generate Report")
st.info("Download a PDF summary of this page.")

def create_home_pdf():
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []
    
    story.append(Paragraph("IRIS Classification Project: An Introduction", styles['h1']))
    story.append(Paragraph("Welcome to the Interactive Data Analysis and Classification App!", styles['h2']))
    
    # Clean up markdown for PDF
    story.append(Paragraph(intro_markdown.replace('*', ''), styles['BodyText']))
    story.append(Spacer(1, 12))
    
    story.append(Paragraph("The Iris Dataset Overview", styles['h2']))
    story.append(Paragraph("Dataset Details", styles['h3']))
    story.append(Paragraph(details_markdown.replace('*', ''), styles['BodyText']))
    story.append(Spacer(1, 12))
    
    story.append(Paragraph("Anatomy of the Iris Flower", styles['h3']))
    story.append(Paragraph(anatomy_markdown.replace('*', '').replace('    *', ' - '), styles['BodyText']))
    
    doc.build(story)
    buffer.seek(0)
    return buffer

pdf_buffer = create_home_pdf()

st.download_button(
    label="Download Home Report (PDF)",
    data=pdf_buffer,
    file_name="home_report.pdf",
    mime="application/pdf",
)
