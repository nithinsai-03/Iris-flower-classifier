import streamlit as st
import duckdb
from sklearn.datasets import load_iris
import pandas as pd
from io import BytesIO
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph
from reportlab.lib.styles import getSampleStyleSheet

st.set_page_config(page_title="SQL Playground", layout="wide")
st.title("🗃️ SQL Playground")

st.markdown("""
Welcome to the SQL Playground. Here, you can directly query the Iris dataset using DuckDB. 
- Use the **Data Editor** to see the full table.
- Use the **Schema Explorer** to see column names.
- Try an **Example Query** or write your own!
""")

# --- Database Setup (Shared Logic) ---
@st.cache_resource
def get_duckdb_conn():
    """Initializes and caches the DuckDB connection."""
    return duckdb.connect(database=':memory:', read_only=False)

@st.cache_data
def load_data_df():
    """Loads Iris data and prepares it as a Pandas DataFrame."""
    iris = load_iris(as_frame=True)
    df = iris.frame
    df.columns = [col.replace(' (cm)', '').replace(' ', '_') for col in df.columns]
    target_names = {0: 'setosa', 1: 'versicolor', 2: 'virginica'}
    df['species_name'] = df['target'].map(target_names)
    df.drop(columns=['target'], inplace=True) # Drop the numeric target
    return df

# Initialize connection and load data
db_conn = get_duckdb_conn()
df_iris = load_data_df()

# Register the DataFrame with DuckDB
try:
    db_conn.register('iris_table', df_iris)
except duckdb.Error:
    # This might fail on a rerun if already registered, which is fine
    pass

# --- Page Layout ---

# 1. Data Editor (Your requested feature!)
st.header("Full Dataset (`iris_table`)")
st.markdown("This is the full dataset, loaded into an interactive `st.data_editor`. You can sort and filter, but edits won't be saved.")
st.data_editor(df_iris, use_container_width=True, height=300)

col1, col2 = st.columns([1, 2])

with col1:
    # 2. Schema Explorer (My suggestion)
    st.header("Schema Explorer")
    st.info("Use these column names in your query.")
    schema_df = pd.DataFrame({
        "Column Name": df_iris.columns,
        "Data Type": [str(dtype) for dtype in df_iris.dtypes]
    })
    st.dataframe(schema_df, use_container_width=True, hide_index=True)

    # 3. Example Queries (My suggestion)
    st.header("Example Queries")
    example_queries = {
        "View all data": "SELECT * FROM iris_table LIMIT 20;",
        "Count per species": "SELECT species_name, COUNT(*) AS count FROM iris_table GROUP BY species_name;",
        "Average petal length by species": "SELECT species_name, AVG(petal_length) AS avg_petal_length FROM iris_table GROUP BY species_name;",
        "Find largest sepals": "SELECT * FROM iris_table ORDER BY sepal_length DESC, sepal_width DESC LIMIT 5;"
    }
    selected_query = st.selectbox("Select an example query:", options=list(example_queries.keys()))
    default_query = example_queries[selected_query]

with col2:
    # 4. SQL Query Box (Moved from EDA page)
    st.header("Your SQL Query")
    query = st.text_area("Enter your SQL query:", default_query, height=150)

    if st.button('Run Query'):
        try:
            result_df = db_conn.execute(query).fetchdf()
            st.subheader("Query Result")
            st.dataframe(result_df, use_container_width=True)
            st.session_state['last_sql_query'] = query
            st.session_state['last_sql_result'] = result_df
        except Exception as e:
            st.error(f"Error: {str(e)}")
            st.session_state['last_sql_query'] = query
            st.session_state['last_sql_result'] = None
    
    # 5. PDF Report Button
    st.markdown("---")
    st.subheader("Generate Report")
    
    if 'last_sql_result' in st.session_state and st.session_state['last_sql_result'] is not None:
        st.info("Click the button to download a PDF report of your last successful query.")

        # Function to generate PDF
        def create_sql_pdf(query, data_df):
            buffer = BytesIO()
            doc = SimpleDocTemplate(buffer, pagesize=letter)
            styles = getSampleStyleSheet()
            story = []
            
            story.append(Paragraph("SQL Playground Report", styles['h1']))
            
            story.append(Paragraph("Query:", styles['h3']))
            story.append(Paragraph(query.replace("\n", "<br/>"), styles['Code']))
            
            story.append(Paragraph("Results:", styles['h3']))
            # Convert DataFrame to a simple string for the PDF
            try:
                result_string = data_df.to_string()
            except:
                result_string = "Error converting results to string."
            
            story.append(Paragraph(result_string.replace("\n", "<br/>"), styles['Code']))

            doc.build(story)
            buffer.seek(0)
            return buffer

        # Get data from session state
        query_to_report = st.session_state.get('last_sql_query', 'No query run.')
        result_to_report = st.session_state.get('last_sql_result')

        if result_to_report is not None:
            pdf_buffer = create_sql_pdf(query_to_report, result_to_report)
            st.download_button(
                label="Download SQL Report (PDF)",
                data=pdf_buffer,
                file_name="sql_playground_report.pdf",
                mime="application/pdf",
            )
    else:
        st.warning("Run a successful query first to generate a report.")
