import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from vizualize import *
from aggregation import *
from preprocessing.data_cleaning import *
from preprocessing.data_loaders import *
from preprocessing.data_transformation import *

# Load the telecom data
@st.cache_data
def load_data():
    """Load dataset from the PostgreSQL database."""
    try:
        return load_data_using_sqlalchemy("SELECT * FROM xdr_data;")
    except Exception as e:
        st.error(f"Failed to load data from the PostgreSQL database: {e}")
        return pd.DataFrame()

        
# Load the data

# Load data (upload a file using Streamlit file uploader)
# uploaded_file = st.file_uploader("Choose a file", type="csv")
# st.sidebar.subheader("Upload your data")
# uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type="csv")
# if uploaded_file:
#     df = load_data(uploaded_file)  # Load data directly from the uploaded file
#     df = pd.read_csv(uploaded_file)
#     st.write(df.head())  # Show first few rows of the uploaded file



# Data loading and preprocessing
df = load_data()
if not df.empty:
    df_cleaned = clean_data(df)
    # df_transformed = transform_data(df_cleaned)

    # Aggregate engagement metrics andget top 10 users for each metric
    df_engagement = aggregate_engagement_metrics(df_cleaned)
    top_10_session_frequency, top_10_session_duration, top_10_traffic = top_10_engagement(df_engagement)
    df_normalized = normalize_metrics(df_engagement, normalize_type='minmax')    
    df_clustered = kmeans_clustering(df_normalized, n_clusters=3)    
    cluster_stats = cluster_statistics(df_clustered)    
    top_users_per_app = aggregate_application_traffic(df_cleaned)  

    # Dashboard title and introduction
    st.title("Telecom Data Visualization and Exploration")
    st.write("""
    This interactive dashboard allows you to explore telecom data through various visualization techniques. 
    Use the tabs below to navigate through different types of plots.
    """)

    # Tabbed layout for visualizations
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs(
        ["Scatter Plot", "Bar Chart", "Histogram", "Boxplot", "Pairplot", "Correlation Matrix", "Distribution Plot", "User Engagement"]
    )

    # Scatter Plot Tab
    with tab1:
        st.subheader("Scatter Plot")
        st.write("Select the X and Y axis to visualize relationships between two metrics.")
        x_axis = st.selectbox("Select X-axis variable:", df_cleaned.columns, key="scatter_x")
        y_axis = st.selectbox("Select Y-axis variable:", df_cleaned.columns, key="scatter_y")
        scatter_color = st.color_picker("Pick a color:", "#1f77b4")
        fig = plot_data(
            df_cleaned, plot_type='scatter', x=x_axis, y=y_axis, 
            title=f"Scatter Plot of {x_axis} vs {y_axis}", color=scatter_color
        )
        st.pyplot(fig)

    # Bar Chart Tab
    with tab2:
        st.subheader("Bar Chart")
        st.write("Select a column to visualize its distribution with a bar chart.")
        bar_column = st.selectbox("Select column for Bar Chart:", df_cleaned.columns, key="bar_column")
        fig = plot_data(df_cleaned, plot_type='bar', x=bar_column, title=f"Bar Chart of {bar_column}")
        st.pyplot(fig)

    # Histogram Tab
    with tab3:
        st.subheader("Histogram")
        st.write("Select a column and adjust the number of bins to see the distribution.")
        hist_column = st.selectbox("Select column for Histogram:", df_cleaned.columns, key="hist_column")
        bins = st.slider("Select number of bins:", 5, 100, 20)
        kde_option = st.checkbox("Overlay KDE Curve", value=True)
        fig = plot_data(
            df_cleaned, plot_type='histogram', x=hist_column, bins=bins, kde=kde_option, 
            title=f"Histogram of {hist_column}"
        )
        st.pyplot(fig)

    # Boxplot Tab
    with tab4:
        st.subheader("Boxplot")
        st.write("Boxplot to visualize the distribution and detect outliers in a selected variable.")
        box_column = st.selectbox("Select column for Boxplot:", df_cleaned.columns, key="box_column")
        fig = plot_data(
            df_cleaned, plot_type='boxplot', x=box_column, title=f"Boxplot of {box_column}"
        )
        st.pyplot(fig)

    # Pairplot Tab
    with tab5:
        st.subheader("Pairplot")
        st.write("Pairwise relationships between variables for better insight into correlations.")
        selected_columns = st.multiselect(
            "Select columns for Pairplot (up to 5):", df_cleaned.columns, default=df_cleaned.columns[:5]
        )
        if len(selected_columns) > 1:
            fig = sns.pairplot(df_cleaned[selected_columns])
            st.pyplot(fig)

    # Correlation Matrix Tab
    with tab6:
        st.subheader("Correlation Matrix")
        st.write("View the correlation matrix to understand relationships between different variables.")
        fig = plot_correlation_matrix(
            df_cleaned, df_cleaned.select_dtypes(include='number').columns
        )
        st.pyplot(fig)

    # Distribution Plot Tab
    with tab7:
        st.subheader("Distribution Plot")
        st.write("Visualize the distribution of a variable using a Kernel Density Estimation (KDE).")
        dist_column = st.selectbox("Select column for Distribution Plot:", df_cleaned.columns, key="dist_column")
        shade_option = st.checkbox("Shade under the curve", value=True)
        fig = plot_data(
            df_cleaned, plot_type='kde', x=dist_column, shade=shade_option, 
            title=f"Distribution Plot of {dist_column}"
        )
        st.pyplot(fig)

    # User Engagement Analysis Tab
    with tab8:
        st.title("User Engagement Analysis")
        st.write("Here you can explore user engagement metrics and see which users have the highest interactions.")

        # Display the top 10 users for each engagement metric
        st.subheader("Top 10 Users by Session Frequency")
        fig = plot_top_10(df_engagement, 'SessionFrequency', xlabel='Customer ID (MSISDN)', ylabel='Sessions Frequency', title='Top 10 Users by Session Frequency')
        st.pyplot(fig)

        st.subheader("Top 10 Users by Session Duration")
        fig = plot_top_10(df_engagement, 'TotalSessionDuration', xlabel='Customer ID (MSISDN)', ylabel='Session Duration (seconds)', title='Top 10 Users by Session Duration')
        st.pyplot(fig)

        st.subheader("Top 10 Users by Total Traffic")
        fig = plot_top_10(df_engagement, 'TotalTraffic', xlabel='Customer ID (MSISDN)', ylabel='Total Traffic (bytes)', title='Top 10 Users by Total Traffic')
        st.pyplot(fig)

        # Display cluster results and statistics
        st.subheader("Clustering Results")
        st.write("Each customer is categorized into a cluster based on their engagement metrics.")
        st.dataframe(df_clustered)

        st.subheader("Cluster Statistics")
        st.write("See the summarized statistics for each of the clusters formed.")
        st.dataframe(cluster_stats)

        # Display top users per application
        st.subheader("Top 10 Engaged Users per Application")
        st.write("This table shows the top 10 users per application.")
        st.dataframe(top_users_per_app)

        # Visualize the distribution of the engagement metrics
        st.subheader("Engagement Metrics Distribution")
        fig1 = plot_histogram(df_clustered)
        st.pyplot(fig1)

        # Display K-means clustering visualization
        st.subheader("K-Means Clustering Visualization")
        fig2 = plot_kmeans_clusters(df_clustered)
        st.pyplot(fig2)

        # Visualize the top applications based on user engagement
        st.subheader("Top 3 Most Used Applications")
        fig3 = plot_top_applications(top_users_per_app)
        st.pyplot(fig3)

else:
    st.error("No data available to display. Please check the data upload or the data source connection.")