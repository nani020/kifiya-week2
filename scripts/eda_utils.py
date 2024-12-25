import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def calculate_user_statistics(dataframe, imsi_col):
    
    unique_users_count = dataframe[imsi_col].nunique()
    total_records = dataframe.shape[0]
    avg_records_per_user = total_records / unique_users_count
    
    return {
        'Total Records': total_records, 
        'Unique Users Count': unique_users_count, 
        'Average Records Per User': avg_records_per_user
    }

def segment_users(dataframe, column, num_deciles):
    """
    Segment users into deciles based on a given column.

    Parameters:
    dataframe (pd.DataFrame): Input dataframe
    column (str): Column to segment users by
    num_deciles (int): Number of deciles to segment users into

    Returns:
    pd.DataFrame: Dataframe with added decile column
    """
    decile_column = f'{column} Decile'
    dataframe[decile_column] = pd.qcut(dataframe[column], num_deciles, labels=False)
    return dataframe

def compute_decile_summary(dataframe, decile_column, agg_columns, agg_functions):
    """
    Compute total data volume (DL + UL) per decile.

    Parameters:
    dataframe (pd.DataFrame): Input dataframe
    decile_column (str): Column to group by
    agg_columns (list): List of column names to aggregate
    agg_functions (list): List of aggregation functions

    Returns:
    pd.DataFrame: Dataframe with aggregated values
    """

    # Automatically build the aggregation dictionary
    agg_dict = {col: func for col, func in zip(agg_columns, agg_functions)}
    
    # Perform the groupby + agg
    return dataframe.groupby(decile_column).agg(agg_dict).reset_index()

def compute_dispersion_measures(dataframe, columns, functions):
    """
    Compute dispersion measures for a given dataframe.

    Parameters:
    dataframe (pd.DataFrame): Input dataframe
    columns (list): List of column names to compute dispersion for
    functions (list): List of dispersion functions

    Returns:
    pd.DataFrame: Dataframe with dispersion measures
    """
    return dataframe[columns].agg(functions)

def melt_dataframe(dataframe, app_columns):
    # Melt the application columns into rows

    return dataframe.melt(
        id_vars=['Bearer Id', 'Dur. (ms)', 'Total Data (Bytes)','Total DL (Bytes)','Total UL (Bytes)'],
        value_vars=app_columns,
        var_name='App Name',
        value_name='Data Volume'
    )

def get_aggregate_dataframe(dataframe):
    # Aggregate per user and application
    
    return dataframe.groupby(['Bearer Id', 'App Name']).agg(
        Number_Of_Sessions=('Dur. (ms)', 'count'),
        Total_Session_Duration=('Dur. (ms)', 'sum'),
        Total_Download_Data=('Total DL (Bytes)', 'sum'),
        Total_Upload_Data=('Total UL (Bytes)', 'sum'),
        Total_Data_Volume=('Data Volume', 'sum')
    ).reset_index().rename(columns=lambda x: x.replace(' ', '_').title())


def get_dispersion_summary(dataframe):
    # Compute dispersion parameters for each quantitative variable

    numeric_dataframe = dataframe.select_dtypes(include='number')

    return pd.DataFrame({
        'Mean': numeric_dataframe.mean(),
        'Median': numeric_dataframe.median(),
        'Range': numeric_dataframe.max() - numeric_dataframe.min(),
        'Variance': numeric_dataframe.var(),
        'Std Dev': numeric_dataframe.std(),
        'IQR': numeric_dataframe.quantile(0.75) - numeric_dataframe.quantile(0.25)
    })

def apply_pca(dataframe, columns, n_components):
    """
    Apply PCA to a given dataframe.

    Parameters:
    dataframe (pd.DataFrame): Input dataframe
    columns (list): List of column names to apply PCA to
    n_components (int): Number of components for PCA

    Returns:
    pd.DataFrame: Dataframe with PCA results
    """

    pca = PCA(n_components=n_components)
    scaler = StandardScaler()
    X = dataframe[columns]
    X_scaled = scaler.fit_transform(X)

    principal_components = pca.fit_transform(X_scaled) # Apply PCA
    pca_dataframe = pd.DataFrame(data=principal_components, columns=[f'PC{i+1}' for i in range(n_components)])
    
    return pca_dataframe, pca.explained_variance_ratio_

def calculate_app_usage(dataframe):
    google_usage = dataframe[["Google UL (Bytes)", "Google DL (Bytes)"]].sum()
    youtube_usage = dataframe[["Youtube UL (Bytes)", "Youtube DL (Bytes)"]].sum()
    email_usage = dataframe[["Email UL (Bytes)", "Email DL (Bytes)"]].sum()
    netflix_usage = dataframe[["Netflix UL (Bytes)", "Netflix DL (Bytes)"]].sum()
    gaming_usage = dataframe[["Gaming UL (Bytes)", "Gaming DL (Bytes)"]].sum()

    app_data = pd.DataFrame({
        'App': ['Google', 'Youtube', 'Netflix', 'Email', 'Gaming'],
        'Upload Usage': [google_usage["Google UL (Bytes)"], youtube_usage["Youtube UL (Bytes)"], 
                         netflix_usage["Netflix UL (Bytes)"], email_usage["Email UL (Bytes)"], 
                         gaming_usage["Gaming UL (Bytes)"]],
        'Download Usage': [google_usage["Google DL (Bytes)"], youtube_usage["Youtube DL (Bytes)"], 
                           netflix_usage["Netflix DL (Bytes)"], email_usage["Email DL (Bytes)"], 
                           gaming_usage["Gaming DL (Bytes)"]]
    }).set_index("App")

    return app_data

def calculate_top_locations(dataframe, n=10):
    location_usage = (
        dataframe.groupby("Last Location Name")[["Total DL (Bytes)", "Total UL (Bytes)"]]
        .sum()
        .sort_values("Total DL (Bytes)", ascending=False)
        .head(n)
    )
    return location_usage

def calculate_user_stats(dataframe):
    unique_users_count = dataframe["IMSI"].nunique()
    total_records = dataframe.shape[0]
    avg_records_per_user = total_records / unique_users_count
    return {
        "Unique Users Count": unique_users_count,
        "Total Records": total_records,
        "Average Records Per User": avg_records_per_user
    }

def identify_outliers(data, column, threshold=1.5):

    q1 = data[column].quantile(0.25)
    q3 = data[column].quantile(0.75)
    iqr = q3 - q1
    return data[
        (data[column] < q1 - threshold * iqr) | (data[column] > q3 + threshold * iqr)
    ]

def group_and_compute_engagement(dataframe):
    """Modularized function to group and compute engagement metrics"""
    
    dataframe['Total Users'] = 1 # Add a column to count each user in a group
    
    # Group and compute required columns
    return dataframe.groupby('Total Session Duration Decile').agg(
        Total_Users=('Total Users', 'count'),
        Avg_Data_Volume=('Total Traffic', 'mean'),
        Median_Data_Volume=('Total Traffic', 'median')
    ).reset_index().rename(columns={'Total Session Duration Decile': 'Total Session Duration Decile', 
                                    'Total_Users': 'Total Users', 'Avg_Data_Volume': 'Avg Data Volume', 
                                    'Median_Data_Volume': 'Median Data Volume'})

def compute_top_bottom_frequent(dataframe, column_name, top_n=10):
    """
    Computes and lists the top, bottom, and most frequent values for a specified column.
    
    Args:
    - dataframe (DataFrame): The dataframe containing the data.
    - column_name (str): The column to analyze.
    - top_n (int): Number of top/bottom and most frequent values to list.
    
    Returns:
    - dict: Top, bottom, and most frequent values.
    """
    top_values = dataframe[column_name].nlargest(top_n)
    bottom_values = dataframe[column_name].nsmallest(top_n)
    most_frequent = dataframe[column_name].mode().head(top_n)  # Mode values
    
    return {
        "Top": top_values,
        "Bottom": bottom_values,
        "Most Frequent": most_frequent
    }