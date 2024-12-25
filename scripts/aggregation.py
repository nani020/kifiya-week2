import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, pairwise_distances
from scipy.spatial.distance import euclidean

def aggregate_engagement_metrics(dataframe):
    """
    Aggregates engagement metrics (session frequency, session duration, total traffic) per user (MSISDN).
    
    Args:
    - dataframe (DataFrame): The dataframe containing session data with relevant columns such as 
                      ['MSISDN/Number', 'Dur. (ms)', 'Total UL (Bytes)', 'Total DL (Bytes)'].
    
    Returns:
    - DataFrame: A DataFrame containing aggregated metrics per user (MSISDN).
    """
    
    # Aggregate session frequency: Count the number of sessions per user (MSISDN)
    session_frequency = dataframe.groupby('MSISDN/Number')['Dur. (ms)'].count().reset_index(name='Session Frequency')
    
    # Aggregate session duration: Sum the session duration for each user (MSISDN)
    session_duration = dataframe.groupby('MSISDN/Number')['Dur. (ms)'].sum().reset_index(name='Total Session Duration')
    
    # Aggregate total traffic: Sum up both download and upload data for each user (MSISDN)
    total_traffic = dataframe.groupby('MSISDN/Number')[['Total DL (Bytes)', 'Total UL (Bytes)']].sum().reset_index()
    total_traffic['Total Traffic'] = total_traffic['Total DL (Bytes)'] + total_traffic['Total UL (Bytes)']
    
    # Merge all aggregated metrics
    engagement_metrics = session_frequency.merge(session_duration, on='MSISDN/Number', how='inner') \
                                         .merge(total_traffic, on='MSISDN/Number', how='inner') 
    
    return engagement_metrics

def top_10_engagement(dataframe):
    """
    Reports the top 10 users based on session frequency, session duration, and total traffic.
    
    Args:
    - dataframe (DataFrame): The dataframe with aggregated engagement metrics.
    
    Returns:
    - Tuple: Three dataframes - Top 10 users by session frequency, session duration, and total traffic.
    """
    # Get top 10 customers by each engagement metric
    top_10_session_frequency = dataframe.nlargest(10, 'Session Frequency')
    top_10_session_duration = dataframe.nlargest(10, 'Total Session Duration')
    top_10_traffic = dataframe.nlargest(10, 'Total Traffic')
    
    return top_10_session_frequency, top_10_session_duration, top_10_traffic

def normalize_features(dataframe,  features=['Session Frequency', 'Total Session Duration', 'Total Traffic'], normalize_type='minmax'):
    """
    Normalizes the selected features.
    
    Args:
    - dataframe (DataFrame): DataFrame containing the selected features.
    - normalize_type (str): Type of normalization - 'minmax' for Min-Max scaling or 'zscore' for Z-score normalization.
    
    Returns:
    - DataFrame: DataFrame with normalized features.
    """
    
    # Choose the scaler type based on the normalization requested
    if normalize_type == 'minmax':
        scaler = MinMaxScaler()
    elif normalize_type == 'zscore':
        scaler = StandardScaler()
    else:
        raise ValueError("Invalid normalize_type. Use 'minmax' or 'zscore'.")
    
    # Apply the normalization and return the updated DataFrame
    dataframe[features] = scaler.fit_transform(dataframe[features])
    
    return dataframe

def kmeans_clustering(dataframe, features, n_clusters=3):
    """
    Performs K-means clustering on the selected features.

    Args:
    - dataframe (DataFrame): The dataframe containing the features to cluster.
    - features (list): List of column names to use for clustering.
    - n_clusters (int): Number of clusters.

    Returns:
    - DataFrame: DataFrame with a new 'Cluster' column representing cluster assignments.
    """
    
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(dataframe[features])

    # Apply KMeans clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    dataframe['Cluster'] = kmeans.fit_predict(scaled_features)
    
    # Cluster descriptions based on centroids
    cluster_centroids = pd.DataFrame(kmeans.cluster_centers_, columns=features)

    return dataframe, kmeans, cluster_centroids

def cluster_statistics(dataframe, features, aggregations=['min', 'max', 'mean', 'sum']):
    """
    Computes cluster statistics for the specified features.

    Args:
    - dataframe (DataFrame): The DataFrame containing the features to compute statistics for and the 'Cluster' column for grouping.
    - features (list): A list of feature names to compute statistics for.
    - aggregations (list, optional): A list of aggregation types to apply. Defaults to ['min', 'max', 'mean', 'sum'].

    Returns:
    - DataFrame: A DataFrame containing the computed cluster statistics for the specified features.
    """
    # Check if all features exist in the dataframe
    missing_features = [feature for feature in features if feature not in dataframe.columns]
    if missing_features:
        raise KeyError(f"Features not found in the DataFrame: {', '.join(missing_features)}")

    # Create the aggregation dictionary
    aggregation_dict = {feature: aggregations for feature in features}

    # Group by 'Cluster' and aggregate
    result = dataframe.groupby('Cluster').agg(aggregation_dict).reset_index()

    # Flatten the MultiIndex columns
    result.columns = ['Cluster'] + [f"{feature} {agg.title()}" for feature, agg in result.columns[1:]]

    return result
    
def aggregate_application_traffic(dataframe):
    """
    This function aggregates the application traffic for each user.
    
    Parameters:
    dataframe (DataFrame): The DataFrame containing the application traffic data.
    
    Returns:
    DataFrame: The aggregated application traffic data.
    """
    
    application_columns = [
        'Social Media DL (Bytes)', 'Google DL (Bytes)', 'Youtube DL (Bytes)', 'Netflix DL (Bytes)', 'Gaming DL (Bytes)', 'Other DL (Bytes)'
    ]
    
    agg_app_traffic = dataframe[application_columns].fillna(0).sum().sort_values(ascending=False).reset_index()
    agg_app_traffic.columns = ['Application', 'Total Traffic (Bytes)']

    
    return agg_app_traffic

def agg_top_user_per_app(dataframe):
    """
    Aggregates the total traffic per application and derives the top 10 most engaged users.
    
    Args:
    - dataframe (DataFrame): DataFrame containing session data with application traffic columns (e.g., 'Social Media DL (Bytes)', 'Google DL (Bytes)').
    
    Returns:
    - DataFrame: Aggregated traffic for each user per application, and top 10 most engaged users per application.
    """
    
    application_columns = [
        'Social Media DL (Bytes)', 'Google DL (Bytes)', 'Youtube DL (Bytes)', 'Netflix DL (Bytes)', 'Gaming DL (Bytes)', 'Other DL (Bytes)'
    ]
    
    # Calculate total traffic per application for each user
    for app_col in application_columns:
        dataframe[f'{app_col}_TotalTraffic'] = dataframe[app_col].fillna(0)
    
    # Aggregate total traffic per application
    app_traffic_aggregate = dataframe.groupby('MSISDN/Number')[application_columns].sum().reset_index()
    
    # Derive the top 10 most engaged users per application
    top_users_per_app = app_traffic_aggregate[application_columns].apply(lambda x: x.sort_values(ascending=False).head(10))
    
    return top_users_per_app, app_traffic_aggregate

def aggregate_experience_metrics(dataframe):
    """
    Aggregate experience metrics per user.
    
    Args:
    - dataframe (DataFrame): The dataframe containing session data including columns for TCP retransmissions, RTT, throughput, and handset type.
    
    Returns:
    - DataFrame: Aggregated metrics per user.
    """
    # Compute the average TCP retransmission by summing both DL and UL TCP retransmissions and taking the average
    dataframe['Average TCP Retransmission'] = (dataframe['TCP DL Retrans. Vol (Bytes)'] + dataframe['TCP UL Retrans. Vol (Bytes)']) / 2
    
    # Compute the average RTT by summing both DL and UL RTT and taking the average
    dataframe['Average RTT'] = (dataframe['Avg RTT DL (ms)'] + dataframe['Avg RTT UL (ms)']) / 2
    
    # Compute the average throughput by summing both DL and UL throughput and taking the average
    dataframe['Average Throughput'] = (dataframe['Avg Bearer TP DL (kbps)'] + dataframe['Avg Bearer TP UL (kbps)']) / 2
    # For each user (MSISDN/Number), compute the average of the necessary metrics and include additional columns
    aggregated_dataframe = dataframe.groupby('MSISDN/Number').agg(
        Average_TCP_Retransmission=('Average TCP Retransmission', 'mean'),
        Average_RTT=('Average RTT', 'mean'),
        Avg_RTT_DL=('Avg RTT DL (ms)', 'mean'),
        Avg_RTT_UL=('Avg RTT UL (ms)', 'mean'),
        TCP_DL_Retrans_Vol=('TCP DL Retrans. Vol (Bytes)', 'mean'),
        TCP_UL_Retrans_Vol=('TCP UL Retrans. Vol (Bytes)', 'mean'),
        Handset_Type=('Handset Type', 'first'),  # Get the most common handset type
        Average_Throughput=('Average Throughput', 'mean'),
        Avg_Bearer_TP_DL=('Avg Bearer TP DL (kbps)', 'mean'),
        Avg_Bearer_TP_UL=('Avg Bearer TP UL (kbps)', 'mean')
    ).reset_index().rename(columns={
        'Avg_RTT_DL': 'Avg RTT DL (ms)',
        'Avg_RTT_UL': 'Avg RTT UL (ms)',
        'TCP_DL_Retrans_Vol': 'TCP DL Retrans. Vol (Bytes)',
        'TCP_UL_Retrans_Vol': 'TCP UL Retrans. Vol (Bytes)',
        'Average_TCP_Retransmission': 'Average TCP Retransmission',
        'Average_RTT': 'Average RTT',
        'Handset_Type': 'Handset Type',
        'Average_Throughput': 'Average Throughput',
        'Avg_Bearer_TP_DL': 'Avg Bearer TP DL (kbps)',
        'Avg_Bearer_TP_UL': 'Avg Bearer TP UL (kbps)'
    })
    # Fill missing values in categorical columns with 'Unknown'
    aggregated_dataframe['Handset Type'].fillna('Unknown', inplace=True)
    # aggregated_dataframe["Handset Type"].fillna(dataframe["Handset Type"].mode()[0], inplace=True)

    return aggregated_dataframe

def compute_distance_to_centroid(dataframe, features, centroid):
    """
    Compute the Euclidean distance from each data point to the centroid.
    """
    distances = np.sqrt(((dataframe[features] - centroid) ** 2).sum(axis=1))
    return distances

# Compute Engagement and Experience Scores
def compute_scores(df_engagement, df_experience, engagement_centroid, experience_centroid, engagement_features, experience_features):
    df_engagement["Engagement Score"] = compute_distance_to_centroid(df_engagement, engagement_features, engagement_centroid)
    df_experience["Experience Score"] = compute_distance_to_centroid(df_experience, experience_features, experience_centroid)
    
    df_scores = df_engagement[["MSISDN/Number", "Engagement Score"]].merge(
        df_experience[["MSISDN/Number", "Experience Score"]], on="MSISDN/Number"
    )
    
    df_scores["Satisfaction Score"] = df_scores[["Engagement Score", "Experience Score"]].mean(axis=1)
    return df_scores

def find_centroid(centroids, features, method="min"):
    """
    Find the centroid with the lowest or highest sum of feature values.

    Args:
    - centroids (pd.DataFrame or np.ndarray): The centroids to evaluate.
    - features (list): Feature names (if DataFrame) or indices (if NumPy array).
    - method (str): Either "min" (for lowest) or "max" (for highest).

    Returns:
    - centroid: The selected centroid based on the method.
    - cluster: Index/identifier of the selected cluster.
    """
    if isinstance(centroids, pd.DataFrame):
        # Ensure the features are column names
        scores = centroids[features].sum(axis=1)
        cluster = scores.idxmin() if method == "min" else scores.idxmax()
        centroid = centroids.loc[cluster]
    elif isinstance(centroids, np.ndarray):
        # Ensure features are column indices
        if not all(isinstance(f, int) for f in features):
            raise ValueError("For NumPy arrays, 'features' must be a list of integers (column indices).")
        scores = np.sum(centroids[:, features], axis=1)
        idx = np.argmin(scores) if method == "min" else np.argmax(scores)
        centroid = centroids[idx]
        cluster = idx
    else:
        raise TypeError("Centroids must be either a pandas DataFrame or a NumPy array.")
    return centroid, cluster

# Get Top 10 Satisfied Customers
def get_top_customers(df_scores, n=10):
    return df_scores.nlargest(n, "Satisfaction Score")

# Build Regression Model
def train_regression_model(df_scores, feature_columns, target_column, model="linear"):
    X = df_scores[feature_columns]
    y = df_scores[target_column]
    
    if model == "linear":
        reg = LinearRegression()
    elif model == "random_forest":
        reg = RandomForestRegressor()
    else:
        raise ValueError("Unsupported model type")
    
    reg.fit(X, y)
    return reg

# Aggregate Averages by Cluster
def aggregate_cluster_averages(data):
    return data.groupby("Cluster").mean()

def kmeans_clustering(dataframe, features, n_clusters=3, scaled=True):
    """
    Performs K-means clustering on the selected features.

    Args:
    - dataframe (DataFrame): The dataframe containing the features to cluster.
    - features (list): List of column names to use for clustering.
    - n_clusters (int): Number of clusters.

    Returns:
    - DataFrame: DataFrame with a new 'Cluster' column representing cluster assignments.
    """
    selected_features = dataframe[features]
    if scaled:
        scaler = StandardScaler()
        selected_features = scaler.fit_transform(selected_features)

    # Apply KMeans clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    dataframe['Cluster'] = kmeans.fit_predict(selected_features)
    
    # Cluster descriptions based on centroids
    cluster_centroids = pd.DataFrame(kmeans.cluster_centers_, columns=features)

    return dataframe, kmeans, cluster_centroids