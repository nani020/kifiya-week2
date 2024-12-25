import numpy as np

def impute_missing_values(dataframe, impute_strategies):
    """
    Impute missing values for specified columns based on strategies.
    
    Args:
    - dataframe (pd.DataFrame): Input dataframe.
    - impute_strategies (dict): Imputation strategies per column ('mean', 'mode').
    
    Returns:
    - pd.DataFrame: Updated dataframe with imputed values.
    """
    for col, strategy in impute_strategies.items():
        if col in dataframe.columns:
            if strategy == 'mean':
                dataframe[col] = dataframe[col].fillna(dataframe[col].mean())
            elif strategy == 'mode':
                dataframe[col] = dataframe[col].fillna(dataframe[col].mode()[0])
    return dataframe

def handle_outliers(dataframe, method="IQR", threshold=1.5, target_columns=None):
    """
    Handle outliers in numerical columns using IQR or trimming.

    Args:
    - dataframe (pd.DataFrame): Input dataframe.
    - method (str): Outlier treatment method ('IQR' or 'std_dev').
    - threshold (float): Threshold multiplier for outlier detection (default 1.5 for IQR).
    - target_columns (list or None): Specific columns to treat. Defaults to all numerical columns.

    Returns:
    - pd.DataFrame: Dataframe with outliers handled.
    """
    if target_columns is None:
        target_columns = dataframe.select_dtypes(include=['number']).columns

    if method == "IQR":
        Q1 = dataframe[target_columns].quantile(0.25)
        Q3 = dataframe[target_columns].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        # dataframe = dataframe[~((dataframe[numerical_columns] < lower_bound) | (dataframe[numerical_columns] > upper_bound)).any(axis=1)]

        for col in target_columns:
            outliers = (dataframe[col] < lower_bound[col]) | (dataframe[col] > upper_bound[col])
            dataframe.loc[outliers, col] = np.nan  # Replace outliers with NaN
            dataframe[col] = dataframe[col].fillna(dataframe[col].mean())  # Impute NaNs

    elif method == "std_dev":
        for col in target_columns:
            if dataframe[col].dtype.kind in 'iufc':  # Numeric types
                mean = dataframe[col].mean()
                std_dev = dataframe[col].std()
                upper_limit = mean + threshold * std_dev
                lower_limit = mean - threshold * std_dev
                dataframe[col] = np.where(
                    dataframe[col] > upper_limit, mean,
                    np.where(dataframe[col] < lower_limit, mean, dataframe[col])
                )
    return dataframe

def clean_dataframe(dataframe, impute_strategies=None, cat_defaults=None, outlier_method="IQR", outlier_threshold=1.5):
    """
    Cleans the dataframe by imputing missing values, treating outliers, and removing duplicates.

    Args:
    - dataframe (pd.DataFrame): Input dataframe.
    - impute_strategies (dict or None): Columns with imputation strategies ('mean', 'mode'). If None, uses auto-detection for numerical columns.
    - cat_defaults (dict or None): Default values for categorical columns. If None, no action is performed.
    - outlier_method (str): Method to handle outliers ('IQR' or 'std_dev').
    - outlier_threshold (float): Threshold for outlier treatment (e.g., 1.5 for IQR).

    Returns:
    - pd.DataFrame: Cleaned dataframe.
    """
    print("Before cleaning:")
    print(dataframe.info())
    print(dataframe.describe())

    # Drop duplicates
    dataframe.drop_duplicates(inplace=True)

    # Impute categorical columns
    if cat_defaults:
        dataframe.fillna(value=cat_defaults, inplace=True)

    # Impute missing values for numerical columns
    # dataframe.ffill(inplace=True)
    # dataframe.bfill(inplace=True)
    if impute_strategies is None:
        impute_strategies = {col: 'mean' for col in dataframe.select_dtypes(include=['number']).columns}
    dataframe = impute_missing_values(dataframe, impute_strategies)

    # Treat outliers
    dataframe = handle_outliers(dataframe, method=outlier_method, threshold=outlier_threshold)

    print("After cleaning:")
    print(dataframe.info())
    print(dataframe.describe())

    print("Data cleaning completed.")
    return dataframe