import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler

def transform_data(dataframe):
    """
    This function transforms the input dataframe by applying the following operations:
    - Scaling numerical columns using StandardScaler
    - Encoding categorical columns using OneHotEncoder
    - Handling missing values
    
    Parameters:
    dataframe (pd.DataFrame): Input dataframe to be transformed
    
    Returns:
    pd.DataFrame: Transformed dataframe
    """
    
    # Separate numerical and categorical columns
    numerical_columns = dataframe.select_dtypes(include=['number']).columns
    categorical_columns = dataframe.select_dtypes(include=['object']).columns
    
    # Scale numerical columns
    scaler = StandardScaler()
    dataframe[numerical_columns] = scaler.fit_transform(dataframe[numerical_columns])
    
    # Encode categorical columns
    encoder = OneHotEncoder(handle_unknown='ignore')
    encoded_data = encoder.fit_transform(dataframe[categorical_columns])
    encoded_columns = encoder.get_feature_names_out(categorical_columns)
    encoded_dataframe = pd.DataFrame(encoded_data.toarray(), columns=encoded_columns)
    
    # Concatenate numerical and encoded categorical columns
    transformed_dataframe = pd.concat([dataframe[numerical_columns], encoded_dataframe], axis=1)
    
    return transformed_dataframe