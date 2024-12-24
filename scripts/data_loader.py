# data_loader.py
import pandas as pd

def load_data(file_path):
    return pd.read_csv( file_path , parse_dates=["Date"])

def preprocess_data(df):
    df['publication_date'] = pd.to_datetime(df['publication_date'], errors='coerce')
    df.dropna(subset=['headline', 'publisher'], inplace=True)
    return df
