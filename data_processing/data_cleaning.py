# data_processing/data_cleaning.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import logging
from config import DROP_NA

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def clean_data(df, drop_na=DROP_NA, numeric_threshold=3, drop_duplicates=True):
    """
    Clean and preprocess the data.

    Args:
        df (pd.DataFrame): Input DataFrame to clean.
        drop_na (bool): Whether to drop rows with missing values.
        numeric_threshold (float): Threshold for outlier removal (in standard deviations).
        drop_duplicates (bool): Whether to drop duplicate rows.

    Returns:
        pd.DataFrame: Cleaned DataFrame.
    """
    try:
        # Drop rows with missing values if specified
        if drop_na:
            initial_rows = df.shape[0]
            df.dropna(inplace=True)
            dropped_rows = initial_rows - df.shape[0]
            logging.info(f"Dropped {dropped_rows} rows with missing values.")

        # Drop duplicate rows if specified
        if drop_duplicates:
            initial_rows = df.shape[0]
            df.drop_duplicates(inplace=True)
            dropped_rows = initial_rows - df.shape[0]
            logging.info(f"Dropped {dropped_rows} duplicate rows.")

        # Remove outliers for numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            initial_rows = df.shape[0]
            for col in numeric_cols:
                col_mean = df[col].mean()
                col_std = df[col].std()
                if col_std > 0:  # Avoid division by zero
                    df = df[np.abs(df[col] - col_mean) <= (numeric_threshold * col_std)]
            dropped_rows = initial_rows - df.shape[0]
            logging.info(f"Dropped {dropped_rows} rows due to outliers.")

        logging.info("Data cleaning completed successfully.")
        return df

    except Exception as e:
        logging.error(f"Error during data cleaning: {e}")
        raise

def preprocess_data(df, numeric_cols=None):
    """
    Preprocess numeric data for ML models.

    Args:
        df (pd.DataFrame): Input DataFrame to preprocess.
        numeric_cols (list): List of numeric columns to preprocess. If None, all numeric columns are used.

    Returns:
        np.ndarray: Preprocessed numeric data.
    """
    try:
        # Select numeric columns
        if numeric_cols is None:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
        numeric_data = df[numeric_cols].to_numpy()

        # Scale numeric data
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(numeric_data)

        logging.info("Data preprocessing completed successfully.")
        return scaled_data

    except Exception as e:
        logging.error(f"Error during data preprocessing: {e}")
        raise