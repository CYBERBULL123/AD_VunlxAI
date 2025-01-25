import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def clean_data(df, drop_na=True, impute_missing=True, handle_outliers=True):
    """
    Clean and preprocess the data.

    Args:
        df (pd.DataFrame): Input DataFrame to clean.
        drop_na (bool): Whether to drop rows with missing values.
        impute_missing (bool): Whether to impute missing values (using mean for numeric, mode for categorical).
        handle_outliers (bool): Whether to handle outliers using IQR or standard deviation.

    Returns:
        pd.DataFrame: Cleaned DataFrame.
    """
    try:
        # Drop rows with missing values if specified
        if drop_na:
            initial_rows = df.shape[0]
            df = df.dropna()
            dropped_rows = initial_rows - df.shape[0]
            logging.info(f"Dropped {dropped_rows} rows with missing values.")

        # Handle outliers for numeric columns
        if handle_outliers:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                initial_rows = df.shape[0]
                for col in numeric_cols:
                    # Use IQR for outlier detection
                    Q1 = df[col].quantile(0.25)
                    Q3 = df[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - (1.5 * IQR)
                    upper_bound = Q3 + (1.5 * IQR)
                    df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
                dropped_rows = initial_rows - df.shape[0]
                logging.info(f"Dropped {dropped_rows} rows due to outliers.")

        # Impute missing values if specified
        if impute_missing:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            categorical_cols = df.select_dtypes(include=['object']).columns

            # Impute numeric columns with mean (if numeric columns exist)
            if len(numeric_cols) > 0:
                numeric_imputer = SimpleImputer(strategy='mean')
                df[numeric_cols] = numeric_imputer.fit_transform(df[numeric_cols])
                logging.info("Imputed missing values in numeric columns using mean.")

            # Impute categorical columns with mode (if categorical columns exist)
            if len(categorical_cols) > 0:
                categorical_imputer = SimpleImputer(strategy='most_frequent')
                df[categorical_cols] = categorical_imputer.fit_transform(df[categorical_cols])
                logging.info("Imputed missing values in categorical columns using mode.")

        # Add synthetic numeric columns if no numeric columns exist
        if len(df.select_dtypes(include=[np.number]).columns) == 0:
            logging.info("No numeric columns found. Adding synthetic numeric columns.")
            df = add_synthetic_numeric_columns(df)

        logging.info("Data cleaning completed successfully.")
        return df

    except Exception as e:
        logging.error(f"Error during data cleaning: {e}")
        raise

def add_synthetic_numeric_columns(df):
    """
    Add synthetic numeric columns to the DataFrame if no numeric columns exist.

    Args:
        df (pd.DataFrame): Input DataFrame.

    Returns:
        pd.DataFrame: DataFrame with synthetic numeric columns.
    """
    # Add a synthetic numeric column (e.g., random values)
    df['synthetic_value'] = np.random.rand(df.shape[0]) * 100  # Random values between 0 and 100

    # Add another synthetic numeric column (e.g., row index)
    df['row_index'] = np.arange(df.shape[0])

    logging.info("Added synthetic numeric columns to the dataset.")
    return df

def preprocess_data(df, numeric_cols=None, categorical_cols=None, encode_categorical=True):
    """
    Preprocess data for ML models.

    Args:
        df (pd.DataFrame): Input DataFrame to preprocess.
        numeric_cols (list): List of numeric columns to preprocess. If None, all numeric columns are used.
        categorical_cols (list): List of categorical columns to encode. If None, all categorical columns are used.
        encode_categorical (bool): Whether to encode categorical columns into numeric values.

    Returns:
        np.ndarray: Preprocessed numeric data.
    """
    try:
        # Select numeric columns
        if numeric_cols is None:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) == 0:
            raise ValueError("No numeric columns found for preprocessing.")

        numeric_data = df[numeric_cols].to_numpy()

        # Scale numeric data
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(numeric_data)

        # Encode categorical columns if specified
        if encode_categorical:
            if categorical_cols is None:
                categorical_cols = df.select_dtypes(include=['object']).columns
            if len(categorical_cols) > 0:
                encoder = OneHotEncoder(handle_unknown='ignore')
                encoded_data = encoder.fit_transform(df[categorical_cols]).toarray()
                scaled_data = np.hstack((scaled_data, encoded_data))
                logging.info("Encoded categorical columns into numeric values.")

        logging.info("Data preprocessing completed successfully.")
        return scaled_data

    except Exception as e:
        logging.error(f"Error during data preprocessing: {e}")
        raise