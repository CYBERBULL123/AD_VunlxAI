import re
from datetime import datetime
import pandas as pd
import numpy as np
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def parse_log_file(log_content):
    """
    Parse log files into a structured format with numeric and categorical columns.
    If no numeric columns are found, synthetic numeric columns are added.

    Args:
        log_content (str): Raw log content as a string.

    Returns:
        pd.DataFrame: Parsed log data as a DataFrame.
    """
    logs = []
    log_patterns = [
        # Common log format: [timestamp] [level] message
        re.compile(r'\[(?P<timestamp>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\] \[(?P<level>\w+)\] (?P<message>.*)'),
        # Apache log format: IP - - [timestamp] "request" status bytes
        re.compile(r'(?P<ip>\d+\.\d+\.\d+\.\d+) - - \[(?P<timestamp>\d{2}/\w+/\d{4}:\d{2}:\d{2}:\d{2} \+\d{4})\] "(?P<request>.*?)" (?P<status>\d+) (?P<bytes>\d+)'),
        # Custom log format: timestamp level message
        re.compile(r'(?P<timestamp>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) (?P<level>\w+) (?P<message>.*)'),
        # JSON log format: {"timestamp": "...", "level": "...", "message": "..."}
        re.compile(r'\{"timestamp": "(?P<timestamp>.*?)", "level": "(?P<level>.*?)", "message": "(?P<message>.*?)"\}')
    ]

    for line in log_content.splitlines():
        for pattern in log_patterns:
            match = pattern.match(line)
            if match:
                log_entry = match.groupdict()
                # Convert timestamp to datetime object
                if 'timestamp' in log_entry:
                    try:
                        log_entry['timestamp'] = datetime.strptime(log_entry['timestamp'], "%Y-%m-%d %H:%M:%S")
                    except ValueError:
                        try:
                            log_entry['timestamp'] = datetime.strptime(log_entry['timestamp'], "%d/%b/%Y:%H:%M:%S %z")
                        except ValueError:
                            logging.warning(f"Unsupported timestamp format: {log_entry['timestamp']}")
                            log_entry['timestamp'] = None
                logs.append(log_entry)
                break
        else:
            logging.warning(f"Skipping line (no matching pattern): {line}")

    # Convert logs to DataFrame
    df = pd.DataFrame(logs)

    # Clean log data
    df = clean_log_data(df)

    # Add synthetic numeric columns if no numeric columns exist
    if len(df.select_dtypes(include=[np.number]).columns) == 0:
        logging.info("No numeric columns found. Adding synthetic numeric columns.")
        df = add_synthetic_numeric_columns(df)

    logging.info("Log parsing completed successfully.")
    return df

def clean_log_data(df):
    """
    Clean and preprocess log data to ensure it contains meaningful numeric and categorical columns.

    Args:
        df (pd.DataFrame): Raw log data as a DataFrame.

    Returns:
        pd.DataFrame: Cleaned log data.
    """
    try:
        # Drop rows with missing timestamps
        if 'timestamp' in df.columns:
            initial_rows = df.shape[0]
            df = df.dropna(subset=['timestamp'])
            dropped_rows = initial_rows - df.shape[0]
            logging.info(f"Dropped {dropped_rows} rows with missing timestamps.")

        # Convert timestamp to numeric (Unix timestamp)
        if 'timestamp' in df.columns:
            df['timestamp'] = df['timestamp'].apply(lambda x: x.timestamp() if x else None)

        # Extract numeric columns from messages
        if 'message' in df.columns:
            # Extract status codes (e.g., 200, 404) from messages
            status_code_pattern = re.compile(r'\b(\d{3})\b')
            df['status_code'] = df['message'].apply(lambda x: int(status_code_pattern.search(x).group(1)) if status_code_pattern.search(x) else None)

            # Extract response sizes (e.g., bytes) from messages
            bytes_pattern = re.compile(r'\b(\d+)\s+bytes\b')
            df['bytes'] = df['message'].apply(lambda x: int(bytes_pattern.search(x).group(1)) if bytes_pattern.search(x) else None)

        # Standardize log levels (optional)
        if 'level' in df.columns:
            df['level'] = df['level'].str.upper()

        logging.info("Log data cleaning completed successfully.")
        return df

    except Exception as e:
        logging.error(f"Error during log data cleaning: {e}")
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