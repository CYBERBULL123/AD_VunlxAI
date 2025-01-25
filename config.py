# Data Processing
CLEAN_DATA = True
DROP_NA = True  # Ensure this is defined

# Model Training
INPUT_SIZE = None  # Adjust based on your data
LEARNING_RATE = 0.001
EPOCHS = 200
PATIENCE = 10  # Early stopping patience

# Model Saving
MODEL_SAVE_PATH = "models/risk_prediction_model.pth"  # Path to save the trained model

# Anomaly Detection
CONTAMINATION = 0.1  # Proportion of outliers in the data

# Nmap Scanning
SCAN_TYPES = {
    "Quick Scan": "-T4 -F",  # Fast scan for common ports
    "Service Detection": "-sV --version-intensity 9",  # Detect service versions
    "OS Detection": "-O",  # OS fingerprinting
    "Aggressive Scan": "-A",  # Aggressive scan (OS, version, script, traceroute)
    "Full Scan": "-p- -T4 -A -v",  # Scan all ports with aggressive options
}