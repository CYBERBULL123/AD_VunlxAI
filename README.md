# 🛡️ Advanced AI-Powered Vulnerability Scanner

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.14.0-orange)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0.0-red)
![Nmap](https://img.shields.io/badge/Nmap-7.90-green)
![Docker](https://img.shields.io/badge/Docker-20.10.8-blue)
![License](https://img.shields.io/badge/License-MIT-yellow)

The **Advanced AI-Powered Vulnerability Scanner** is a cutting-edge cybersecurity tool designed to detect vulnerabilities, predict risks, and monitor network intrusions using **machine learning** and **advanced scanning techniques**. This tool is perfect for cybersecurity professionals, network administrators, and developers looking to secure their systems effectively.

---

## ✨ **Features**

### 📂 **Data Analysis**
- Upload and analyze **CSV, Excel, JSON, or log files**.
- Clean and preprocess data for machine learning.
- Detect anomalies using **Isolation Forest**.
- Predict risk scores using a **PyTorch-based neural network**.

### 🌐 **Network Scanning**
- Perform **Nmap scans** (Quick Scan, Service Detection, OS Detection, etc.).
- Visualize open ports, services, and OS detection results.
- Predict **vulnerability likelihood** based on scan results.

### 🚨 **Intrusion Detection**
- Monitor network traffic for **suspicious activity**.
- Detect intrusions using predefined rules (e.g., port scans, SQL injection).

### 📊 **Visualization**
- Interactive charts (bar, pie, scatter, heatmap, histogram).
- Visualize **risk scores**, **anomalies**, and **vulnerabilities**.

### 🤖 **AI-Powered Reporting**
- Generate **intelligent reports** using **Gemini API**.
- Automatically summarize scan results and provide actionable insights.

---

## 🚀 **Getting Started**

### **Installation**

1. Clone the repository:
   ```bash
   git clone https://github.com/CYBERBULL123/AD_VunlxML.git
   cd AD_VunlxML
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the app:
   ```bash
   streamlit run app.py
   ```

---

## 🐳 **Docker Setup**

### **1. Using Docker**

1. Build the Docker image:
   ```bash
   docker-compose build
   ```

2. Start the container:
   ```bash
   docker-compose up
   ```

3. Access the app:
   Open your browser and navigate to:
   ```
   http://localhost:8501
   ```

### **2. Dockerfile Details**
The `Dockerfile` sets up a Python environment, installs dependencies, and runs the Streamlit app.

```Dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

### **3. Docker Compose Details**
The `docker-compose.yml` file simplifies running the application in a container.

```yaml
version: '3.8'
services:
  app:
    build: .
    ports:
      - "8501:8501"
    volumes:
      - .:/app
    environment:
      - PYTHONUNBUFFERED=1
      - GEMINI_API_KEY=${GEMINI_API_KEY}  # Pass the Gemini API key
    stdin_open: true
    tty: true
```

---

## 🔑 **Environment Variables**

To use the **Gemini API** for AI-powered report generation, you need to set up the `GEMINI_API_KEY` environment variable.

### **1. Create a `.env` File**
Create a `.env` file in the root of your project and add your Gemini API key:

```plaintext
GEMINI_API_KEY=your_gemini_api_key_here
```

### **2. Load Environment Variables**
The app uses `python-dotenv` to load the `.env` file. Ensure the following code is present in your `app.py`:

```python
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Access the Gemini API key
gemini_api_key = os.getenv("GEMINI_API_KEY")
```

### **3. Using Docker with `.env`**
When using Docker, the `docker-compose.yml` file automatically loads the `.env` file and passes the `GEMINI_API_KEY` to the container.

---

## 🗂️ **Project Structure**

```
AD_VunlxML/
│
├── app.py                       # Main Streamlit application
├── config.py                    # Configuration settings
├── requirements.txt             # Dependencies
├── README.md                    # Project documentation
├── Dockerfile                   # Docker configuration
├── docker-compose.yml           # Docker Compose configuration
├── .env                         # Environment variables (e.g., Gemini API key)
│
├── data_processing/             # Data preprocessing and parsing
│   ├── __init__.py
│   ├── data_cleaning.py         # Data cleaning functions
│   ├── log_parser.py            # Log file parsing
│
├── models/                      # Machine learning models
│   ├── __init__.py
│   ├── anomaly_detection.py     # Isolation Forest for anomaly detection
│   ├── risk_prediction.py       # PyTorch model for risk prediction
│
├── network_scanning/            # Network scanning functionality
│   ├── __init__.py
│   ├── nmap_scanner.py          # Nmap scanning logic
│
├── visualization/               # Visualization functions
│   ├── __init__.py
│   ├── plotly_charts.py         # Plotly-based visualizations
│
└── ids/                         # Intrusion Detection System
    ├── __init__.py
    ├── intrusion_detection.py   # IDS logic
```

---

## 📚 **Documentation**

### **1. Data Processing**
- **Data Cleaning**:
  - Removes missing values and duplicates.
  - Handles outliers using standard deviation.
- **Log Parsing**:
  - Parses log files into a structured format (timestamp, level, message).

### **2. Machine Learning Models**
- **Anomaly Detection**:
  - Uses Isolation Forest to detect anomalies in numeric data.
- **Risk Prediction**:
  - A PyTorch-based neural network predicts risk scores (0 to 1).

### **3. Network Scanning**
- **Nmap Integration**:
  - Supports multiple scan types (Quick Scan, Full Scan, etc.).
  - Visualizes open ports, services, and OS detection results.
- **Vulnerability Prediction**:
  - Predicts vulnerability likelihood based on Nmap scan results.

### **4. Intrusion Detection System (IDS)**
- **Rule-Based Detection**:
  - Detects intrusions using predefined rules (e.g., port scans, SQL injection).

### **5. Visualization**
- **Interactive Charts**:
  - Bar, pie, scatter, heatmap, and histogram charts for data visualization.

### **6. AI-Powered Reporting**
- **Gemini API Integration**:
  - Generates intelligent reports summarizing scan results.
  - Provides actionable insights and recommendations.

---

## 🤝 **Contributing**

We welcome contributions! Please follow these steps:

1. Fork the repository.
2. Create a new branch:
   ```bash
   git checkout -b feature/your-feature-name
   ```
3. Commit your changes:
   ```bash
   git commit -m "Add your feature"
   ```
4. Push to the branch:
   ```bash
   git push origin feature/your-feature-name
   ```
5. Open a pull request.

---

## 📜 **License**

This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for details.

---

## ❓ **Support**

For questions or issues, please open an issue on the [GitHub repository](https://github.com/CYBERBULL123/AD_VunlxML/issues).

---

## 🙏 **Acknowledgments**

- **Streamlit** for the interactive web framework.
- **PyTorch** for the machine learning model.
- **Nmap** for network scanning.
- **Plotly** for data visualization.
- **Docker** for containerization.
- **Gemini API** for AI-powered reporting.

---

## 📄 **Changelog**

### **Version 2.0 (Latest)**
- Added **LangChain** and **Gemini** integration for intelligent reporting.
- Improved UI with **modern CSS design** and animations.
- Enhanced **intrusion detection** with dynamic threat intelligence.
- Added **automated vulnerability reporting**.
- Optimized performance for large datasets.
- Added **Docker support** for easy deployment.

### **Version 1.0**
- Initial release with basic data analysis, network scanning, and intrusion detection features.

---

## 📧 **Contact**

For inquiries, feel free to reach out:

- **Email**: [Aditya Pandey](mailto:opaadi98@gmail.com)
- **GitHub**: [CYBERBULL123](https://github.com/CYBERBULL123)