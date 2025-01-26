# 🛡️ AD_VunlxAI (Advanced Vulnerability AI)

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.14.0-orange)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0.0-red)
![Nmap](https://img.shields.io/badge/Nmap-7.90-green)
![Docker](https://img.shields.io/badge/Docker-20.10.8-blue)
![License](https://img.shields.io/badge/License-GPL-yellow)

The **AD_VunlxAI** is a cutting-edge cybersecurity tool designed to detect vulnerabilities, predict risks, and monitor network intrusions using **machine learning** and **advanced scanning techniques**. This tool is perfect for cybersecurity professionals, network administrators, and developers looking to secure their systems effectively.

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

## ✨ **Features with Visuals**

### 📂 **Data Analysis**
Here are some screenshots of the **Data Analysis** feature in action:

| **Data Cleaning** | **Anomaly Detection** | **Risk Prediction** |
|-------------------|-----------------------|---------------------|
| ![Data Cleaning](/imgs/dataAnalysis/1.png) | ![Anomaly Detection](/imgs/dataAnalysis/6.png) | ![Risk Prediction](/imgs/dataAnalysis/4.png) |

| **Interactive Charts** | **Log Parsing** | **Data Preprocessing** |
|------------------------|-----------------|------------------------|
| ![Interactive Charts](/imgs/dataAnalysis/8.png) | ![Interactive chart 2](/imgs/dataAnalysis/5.png) | ![Data Preprocessing](/imgs/dataAnalysis/3.png) |

---

### 🌐 **Network Scanning**
Here are some screenshots of the **Network Scanning** feature:

| **Quick Scan** | **Service Detection** | **OS Detection** |
|----------------|-----------------------|------------------|
| ![Quick Scan](/imgs/nmapscan/1.png) | ![Service Detection](/imgs/nmapscan/2.png) | ![OS Detection](/imgs/nmapscan/3.png) |

| **Vulnerability Prediction** | **Scan Results** | **Port Visualization** |
|------------------------------|------------------|------------------------|
| ![Vulnerability Prediction](/imgs/nmapscan/4.png) | ![Scan Results](/imgs/nmapscan/5.png) | ![Port Visualization](/imgs/nmapscan/6.png) |

---

### 📊 **AI-Powered Reporting**
Here are some screenshots of the **AI-Powered Reporting** feature:

| **Report Generation** | **Actionable Insights** | **Summary** |
|-----------------------|-------------------------|-------------|
| ![Report Generation](/imgs/report/1.png) | ![Actionable Insights](/imgs/report/2.png) | ![Summary](/imgs/report/3.png) |

| **Detailed Analysis** |
|-----------------------|
| ![Detailed Analysis](/imgs/report/4.png) |

## 🚀 **Getting Started**

### **Installation**

1. Clone the repository:
   ```bash
   git clone https://github.com/CYBERBULL123/AD_VunlxAI.git
   cd AD_VunlxAI
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

To use the **Gemini API** for AI-powered report generation, you need to set up the `GEMINI_API_KEY` environment variable. This can be done using **Streamlit Secrets** for both **local development** and **Streamlit Cloud deployment**.

---

### **1. Setting Up Streamlit Secrets**

#### **For Streamlit Cloud Deployment**
1. Go to the **Settings** tab in your Streamlit Cloud app.
2. Scroll down to the **Secrets** section.
3. Add your Gemini API key in the following format:
   ```toml
   [api_keys]
   gemini = "your_gemini_api_key_here"
   ```

#### **For Local Development**
1. Create a `.streamlit/secrets.toml` file in your project root directory.
2. Add your Gemini API key in the following format:
   ```toml
   [api_keys]
   gemini = "your_gemini_api_key_here"
   ```

---

### **2. Using Docker with Streamlit Secrets**

When deploying your app using **Docker**, you can pass the `GEMINI_API_KEY` as an environment variable to the container. Update your `docker-compose.yml` file as follows:

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

## 🗂️ **Project Structure**

```
AD_VunlxAI/
│
├── app.py                       # Main Streamlit application
├── config.py                    # Configuration settings
├── requirements.txt             # Python dependencies
├── packages.txt                 # System dependencies for Streamlit Cloud
├── Dockerfile                   # Docker configuration
├── docker-compose.yml           # Docker Compose configuration
├── LICENSE                      # License file
├── README.md                    # Project documentation
├── style.css                    # Custom CSS for Streamlit UI
│
├── .streamlit/                  # Streamlit configuration files
│   ├── config.toml              # Streamlit app configuration
│   └── secrets.toml             # Streamlit secrets (for local development)
│
├── dataset/                     # Folder containing datasets
│   ├── vulnerability_dataset.csv               # Sample dataset
│   └── vulnerability_dataset_with_anomalies.csv # Dataset with anomalies
│
├── modules/                     # All Python modules
│   ├── data_processing/         # Data preprocessing and parsing
│   │   ├── data_cleaning.py     # Data cleaning functions
│   │   └── log_parser.py        # Log file parsing
│   │
│   ├── models/                  # Machine learning models
│   │   ├── anomaly_detection.py # Isolation Forest for anomaly detection
│   │   └── risk_prediction.py   # PyTorch model for risk prediction
│   │
│   ├── network_scanning/        # Network scanning functionality
│   │   └── nmap_scanner.py      # Nmap scanning logic
│   │
│   ├── visualization/           # Visualization functions
│   │   └── plotly_charts.py     # Plotly-based visualizations
│   │
│   ├── ids/                     # Intrusion Detection System
│   │   └── intrusion_detection.py # IDS logic
│   │
│   └── reporting/               # Reporting and threat intelligence
│       ├── report_generator.py  # Generate AI-powered reports
│       └── threat_intelligence.py # Threat intelligence integration
│
├── images/                      # Folder containing all images
│   ├── data_analysis/           # Data analysis screenshots
│   ├── network_scanning/        # Network scanning screenshots
│   └── reporting/               # Reporting screenshots
│
└── .dockerignore                # Files to ignore in Docker builds
```

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

This project is licensed under the **GPL License**. See the [LICENSE](LICENSE) file for details.

---

## ❓ **Support**

For questions or issues, please open an issue on the [GitHub repository](https://github.com/CYBERBULL123/AD_VunlxAI/issues).

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