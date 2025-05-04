# 🌪️ **Nepal Climate Sentinel**  
### *Extreme Weather Analytics & Forecasting Platform*

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/) [![Streamlit App](https://img.shields.io/badge/Streamlit-Live-brightgreen.svg)](https://omdena-nic-nepal-capstone-project-kpphelu-app-2xrh3e.streamlit.app/) [![GitHub Repo](https://img.shields.io/badge/GitHub-Repo-black?logo=github)](https://github.com/Omdena-NIC-Nepal/capstone-project-KPPhelu)

---

> 🔮 *An interactive Streamlit platform delivering deep geospatial insights, historic climate analysis, and machine learning-powered extreme weather forecasts for Nepal.*

---

## 📌 Table of Contents

1. [🌐 About the Project](#-about-the-project)  
2. [🚀 Key Features](#-key-features)  
3. [📁 Directory Layout](#-directory-layout)  
4. [⚙️ Setup Instructions](#-setup-instructions)  
5. [▶️ Running the App](#️-running-the-app)  
6. [🌍 Deployment Guide](#-deployment-guide)  
7. [🤝 Getting Involved](#-getting-involved)  
8. [📜 License & Credits](#-license--credits)

---

## 🌐 About the Project

**Nepal Climate Sentinel** is a geospatial weather intelligence platform designed to:

- 🧭 **Explore** district-wise daily metrics: temperature, rainfall, wind speed  
- 🗺️ **Map** weather trends across Nepal via interactive choropleth layers  
- 🧾 **Label** historical extreme events using custom thresholds  
- 🔮 **Forecast** future weather and predict high-risk extreme events with ML models

> Powered by **Python**, **Streamlit**, **GeoPandas**, and **scikit-learn** — the app fuses data science with climate awareness for smarter environmental decision-making.

🎯 **[Live App Demo: Powered on Streamlit](https://sureshsubedi.streamlit.app/)**  
💻 **[GitHub Repository](https://github.com/Omdena-NIC-Nepal/capstone-project-su-resh/)**

---

## 🚀 Key Features

✨ **Visual Dashboards**  
- Real-time exploration of weather metrics via interactive charts & maps  
- District-wise time-series comparisons  

⚡ **Extreme Event Detection**  
- Tag & visualize historic anomalies like floods, heatwaves, and storms  
- Define your own thresholds for event labeling  

🧠 **Smart ML Workflows**  
- Support for regression, classification, and anomaly detection models  
- Dynamic test/train split with cross-validation support  

🛠️ **Rapid Feature Engineering**  
- Automatic derivation of model-ready inputs from raw daily data  

🔔 **Live Forecast & Risk Alerts**  
- Predict next-day weather conditions  
- Highlight high-risk districts based on custom probability thresholds  

🎨 **Customizable Visuals**  
- Geo-based overlays and analytical plots for powerful storytelling  

---

## 📁 Directory Layout

```bash
Nepal-Climate-Sentinel/
├── app.py                       # Streamlit entry point
├── data/
│   ├── Shape_Data_district/    # District-level shapefiles
│   └── dailyclimate-2.csv      # Raw weather data
├── models/                     # Trained model artifacts
├── utils/
│   ├── data_utils.py           # Data loaders
│   ├── preprocessing.py        # Clean + Impute logic
│   ├── visualizations.py       # Plotting helpers
│   ├── feature_engineering.py  # Feature creation
│   ├── label_generation.py     # Tagging logic
│   ├── models.py               # ML definitions
│   └── prediction.py           # Forecast pipeline
├── pages_streamlit/            # Modular Streamlit pages
│   ├── home_pg.py
│   ├── data_exploration_pg.py
│   ├── eda_with_climate_event_pg.py
│   ├── feature_engineering_pg.py
│   ├── model_training_pg.py
│   ├── prediction_pg.py
│   └── about_pg.py
├── requirements.txt            # Python dependencies
├── README.md                   # This file
└── .gitignore
```

---

## ⚙️ Setup Instructions

1. **Clone the Repository**
```bash
git clone https://github.com/Omdena-NIC-Nepal/capstone-project-su-resh
cd capstone-project-su-resh
```

2. **Create & Activate Virtual Environment**
```bash
python3 -m venv venv
source venv/bin/activate      # macOS/Linux
venv\Scripts\activate         # Windows
```

3. **Install Dependencies**
```bash
pip install -r requirements.txt
```

---

## ▶️ Running the App Locally

```bash
streamlit run app.py
```

🧭 Use the sidebar to navigate through:

- 📊 Data exploration  
- 🔍 EDA with labeled events  
- ⚙️ Feature engineering  
- 🧠 Model training  
- 🔮 Forecasting & alerts  

---

## 🌍 Deployment Guide (Streamlit Cloud)

1. Commit your code to GitHub.
2. Sign in to [Streamlit Community Cloud](https://streamlit.io/cloud).
3. Click **"New App"**, choose:
   - Repository: your GitHub repo  
   - Branch: `main`  
   - File: `app.py`
4. Click **"Deploy"** 🎉

🔗 Your app will be live at:

```
https://share.streamlit.io/<USERNAME>/<REPO_NAME>/main/app.py
```

---

## 🤝 Getting Involved

We welcome all contributors—data scientists, geographers, and climate enthusiasts!

📄 Read our [CONTRIBUTING.md](CONTRIBUTING.md) and [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md) before submitting your PRs.

---

## 📜 License & Credits

- 📄 **MIT License** – see [LICENSE](LICENSE)
- 💡 Created & Maintained by **Suresh Subedi** with Omdena NIC Nepal ✨  
- 📢 Data courtesy of public weather records and local shapefiles.

---