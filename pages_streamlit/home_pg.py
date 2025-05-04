import streamlit as st

def show():
    st.title("Home Page")

    st.markdown("""
# 🌦️ Extreme Weather Event Forecasting in Nepal

Welcome to the **Extreme Weather Prediction App**, a powerful tool designed to assist with forecasting and analyzing climate-driven extreme weather events in Nepal.

### 🔍 What this app does:
- 🔮 **Predict climate variables** (temperature, precipitation, humidity) for the upcoming days
- ⚠️ **Detect potential extreme weather events**, such as:
  - ColdWave
  - HighTemp
  - Heatwave
  - HeavyRain
- 📊 Visualize historical and forecasted data with district-wise breakdowns

### 👇 Get started
Use the navigation on the left **in sequence** from top to bottom:
- **Data Exploration** - Explore the climate data and geographical boundary of districts of Nepal along with visualization
- **EDA with climate events** - Explore the climate data with Weather events added
- **Feature Engineering** - Feature Engineering to the dataset and its visualization
- **Model Training and Evaluation** - ML model training and its performance evaluation visualization
- **Prediction** - Make prediction with the ML model and its visualization
                
**Note:** This app is for Caption Project purpose.
                """)

#     st.markdown("""
# Welcome to the **Extreme Weather Event Detection and Prediction App** developed for Nepal.

# This application is a part of the Omdena-NIC-Nepal collaborative AI project that focuses on analyzing and predicting extreme weather conditions using historical climate data from 93 weather stations spanning 62 districts in Nepal and machine learning models.

# ### 🚀 Navigation
# Use the sidebar to explore the following sections:

# - 📊 **Data Exploration**  
# - 🧪 **Model Training and Evaluation**  
# - 🔮 **Extreme Weather Prediction**  
# - ℹ️ **About the Project**
# """)