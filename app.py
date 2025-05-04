import streamlit as st
from pathlib import Path
import sys
import logging

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO
)
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).parent
for sub in (BASE_DIR / "utils", BASE_DIR / "pages_streamlit"):
    sys.path.append(str(sub))

from data_utils import DataLoader
from preprocessing import DataPreprocessor
from label_generation import LabelGenerator

import home_pg
import data_exploration_pg
import eda_with_climate_event_pg
import feature_engineering_pg
import model_training_pg
import prediction_pg
import about_pg

st.set_page_config(
    page_title="Nepal Climate Sentinel",
    page_icon="🌐",
    layout="wide"
)

PAGES = {
    "🏠 Home": home_pg,
    "🗺️ Data Exploration": data_exploration_pg,
    "📊 EDA with Climate Events": eda_with_climate_event_pg,
    "⚙️ Feature Engineering": feature_engineering_pg,
    "📈 Model Training & Evaluation": model_training_pg,
    "🔮 Prediction": prediction_pg,
    "ℹ️ About": about_pg,
}

def load_data_pipeline():
    loader = DataLoader()
    st.info("🔄 Loading data...")
    try:
        loader.load_shape_file()
        loader.load_climate_data()
    except FileNotFoundError as e:
        st.error(f"Data file missing: {e}")
        st.stop()
    st.success("✅ Data loaded")

    pre = DataPreprocessor(loader.climate_df, loader.district_shp)
    pre.preprocess()
    st.success("✅ Data preprocessed")

    lg = LabelGenerator(pre.df)
    lg.label_generation_pipeline()
    st.success("✅ Labels generated")

    return pre.gdf, lg.df, lg.thresholds

def sidebar_nav():
    st.sidebar.header("📋 Navigation")
    choice = st.sidebar.selectbox(
        "Go to", list(PAGES.keys()), index=0, key="nav_selectbox"
    )
    st.sidebar.markdown("---")
    st.sidebar.caption("Managed by : Suresh Subedi (github.com/su-resh)")
    return choice

def main():
    st.markdown("# 🌐 Nepal Climate Sentinel")
    st.markdown("##### Extreme Weather Analytics & Forecasting")

    gdf, df, thresholds = load_data_pipeline()
    choice = sidebar_nav()
    page = PAGES[choice]

    if choice == "🏠 Home":
        page.show()
    elif choice == "🗺️ Data Exploration":
        page.show(gdf=gdf, df=df)
    elif choice == "📊 EDA with Climate Events":
        page.show(gdf=gdf, df=df, thresholds=thresholds)
    elif choice == "⚙️ Feature Engineering":
        page.show(df=df)
    elif choice == "📈 Model Training & Evaluation":
        page.show(df=df)
    elif choice == "🔮 Prediction":
        page.show()
    elif choice == "ℹ️ About":
        page.show()

if __name__ == "__main__":
    main()
