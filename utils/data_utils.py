"""
This file handles data loading
"""
import pandas as pd
import geopandas as gpd
from pathlib import Path
import streamlit as st

@st.cache_data(show_spinner=False, ttl=3600)
def load_cached_shape_file(shape_file):
    st.info("Loading shapefile from cache...")
    return gpd.read_file(shape_file)

@st.cache_data(show_spinner="Loading climate data...", ttl=3600)
def load_cached_climate_data(climate_file):
    st.info("Loading climate CSV from cache...")
    df = pd.read_csv(climate_file)
    df = preprocess_dates(df)
    return df

def preprocess_dates(df):
    """Ensure proper datetime serialization"""
    if 'Date' in df.columns:
        # Convert to datetime and normalize to avoid nanoseconds
        df['Date'] = pd.to_datetime(df['Date']).dt.normalize()
        # Convert to string for Arrow compatibility
        df['Date'] = df['Date'].dt.strftime('%Y-%m-%d')
    return df

class DataLoader:
    """
    This class handle data loading for the project
    """
    def __init__(self, shape_file = r'data/Shape_Data_district/district.shp',
                 climate_file = r'data/dailyclimate-2.csv'):
        self.shape_file = shape_file
        self.climate_file = climate_file

        # initialization of variables
        self.district_shp = None
        self.climate_df = None
        
        # self.load_shape_file()
        # self.load_climate_data()

    def file_exists(self, file_name) -> bool:
        """
        Checks if file with file_name exists
        """
        file_path = Path(file_name)        
        exists = file_path.exists()
        # print(f'The file {file_name} exists: {exists}.')
        return exists
    
    def load_shape_file(self):
        """
        Loads the shapefile as a GeoDataFrame.
        """
        if self.file_exists(self.shape_file):
            self.district_shp = load_cached_shape_file(self.shape_file)
            # self.district_shp = gpd.read_file(self.shape_file)
            print(f'Shape data for Districts of Nepal loaded successfully.') # show the messages in the Streamlit app UI
        else:
            print(f'Shape file "{self.shape_file}" does not exits.')

    def load_climate_data(self):
        """
        Loads the climate CSV data into a DataFrame.
        """
        if self.file_exists(self.climate_file):
            self.climate_df = load_cached_climate_data(self.climate_file)
            # self.climate_df = pd.read_csv(self.climate_file)
            print(f'Climate data from 93 weather stations spanning 62 districts in Nepal loaded successfully.')
        else:
            print(f'Climate data file "{self.climate_file}" does not exits.')
    
class PrepareData:
    """
    This class handles data preparation (Features and target variable) for three types of predictions.
    The 3 types of predictions are:
    1. regression: predicting climate variables (Precip, Humidity_2m, Temp_2m, MaxTemp_2m, MinTemp_2m)
    2. multi-class classifier: predicting the type of event (EventType)
    3. binary classifier: predicting whether an extreme event occurs (ExtremeEvent)
    """
    def __init__(self, df):
        """_summary_
        Initialize class
        Args:
            df : DataFrame input for Data Preparation. This is dataframe after feature engineering columns are added
        """
        self.df = df.copy()
        self.all_columns = df.columns

    def prepare_data(self, target_cols, exclude_cols):
        """
        Prepare data from given target columns and exclusion columns
        """
        feature_cols = [col for col in self.all_columns if col not in target_cols + exclude_cols]
        
        X = self.df[feature_cols]
        y = self.df[target_cols]

        # Reset index after splitting
        X.reset_index(drop=True, inplace=True)
        y.reset_index(drop=True, inplace=True)

        return X, y
    
    def prepare_data_regression(self):
        """
        prepare data for regression
        """
        target_cols = ['Precip', 'Humidity_2m', 'Temp_2m', 'MaxTemp_2m', 'MinTemp_2m']
        exclude_cols  = ['Date', 'ExtremeEvent', 'eventtype_encoded', 'temp_precip_combined', 'temp_humidity_combined']        
        
        return self.prepare_data(target_cols, exclude_cols)

    def prepare_data_multi_classifier(self, direct_features = ['Temp_2m', 'MinTemp_2m', 'MaxTemp_2m', 'Precip']):
        """
        prepare data for multi-class classifier
        direct_features are List of features to exclude from the dataset
        """
        target_cols = ['eventtype_encoded']    # the target column for multi-class classification
        exclude_cols = ['Date', 'ExtremeEvent'] + direct_features # columns to exclude (date, binary event flag, and direct features)

        return self.prepare_data(target_cols, exclude_cols)

    def prepare_data_binary_classifier(self, direct_features = ['Temp_2m', 'MinTemp_2m', 'MaxTemp_2m', 'Precip']):
        """
        prepare data for binary classifier
        direct_features are List of features to exclude from the dataset
        """
        target_cols = ['ExtremeEvent']    # the target column for binary classification
        exclude_cols = ['Date', 'eventtype_encoded'] + direct_features # columns to exclude (date, categorical eventtype_encoded, and direct features)

        return self.prepare_data(target_cols, exclude_cols)