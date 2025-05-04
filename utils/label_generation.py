"""
This module defines a class for generating categorical Climate Event tpye and 
binary labels indicating extreme weather events.
Event types are determined based on configurable thresholds and include:
- Heatwave (MaxTemp_2m > threshold)
- Coldwave (MinTemp_2m < threshold)
- HeavyRain (Precip > threshold)
- HighWind (WindSpeed_10m > threshold)
- HighTemp (Temp_2m > threshold)
"""
import pandas as pd

class LabelGenerator:
    def __init__(self, df, thresholds=None):
        """
        Initialize class
        """
        self.df = df.copy()
        if thresholds:
            self.thresholds = thresholds
        else:
            self.thresholds = {
                'Precip': 50,          # mm (heavy rainfall threshold)
                'Temp_2m': 35,         # °C (high temperature threshold)
                # 'WindSpeed_10m': 15,    # m/s (high wind speed threshold)
                'MaxTemp_2m': 38,         # °C (heatwave threshold)
                'MinTemp_2m': -3           # °C (cold event threshold)
            }

    def label_generation_pipeline(self):        
        """
        Runs the full label generation pipeline.
        """
        self.generate_event_type()
        self.generate_extreme_event()
        return self.df
    
    def generate_event_type(self):
        """
        Generate EventType (categorical) columns based on threholds.
        EventType is based on priority-based overwrite. This means later conditions overwrite earlier ones
        """
        # Create condition masks
        # high_wind = self.df['WindSpeed_10m'] > self.thresholds['WindSpeed_10m']
        coldwave = self.df['MinTemp_2m'] < self.thresholds['MinTemp_2m']
        high_temp = self.df['Temp_2m'] > self.thresholds['Temp_2m']
        heatwave = self.df['MaxTemp_2m'] > self.thresholds['MaxTemp_2m']        
        heavy_rain = self.df['Precip'] > self.thresholds['Precip']

        # Assign event type (priority-based, overwrite as needed, later conditions overwrite earlier ones)
        self.df['EventType'] = 'Normal'
        # self.df.loc[high_wind, 'EventType'] = 'HighWind'
        self.df.loc[coldwave, 'EventType'] = 'ColdWave'
        self.df.loc[high_temp, 'EventType'] = 'HighTemp'
        self.df.loc[heatwave, 'EventType'] = 'Heatwave'        
        self.df.loc[heavy_rain, 'EventType'] = 'HeavyRain'

    def generate_extreme_event(self):
        """
        Generates binary ExtremeEvent column based on EventType
        """
        self.df['ExtremeEvent'] = (self.df['EventType'] != 'Normal').astype(int)

