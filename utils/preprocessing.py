"""
This file pre-process data
"""
import pandas as pd
import geopandas as gpd
import streamlit as st

class DataPreprocessor:
    def __init__(self, climate_df: pd.DataFrame, district_gdf: gpd.GeoDataFrame):
        self.df = climate_df.copy()
        self.gdf = district_gdf.copy()
        # self.preprocess()

    def preprocess(self):
        with st.spinner("Preprocessing climate data..."):
            self.preprocess_df()
        with st.spinner("Preprocessing district boundaries..."):
            self.preprocess_gdf()
        return self.df, self.gdf

    def preprocess_df(self):
        """
        DataFrame pre-processing pipeline.
        """
        self.drop_columns()
        self.convert_date()
        self.drop_missing()

    def preprocess_gdf(self):
        """
        Geo-DataFrame pre-processing pipeline.
        """
        # merges ["NAWALPARASI_E", "NAWALPARASI_W"] to "NAWALPARASI"
        self.merge_two_districts(to_merge=["NAWALPARASI_E", "NAWALPARASI_W"], new_name="NAWALPARASI")
        # merges ["RUKUM_E", "RUKUM_W"] to "RUKUM"
        self.merge_two_districts(to_merge=["RUKUM_E", "RUKUM_W"], new_name="RUKUM")
        self.rename_districts()
        # print(self.gdf.head())

    def drop_columns (self, columns=['Latitude', 'Longitude']):
        """
        Delete unnecessary columns.
        """
        self.df.drop(columns=columns, errors='ignore', inplace=True)
        print(f"'{columns}' columns deleted.")

    def convert_date(self, date_column = 'Date'):
        """
        Convert Date column into datetime format if it exists
        """
        if date_column in self.df.columns:
            self.df[date_column] = pd.to_datetime(self.df[date_column], errors="coerce")
            print(f"'{date_column}' column converted to datetime.")

    def drop_missing(self, key_columns = ['Date', 'District']):
        """
        Drop rows with missing values in key columns.
        By default Date and District are key fields.
        """
        before = len(self.df)
        self.df.dropna(subset=key_columns, inplace=True)
        after = len(self.df)
        if after < before:
            print(f'{before - after} rows with missing vlaues in "{key_columns}" columns dropped.')
        else:
            print(f'There is no missing values in "{key_columns}" columns.')

    def merge_two_districts(self, to_merge, new_name):
        """
        This method merges two districts of gdf to single one
        This method merges ["NAWALPARASI_E", "NAWALPARASI_W"] to "NAWALPARASI" and ["RUKUM_E", "RUKUM_W"] to "RUKUM"
        This is because climate data is available for combined districts prior to divisio of districts.
        """
        if self.gdf is None:
            print("GeoDataFrame not loaded.")
            return

        # Filter the two districts
        districts_to_merge = self.gdf[self.gdf["DISTRICT"].isin(to_merge)]
        if districts_to_merge.empty:
            print(f"No districts found for merging: {to_merge}")
            return
        # Combine (union) their geometries
        combined_geometry = districts_to_merge.unary_union
        # Create a new GeoDataFrame for the merged district
        merged_district = gpd.GeoDataFrame({
                "DISTRICT": [new_name],
                "geometry": [combined_geometry]
            }, crs=self.gdf.crs)
        # Drop the original districts and append the new one
        self.gdf = self.gdf[~self.gdf["DISTRICT"].isin(to_merge)]
        self.gdf = pd.concat([self.gdf, merged_district], ignore_index=True)
        print(f'{to_merge} districts merged to {new_name}')

    def rename_districts(self):
        """
        This method rename name of 8 districts to match with the climate data df's district name
        """
        name_map ={
            'BAJHANG':'BAJANG',
            'DHANUSHA':'DHANUSA',
            'DOLAKHA':'DOLKHA',
            'KABHREPALANCHOK':'KABHRE',
            'MAKAWANPUR':'MAKWANPUR',
            'PANCHTHAR':'PANCHTHER',
            'RAUTAHAT':'ROUTAHAT',
            'TANAHU':'TANAHUN',
        }
        self.gdf['DISTRICT'] = self.gdf['DISTRICT'].replace(name_map)
        print(f'Renamed districts: {name_map}')

