import streamlit as st
import pandas as pd

# utils_dir = os.path.join(Path(__file__).parent, "utils")
# sys.path.append(utils_dir)
# from data_utils import DataLoader
# from data_utils import DataLoader
# from preprocessing import DataPreprocessor
# from visualizations import plot_histogram, plot_pairplot, plot_correlation_heatmap 
# from visualizations import plot_district_map, choropleth_map, plot_time_series, plot_boxplot_monthly

from utils.data_utils import DataLoader
from utils.preprocessing import DataPreprocessor
from utils.visualizations import plot_histogram, plot_pairplot, plot_correlation_heatmap 
from utils.visualizations import plot_district_map, choropleth_map, plot_time_series, plot_boxplot_monthly


columns_required = {'Precip': {'aggregation': 'mean', 'unit': 'mm/day'},
                # 'Pressure': {'aggregation': 'mean', 'unit': 'kPa'},
                'Humidity_2m': {'aggregation': 'mean', 'unit': 'g/kg'},
                # 'RH_2m': {'aggregation': 'mean', 'unit': '%'},
                'Temp_2m': {'aggregation': 'mean', 'unit': '°C'},
                'MaxTemp_2m': {'aggregation': 'max', 'unit': '°C'},
                'MinTemp_2m': {'aggregation': 'min', 'unit': '°C'},
                # 'WindSpeed_10m': {'aggregation': 'mean', 'unit': 'm/s'},
                # 'MaxWindSpeed_10m': {'aggregation': 'max', 'unit': 'm/s'},
                # 'MinWindSpeed_10m': {'aggregation': 'min', 'unit': 'm/s'}
                }
agg_dict = {}  ## dictionary representing aggregation function for each columns
for key,value in columns_required.items():
    agg_dict[key] = value['aggregation']

def find_date_column(df):
    """Case-insensitive date column detection"""
    date_cols = ['date', 'datetime', 'time', 'timestamp']
    for col in df.columns:
        if str(col).lower() in date_cols:
            return col
    return None

def show(gdf, df):
    """
    Display the data exploration page
    """
    st.title("Data Exploration")

    # 1. Date column handling
    date_col = find_date_column(df)
    if not date_col:
        st.error("❌ No date column found in data. Available columns: " + ", ".join(df.columns))
        return
    
    try:
        # Convert to datetime and sort
        df['Date'] = pd.to_datetime(df[date_col])
        df = df.sort_values('Date')
    except Exception as e:
        st.error(f"Failed to parse dates: {str(e)}")
        st.write("Sample date values:", df[date_col].head(3))
        return

    # Compute outside the tab so it's not repeated on selectbox or other widget changes
    # Only compute heavy stuff once and store in session, to avoid re-compute on selectbox or other widget changes 
    # if "df_avg_districtwise" not in st.session_state:
    #     st.session_state["df_avg_districtwise"] = aggregated_data_per_district(gdf, df, columns_required)
    # # df_average_districtwise = st.session_state["df_avg_districtwise"]

        ## multiple tabs
    tab1, tab2, tab3 = st.tabs(["Overview", "Map", "Trends"])

    if 'df_aggregate_monthwise' not in st.session_state:
        st.session_state['df_aggregate_monthwise'] = aggregated_data_per_month(df)

    with tab1:
        st.subheader("Climate Data Summary")
        # Show the raw data
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Observations", df.shape[0])
            st.metric("Variables", df.shape[1])
        with col2:
            # print("Available columns:", df.columns.tolist())
            if "Date" in df.columns and not df["Date"].empty:
                st.metric("Date Range", f"{df['Date'].min().date()} to {df['Date'].max().date()}")
            else:
                st.warning("Date information not available.")

            # st.metric("Date Range", f"{df['Date'].min().date()} to {df['Date'].max().date()}")
            st.metric("Districts Covered", len(df['District'].unique()))

        # st.markdown("### Dataset Snapshot")
        st.dataframe(df.head())
        st.dataframe(df.tail())


        st.markdown("### Basic Info")
        st.write("**Column Names:**", list(df.columns))

        st.markdown("### Missing Values")
        missing = df.isnull().sum()
        st.dataframe(missing)

        # basic statistics
        st.subheader("Statistical Summary")
        st.write(df.describe())

        # Get aggregated data per month
        # df_aggregate_monthwise = aggregated_data_per_month(df)
        df_aggregate_monthwise = st.session_state['df_aggregate_monthwise']

        st.subheader("Distribution of climate variables.")
        fig = plot_histogram(df_aggregate_monthwise, (10,5), 2, 3)
        st.pyplot(fig)

        st.subheader("Pair Plot of some climate variables.")
        with st.expander("Pair Plot", expanded=False):
            # fig = plot_pairplot(df_aggregate_monthwise[['Precip','Pressure','Humidity_2m','Temp_2m','WindSpeed_10m']])
            fig = plot_pairplot(df_aggregate_monthwise[[key for key in columns_required.keys()]])
            st.pyplot(fig)

        st.subheader("Correlation Coefficient Heatmap of climate variables.")
        with st.expander("Correlation Heatmap", expanded=False):
            fig = plot_correlation_heatmap(df_aggregate_monthwise, (8,8))
            st.pyplot(fig)


    with tab2:
        ## map of nepal with district boundary
        st.subheader("District Map of Nepal")
        if gdf is not None:
            fig = plot_district_map(gdf)
            st.pyplot(fig)
        else:
            st.warning("District shapefile not loaded.")

        # Environment data on map
        st.subheader("District-wise climate data") 

        # df_average_districtwise = st.session_state["df_avg_districtwise"]
        df_average_districtwise = aggregated_data_per_district(gdf, df, columns_required)
        print(len(df_average_districtwise))  
        st.markdown("#### Select variable(s) to display choropleth maps:")

        for key, value in columns_required.items():
            plot_checkbox = st.checkbox(f'{key} ({value["unit"]}) - {value["aggregation"]}', key=f"chk_{key}")
            if plot_checkbox:
                print(key)            
                title = f'Aggregated ({value["aggregation"]}) of "{key} ({value['unit']})" over 1981-Jan-01 to 2019-Dec-31.'
                with st.spinner("Preprocessing title..."):                
                    fig = get_choropleth_plot(df_average_districtwise, key,title)
                st.success(f'{title} GENERATED')
                st.pyplot(fig)

    with tab3:
        st.subheader("Trends Over Time")

        # Create two columns
        col1, col2 = st.columns(2) ## use 2 selectbox in single row
        with col1:
            # Select variable to plot
            climate_variables = list(columns_required.keys())
            selected_variable = st.selectbox("Select climate variable", climate_variables)
        with col2:
        # district filter
            districts = sorted(df["District"].dropna().unique())
            selected_district = st.selectbox("Select District", ["All"] + districts)

        # aggregation function and unit of selected climate variable
        aggregation_selected = columns_required[selected_variable]['aggregation']
        unit_selected = columns_required[selected_variable]['unit']
        
        # df_average_monthly = aggregated_data_per_month(df) # get aggregated monthwise data
        df_average_monthly = st.session_state['df_aggregate_monthwise']
        # Filter data
        if selected_district != "All":
            df_filtered = df_average_monthly[df_average_monthly["District"] == selected_district].copy(deep=False)
        else:
            df_filtered = df_average_monthly.copy(deep=False)
        ## Now remove district column and aggregate in case all districts are selected
        # in case 1 district is selected aggregate value is same as non-aggregate value
        df_filtered_agg_timeseries = df_filtered.drop(columns=['District','Month']).groupby(by='Date').mean().reset_index()

        st.markdown("### Aggregated month-wise Climate data")
        title = f"Monthly Aggregated ({aggregation_selected}) trend of '{selected_variable}({unit_selected})' over time for {selected_district} district:"  
        fig = plot_time_series(df_filtered_agg_timeseries[['Date', selected_variable]], 
                               column = selected_variable, title = title)    
        # st.success(f'{title} GENERATED')       
        st.pyplot(fig)   

        st.markdown("### Box plot of Month-wise Climate data")
        title = f"Month-wise box plot of '{aggregation_selected} of {selected_variable}({unit_selected})' for {selected_district} district:"  
        fig = plot_boxplot_monthly(df_filtered[['Month', selected_variable]], 
                               column = selected_variable, title = title)    
        # st.success(f'{title} GENERATED')       
        st.pyplot(fig) 



@st.cache_data
def aggregated_data_per_district(_gdf, df, columns_required):
    """
    This method computes average values of the environmental data per district from given gdf and df
    """   
    df_aggregated = df.groupby(by = 'District').agg(agg_dict).reset_index()
    df_aggregated['District'] = df_aggregated['District'].str.upper()

    merged_data = _gdf.merge(df_aggregated, left_on='DISTRICT',right_on='District')
    merged_data.drop(columns=['DISTRICT','Province'],inplace=True)

    return merged_data

@st.cache_data
def aggregated_data_per_month(df):
    """
    This method returns monthly aggregated data from daily data, to reduce sample points
    """
    # df['Date'] = pd.to_datetime(df['Date'])  # Ensure datetime
    df.set_index('Date', inplace=True)       # Set Date as index
    # Resample monthly for each district
    df_monthly = (
        df.groupby('District')
        .resample('ME')
        .agg(agg_dict)
        .reset_index()
    )
    df_monthly['Month'] = df_monthly['Date'].dt.month_name()
    return df_monthly

def get_choropleth_plot(gdf_districtwise, key, title):
    """
    Returns a cached choropleth plot from session state or creates it if not available.
    This allow only compute heavy stuff once and store in session, 
    to avoid re-compute on selectbox or other widget changes 
    """
    # plot_key = f"{key}_choropleth"
    # if plot_key not in st.session_state:
    #     fig = choropleth_map(gdf_districtwise, key, title)
    #     st.session_state[plot_key] = fig
    # return st.session_state[plot_key]

    return choropleth_map(gdf_districtwise, key, title)
    
