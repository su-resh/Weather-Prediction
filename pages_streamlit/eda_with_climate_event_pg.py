import streamlit as st
# from visualizations import plot_event_type_distribution, plot_event_type_districtwise
from utils.visualizations import plot_event_type_distribution, plot_event_type_districtwise

def show(gdf, df, thresholds):
    """
    Display EDA after label generation
    """
    st.title("Data Exploration with Climate events")
    st.markdown(f""" Categorical "EventType" is added based on Extreme weather event. 
               \n Following are Extreme weather event "EventType" labels based on raw climate data:                                
- Coldwave if MinTemp_2m < {thresholds['MinTemp_2m']} (°C)
- HighTemp if Temp_2m > {thresholds['Temp_2m']} (°C)
- Heatwave if MaxTemp_2m > {thresholds['MaxTemp_2m']} (°C)
- HeavyRain if Precip > {thresholds['Precip']} (mm)
- Normal if no extreme weather event

            \n Binary label "ExtremeEvent" is added based on "EventType"
- 1 if EventType is not Normal
- 0 if EventType is Normal    
""")
    st.subheader("'ExtremeEvent' and 'EventType' Summary")
    # Show the raw data
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total Observations", df.shape[0])
        st.metric("Count of ExtremeEvent", df['ExtremeEvent'].sum())       

    with col2:
        col2.markdown("#### 'EventType' distribution")
        st.metric("Count of 'Normal' EventType", len(df[df['EventType']=='Normal']))  
        st.metric("Count of 'HeavyRain' EventType", len(df[df['EventType']=='HeavyRain']))  
        st.metric("Count of 'Heatwave' EventType", len(df[df['EventType']=='Heatwave']))  
        st.metric("Count of 'HighTemp' EventType", len(df[df['EventType']=='HighTemp']))  
        st.metric("Count of 'ColdWave' EventType", len(df[df['EventType']=='ColdWave']))  
        # st.metric("Count of 'HighWind' EventType", len(df[df['EventType']=='HighWind']))  

    st.subheader("Climate ExtremeEvent Event Type Distribution")
    fig = plot_event_type_distribution(df)
    st.plotly_chart(fig)

    st.subheader("District-wise EventType Distribution")
    fig = plot_event_type_districtwise(df)
    st.plotly_chart(fig)