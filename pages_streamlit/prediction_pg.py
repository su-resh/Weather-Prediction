import streamlit as st
import pandas as pd

from utils.models import load_model
from utils.prediction import predict_until_date
from utils.visualizations import plot_regression_predictions

def show():
    """
    Page to make prediction
    df: data frame containing feature_columns (List of feature names expected by the models)
    """
    st.title("Make Prediction")
    
    if 'fe_obj' not in st.session_state:
        st.error("Feature engineering not performed yet. Please first load data and perform feature engineerig from 'Feature Engineering' page .")
        return None
    
    # Load trained models
    reg_model, reg_scaler = load_model('regression')   # No .pkl needed because your load_model handles that
    multi_class_model, multi_class_scaler = load_model('multi_class_classifier')
    binary_class_model, binary_class_scaler = load_model('binary_classifier')

    if reg_model is None or multi_class_model is None or binary_class_model is None:
        st.error("Models not found. Please first train and save the models from 'Model Training and Evaluation' page .")
        return None

    # Input for District and Date
    col1, col2 = st.columns(2)
    with col1:
        target_date = st.date_input("Select Target Date", value=pd.to_datetime("2021-01-01"))
    with col2: 
        districts = st.session_state.fe_obj.district_classes
        selected_district = st.selectbox("Select District", districts)

    # Get district_encoded
    district_mapping = {district: idx for idx, district in enumerate(districts)}
    district_encoded = district_mapping[selected_district]

    # data loader from 
    # Set rolling and lag days
    rolling_window = st.session_state['rolling_window_size']
    lag_days = st.session_state['lag_days']
    last_n_rows = max(rolling_window, lag_days)
    # Select only necessary columns for prediction
    required_columns = ['Date', 'district_encoded', 'Precip', 'Humidity_2m', 'Temp_2m', 'MaxTemp_2m', 'MinTemp_2m']
    # existing_latest_climate_df = st.session_state.fe_obj.df.tail(last_n_rows)
    existing_latest_climate_df = st.session_state.fe_obj.df[required_columns][
        st.session_state.fe_obj.df['district_encoded'] == district_encoded
    ].tail(last_n_rows)

    if st.button("Predict"):
        st.write(f"Making prediction of climate data and extreme event upto {target_date} for {selected_district}.")
        event_type_mapping = {
            idx: label for idx, label in enumerate(st.session_state.fe_obj.event_type_classes)
        }
        predicted_df = predict_until_date(reg_model, reg_scaler, multi_class_model, binary_class_model,
                                          existing_latest_climate_df, 
                                          pd.to_datetime(target_date), district_encoded,
                                          rolling_window, lag_days,
                                          event_type_mapping)
        st.write("Predicted climate data along with all feature engineered columns:")
        st.dataframe(predicted_df)

        st.subheader(f"Climate variable and Weather Event prediction on {target_date} for {selected_district}:")
        display_last_day_predictions(prediction = predicted_df.iloc[-1])

        st.subheader(f"Historical and predicted climate variables prediction upto {target_date} for {selected_district}:")
        fig = plot_regression_predictions(historical_df = st.session_state.fe_obj.df[required_columns], 
                                          predicted_df=predicted_df, 
                                          district_encoded = district_encoded, 
                                          district_name=selected_district)
        st.pyplot(fig)


def display_last_day_predictions(prediction):
    """Display last day's predictions in a Streamlit-friendly format with visual enhancements."""

    # Create columns for better layout
    col1, col2 = st.columns(2)
    
    with col1:
        # st.subheader("Last Day Predictions")
        st.metric("Date", prediction['Date'].strftime('%Y-%m-%d'))        
        # Weather metrics
        st.metric("Precipitation (mm)", f"{prediction['Precip']:.1f}")
        st.metric("Temperature (°C)", f"{prediction['Temp_2m']:.1f}")
    
    with col2:
        st.write("")  # Spacer
        st.write("")  # Spacer
        st.metric("Min Temperature (°C)", f"{prediction['MinTemp_2m']:.1f}")
        st.metric("Max Temperature (°C)", f"{prediction['MaxTemp_2m']:.1f}")
        st.metric("Humidity (%)", f"{prediction['Humidity_2m']:.1f}")
    
    # Event warnings (with visual emphasis)
    # if prediction['EventType'] is not None or extreme_flag is not None:
    st.markdown("---")
    warning_container = st.container()
    
    with warning_container:
        if prediction['EventType_Label'] != 'Unknown':
            emoji = "⚠️" if prediction['EventType_Prob'] > 0.3 else "ℹ️"
            st.warning(f"{emoji} Predicted Event: **{prediction['EventType_Label']}** (Probability: {prediction['EventType_Prob']:.0%})")
        else:
            st.warning(f"✅ Normal Climet event day")
        
        if prediction['ExtremeEvent']:
            st.warning(f"🚨 **Extreme Event Warning** (Probability: {prediction['ExtremeEvent_Prob']:.0%}) - Take necessary precautions!")
        else:
            st.success("✅ No extreme events predicted")
