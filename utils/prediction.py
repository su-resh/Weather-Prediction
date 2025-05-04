import numpy as np
import pandas as pd

def prepare_features_for_date(working_df, target_date, district_encoded, rolling_window, lag_days):
    """
    Prepare input features for regression model for a given target_date and district.
    working_df: dataframe containing the latest known (real+predicted) climate data
    rolling_window: int, number of days for moving average
    lag_days: int, number of days to look back for lag features
    """

    # Extract temporal features
    year = target_date.year
    month = target_date.month
    dayofweek = target_date.dayofweek
    dayofyear = target_date.dayofyear
    season = month % 12 // 3 + 1  # Rough way to calculate season (1: Winter, 2: Spring, 3: Summer, 4: Fall)

    # Cyclical features
    dayofyear_sin = np.sin(2 * np.pi * dayofyear / 365.0)
    month_cos = np.cos(2 * np.pi * month / 12.0)

    # Filter data for the specific district
    district_data = working_df[working_df['district_encoded'] == district_encoded].sort_values('Date')

    # Rolling (moving average) features
    recent_rolling = district_data.tail(rolling_window)

    precip_rolling_avg = recent_rolling['Precip'].mean()
    humidity_rolling_avg = recent_rolling['Humidity_2m'].mean()
    temp_rolling_avg = recent_rolling['Temp_2m'].mean()
    maxtemp_rolling_avg = recent_rolling['MaxTemp_2m'].mean()
    mintemp_rolling_avg = recent_rolling['MinTemp_2m'].mean()

    # Lag features
    if len(district_data) >= lag_days:
        lag_row = district_data.iloc[-lag_days]  # -lag_days from the end
        precip_lagged = lag_row['Precip']
        humidity_lagged = lag_row['Humidity_2m']
        temp_lagged = lag_row['Temp_2m']
        maxtemp_lagged = lag_row['MaxTemp_2m']
        mintemp_lagged = lag_row['MinTemp_2m']
    else:
        precip_lagged = humidity_lagged = temp_lagged = maxtemp_lagged = mintemp_lagged = 0  # fallback if not enough data

    # temp_range = district_data['MaxTemp_2m'].iloc[-1] - district_data['MinTemp_2m'].iloc[-1] if not district_data.empty else 0

    # Create final input dataframe
    features = pd.DataFrame({
        'year': [year],
        'month': [month],
        'dayofweek': [dayofweek],
        'dayofyear': [dayofyear],
        'season': [season],
        'dayofyear_sin': [dayofyear_sin],
        'month_cos': [month_cos],
        'Precip_rolling_avg': [precip_rolling_avg],  # Note: the column names in training were hardcoded for 7d
        'Humidity_2m_rolling_avg': [humidity_rolling_avg],
        'Temp_2m_rolling_avg': [temp_rolling_avg],
        'MaxTemp_2m_rolling_avg': [maxtemp_rolling_avg],
        'MinTemp_2m_rolling_avg': [mintemp_rolling_avg],
        'Precip_lagged': [precip_lagged],
        'Humidity_2m_lagged': [humidity_lagged],
        'Temp_2m_lagged': [temp_lagged],
        'MaxTemp_2m_lagged': [maxtemp_lagged],
        'MinTemp_2m_lagged': [mintemp_lagged],
        # 'temp_range': [temp_range],
        'district_encoded': [district_encoded]
    })

    return features

def predict_until_date(regression_model, regression_scaler, multi_class_model, binary_class_model,
                       latest_known_df, 
                       target_date, district_encoded,
                       rolling_window, lag_days,
                       event_type_mapping):
    """
    Predict day-by-day climate data for a district up to target_date.
    Returns a dataframe containing all the newly predicted rows.
    latest_known_df: Only for the selected district (already filtered)
    """
    current_date = latest_known_df['Date'].max() + pd.Timedelta(days=1)
    working_df = latest_known_df.copy()

    predicted_rows_all = []  # to collect all predicted rows with featured engineered columns as well

    while current_date <= target_date:
        input_features = prepare_features_for_date(working_df, current_date, district_encoded, 
                                                   rolling_window, lag_days)
        # print(input_features.keys())
        # Convert single-row DataFrame to dict of scalars
        input_features_dict = input_features.iloc[0].to_dict()

        # Scale input features if scaler is available
        if regression_scaler != None:
            # Ensure scaler sees features in same order as training
            # if hasattr(regression_scaler, 'feature_names_in_'):  # sklearn >=1.0
            #     input_for_scaler = input_features[regression_scaler.feature_names_in_]
            # elif hasattr(regression_scaler, 'feature_names_'):  # Custom fallback
            #     input_for_scaler = input_features[regression_scaler.feature_names_]
            # else:
            #     input_for_scaler = input_features
            # input_features_scaled = regression_scaler.transform(input_for_scaler)
            input_features_scaled = regression_scaler.transform(input_features)
        else:
            input_features_scaled = input_features

        # Handle features order
        if hasattr(regression_model, 'feature_names_in_'):  # sklearn >=1.0
            input_for_reg = input_features_scaled[regression_model.feature_names_in_]
        elif hasattr(regression_model, 'feature_names_'):  # Custom fallback
            input_for_reg = input_features_scaled[regression_model.feature_names_in_]
        else:
            input_for_reg = input_features_scaled

        predicted_values = regression_model.predict(input_for_reg)[0]

        new_row = {
            'Date': current_date,
            'year': input_features_dict['year'], 
            'month': input_features_dict['month'],
            'dayofweek': input_features_dict['dayofweek'],
            'dayofyear': input_features_dict['dayofyear'],
            'season': input_features_dict['season'],
            'dayofyear_sin': input_features_dict['dayofyear_sin'],
            'month_cos': input_features_dict['month_cos'], 
            'district_encoded': district_encoded,
            'Precip': predicted_values[0],
            'Humidity_2m': predicted_values[1],
            'Temp_2m': predicted_values[2],
            'MaxTemp_2m': predicted_values[3],
            'MinTemp_2m': predicted_values[4],
        }

        # Create temporary df to calculate rolling, lagged, and combined features
        temp_df = pd.concat([working_df, pd.DataFrame([new_row])], ignore_index=True)

        # Recompute rolling averages
        district_temp = temp_df[temp_df['district_encoded'] == district_encoded].sort_values('Date')

        # Rolling averages
        precip_rolling_avg = district_temp['Precip'].tail(rolling_window).mean()
        humidity_rolling_avg = district_temp['Humidity_2m'].tail(rolling_window).mean()
        temp_rolling_avg = district_temp['Temp_2m'].tail(rolling_window).mean()
        maxtemp_rolling_avg = district_temp['MaxTemp_2m'].tail(rolling_window).mean()
        mintemp_rolling_avg = district_temp['MinTemp_2m'].tail(rolling_window).mean()

        # Lag values
        if len(district_temp) >= lag_days:
            precip_lagged = district_temp['Precip'].iloc[-lag_days]
            humidity_lagged = district_temp['Humidity_2m'].iloc[-lag_days]
            temp_lagged = district_temp['Temp_2m'].iloc[-lag_days]
            maxtemp_lagged = district_temp['MaxTemp_2m'].iloc[-lag_days]
            mintemp_lagged = district_temp['MinTemp_2m'].iloc[-lag_days]
        else:
            precip_lagged = humidity_lagged = temp_lagged = maxtemp_lagged = mintemp_lagged = 0

        # Combined features
        temp_humidity_combined = new_row['Temp_2m'] * new_row['Humidity_2m']
        temp_precip_combined = new_row['Temp_2m'] * new_row['Precip']

        new_row.update({
            # Rolling features
            'Precip_rolling_avg': precip_rolling_avg,
            'Humidity_2m_rolling_avg': humidity_rolling_avg,
            'Temp_2m_rolling_avg': temp_rolling_avg,
            'MaxTemp_2m_rolling_avg': maxtemp_rolling_avg,
            'MinTemp_2m_rolling_avg': mintemp_rolling_avg,

            # Lag features
            'Precip_lagged': precip_lagged,
            'Humidity_2m_lagged': humidity_lagged,
            'Temp_2m_lagged': temp_lagged,
            'MaxTemp_2m_lagged': maxtemp_lagged,
            'MinTemp_2m_lagged': mintemp_lagged,

            # Combined features
            'temp_humidity_combined': temp_humidity_combined,
            'temp_precip_combined': temp_precip_combined
        })

        # Save prediction
        predicted_rows_all.append(new_row)

        # Update latest data to working_df only with new_row which does not contain feature engineered columns
        working_df = pd.concat([working_df, pd.DataFrame([new_row])], ignore_index=True)

        current_date += pd.Timedelta(days=1)    

    # Create a DataFrame of only predicted rows
    predicted_df = pd.DataFrame(predicted_rows_all)

    # Predict Event Type (Multi-Class Classifier) and Extreme Event (Binary Classifier)
    direct_features = ['Temp_2m', 'MinTemp_2m', 'MaxTemp_2m', 'Precip']
    exclude_cols = ['Date'] + direct_features

    # Drop unneeded columns but ensure necessary columns are preserved
    input_for_classification = predicted_df.drop(columns=[col for col in exclude_cols if col in predicted_df.columns])

    # Ensure input features match the ones used during training (both models)
    # input_for_multiclass = input_for_classification[multi_class_model.feature_names_in_]
    # input_for_binary = input_for_classification[binary_class_model.feature_names_in_]
    # safe way to use feature names
    if hasattr(multi_class_model, 'feature_names_in_'):
        input_for_multiclass = input_for_classification[multi_class_model.feature_names_in_] # sklearn >=1.0
    elif hasattr(multi_class_model, 'feature_names_'):
        input_for_multiclass = input_for_classification[multi_class_model.feature_names_] # Custom fallback
    else:
        input_for_multiclass = input_for_classification
    
    if hasattr(binary_class_model, 'feature_names_in_'):
        input_for_binary = input_for_classification[binary_class_model.feature_names_in_]
    elif hasattr(binary_class_model, 'feature_names_'):
        input_for_binary = input_for_classification[binary_class_model.feature_names_]
    else:
        input_for_binary = input_for_classification

    # Make predictions of event type and its probability
    predicted_event_types = multi_class_model.predict(input_for_multiclass)
    predicted_event_probs = multi_class_model.predict_proba(input_for_multiclass)

    # Make predictions of extreme event and its probability
    predicted_extreme_events = binary_class_model.predict(input_for_binary)
    predicted_extreme_probs = binary_class_model.predict_proba(input_for_binary)[:, 1]  # probability for class 1

    # Map event type class index to label names
    # e.g., if you used LabelEncoder or manually know the mapping
    predicted_event_labels = [event_type_mapping.get(label, 'Unknown') for label in predicted_event_types]

    # Add predictions to DataFrame
    predicted_df['EventType'] = predicted_event_types
    predicted_df['EventType_Label'] = predicted_event_labels
    predicted_df['EventType_Prob'] = predicted_event_probs.max(axis=1)  # Max confidence, highest class probability

    predicted_df['ExtremeEvent'] = predicted_extreme_events
    predicted_df['ExtremeEvent_Prob'] = predicted_extreme_probs # probability of being an extreme event

    return predicted_df