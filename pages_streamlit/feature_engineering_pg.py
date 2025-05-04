import streamlit as st

# utils_dir = os.path.join(Path(__file__).parent, "utils")
# sys.path.append(utils_dir)
# from feature_engineering import FeatureEngineering
# from visualizations import plot_histogram

from utils.feature_engineering import FeatureEngineering
from utils.visualizations import plot_histogram

def show(df):
    """
    Display the model training page
    """
    st.title("Feature Engineering")    

    # # Initialize session state for rolling/lagged checkboxes
    # if 'generate_rolling' not in st.session_state:
    #     st.session_state['generate_rolling'] = False
    # if 'generate_lagged' not in st.session_state:
    #     st.session_state['generate_lagged'] = False

    added_feature = []
    fe = FeatureEngineering(df)  
    # st.session_state.df_feature_added = None
    st.session_state.fe_obj = None
    with st.form("feature_eng_form"):
        st.subheader("Feature Engineering Options")
        col1, col2 = st.columns(2)
        with col1:
            generate_temporal = st.checkbox("Generate Temporal Features", value=True, disabled=True)
            generate_cyclical = st.checkbox("Generate Cyclical Features", value=True, disabled=True)
            generate_combined = st.checkbox("Generate Combined Features", value=True, disabled=True)
        with col2:
            col21, col22 = st.columns(2)
            with col21:
                generate_rolling = st.checkbox("Generate Rolling Features", value=True, disabled=True)
                window_size = st.slider("Rolling Window Size (days)", 3, 15, 7,
                                        help="Only used if 'Generate Rolling Features' is checked.")
                st.session_state['rolling_window_size'] = window_size
            with col22:
                generate_lagged = st.checkbox("Generate Lagged Features", value=True, disabled=True)
                lag_value = st.slider("Lag Value (days)", 1, 20, 7,
                                      help="Only used if 'Generate Lagged Features' is checked.")
                st.session_state['lag_days'] = lag_value

        submit = st.form_submit_button("Apply Feature Engineering")

        if submit:
            # fe = FeatureEngineering(df)           
            
            if generate_temporal:
                features_add = fe.generate_temporal_features()
                added_feature.extend(features_add)
                st.write(f"{len(features_add)} Temporal Features {features_add} created.")
            if generate_cyclical:
                features_add = fe.generate_cyclical_features()
                added_feature.extend(features_add)
                st.write(f"{len(features_add)} Cyclical Features {features_add} created.")
            if generate_rolling:
                features_add = fe.generate_rolling_features(window=window_size)
                added_feature.extend(features_add)
                st.write(f"{len(features_add)} Rolling window features for {features_add} with window size of {window_size} created.")
            if generate_lagged:
                features_add = fe.generate_lagged_features(lag=lag_value)
                added_feature.extend(features_add)
                st.write(f"{len(features_add)} Lagged features for {features_add} with lag value of {lag_value} created.")
            if generate_combined:
                features_add = fe.generate_combined_features()
                added_feature.extend(features_add)
                st.write(f"{len(features_add)} Combined climate Features {features_add} created.")
            fe.encode_district()
            district_mapping = {i: district for i, district in enumerate(fe.district_classes)}
            st.write(f"'District' column is encoded with: {district_mapping} and original 'District' column deleted")
            fe.encode_climate_event_type()
            event_mapping = {i: event for i, event in enumerate(fe.event_type_classes)}
            st.write(f"'EventType' column is encoded with: {event_mapping} and original 'EventType' column deleted")
            
            # st.session_state.df_feature_added = fe.df
            added_feature.extend(['district_encoded','eventtype_encoded'])
            st.success(f"Feature Engineering Applied Successfully! {len(added_feature)} features added.")

            st.write('Dataset sample after Feature addition')
            st.dataframe(fe.df.head())
            # Show data stats
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Total Observations",  fe.df.shape[0])
            with col2:
                st.metric("Variables",  fe.df.shape[1])
    st.session_state.fe_obj = fe
    return fe.df
