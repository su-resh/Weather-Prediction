import streamlit as st
# from pathlib import Path
import pandas as pd
# import os
# import sys

# utils_dir = os.path.join(Path(__file__).parent, "utils")
# sys.path.append(utils_dir)
# from data_utils import PrepareData
# from models import split_data, train_model, evaluate_model, save_model
# from visualizations import plot_confusion_matrix, plot_regression_results

from utils.data_utils import PrepareData
from utils.models import split_data, train_model, evaluate_model, save_model
from utils.visualizations import plot_confusion_matrix, plot_regression_evaluation

def show(fe = None):
    """
    Page for model loading and training
    """
    st.title("Model Training and Evaluation")  

    if 'fe_obj' not in st.session_state:
        st.warning("First you need to perform Feature Engineering. Go to 'Feature Engineering' page for Feature engineering")
    else:
        # Train test split
        test_size = st.slider("Test data size (%)", 10, 40, 20)

        test_size_rows = int(len(st.session_state.fe_obj.df) * test_size/100)
        train_size_rows = int(len((st.session_state.fe_obj.df)) - test_size_rows)
        st.write(f"Training Data: {train_size_rows} Samples")
        st.write(f"Test data : {test_size_rows} samples")

        ### information abot the 3 models ["Regression", "Multi-Class Classifier", "Binary Classifier"]
        st.markdown("""
            **Regression** is used to predict continuous numerical values. In this case, the model will be predicting 
            climate variables ['Precip', 'Humidity_2m', 'Temp_2m', 'MaxTemp_2m', 'MinTemp_2m'].
                    
            **Multi-Class Classifier** is used to predict one of several categories (classes). 
            In this case, the model will be predicting the type of events ['ColdWave', 'HighTemp', 'Heatwave', 'HeavyRain'].
                    
            **Binary Classifier** is used to predict one of two possible outcomes. 
            In this case, the model will be predicting whether an extreme event occurs (yes/no).
            """)
        
        # Prepare Data and Display description based on the selected model type
        data_preparer = PrepareData(st.session_state.fe_obj.df)

        # Prepare datasets for ML models
        X_reg, y_reg = data_preparer.prepare_data_regression()
        X_multi, y_multi = data_preparer.prepare_data_multi_classifier()
        X_binary, y_binary = data_preparer.prepare_data_binary_classifier()

        # Split data
        X_train_reg, X_test_reg, y_train_reg, y_test_reg = split_data(X_reg, y_reg, test_size/100)
        X_train_multi, X_test_multi, y_train_multi, y_test_multi = split_data(X_multi, y_multi, test_size/100)
        X_train_binary, X_test_binary, y_train_binary, y_test_binary = split_data(X_binary, y_binary, test_size/100)
        # X_train, X_test, y_train, y_test, = split_data(X, y, test_size/100)

        # train model Button
        if st.button('Train All Model'):
            ### 1. Regression Model Training
            st.subheader("Training Regression Model...")
            st.markdown("""
            **Regression** is used to predict continuous numerical values. In this case, the model will be predicting 
            climate variables ['Precip', 'Humidity_2m', 'Temp_2m', 'MaxTemp_2m', 'MinTemp_2m'].
            """)
            progress_reg = st.progress(0)
            with st.spinner("Training Regression Model..."):
                # List features
                st.markdown(f"**Features (X):** {list(X_train_reg.columns)}")
                # List targets
                st.markdown(f"**Target (y):** {list(y_train_reg.columns)}")

                progress_reg.progress(10)
                model_reg, scaler_reg = train_model(X_train_reg, y_train_reg, model_type = 'Regression')
                progress_reg.progress(60)
                metrics_reg = evaluate_model(model_reg, scaler_reg, X_train_reg, y_train_reg, X_test_reg, y_test_reg, "Regression")
                progress_reg.progress(90)
                st.session_state['model_regression'] = model_reg
                st.session_state['scaler_regression'] = scaler_reg
                progress_reg.progress(100)
                st.success("✅ Regression Model Trained Successfully!")
                st.subheader("Regression Model Metrics")
                show_metrics(metrics_reg)
                # st.json(metrics_reg)
                plt_reg_test = plot_regression_evaluation(metrics_reg['y_test'], metrics_reg['y_pred_test'])
                st.pyplot(plt_reg_test)
                save_model(model_reg, scaler_reg, 'regression')

            ### 2. Multi-Class Classifier Training
            st.subheader("Training Multi-Class Classifier Model...")
            st.markdown("""        
            **Multi-Class Classifier** is used to predict one of several categories (classes). 
            In this case, the model will be predicting the type of events ['Normal', 'ColdWave', 'HighTemp', 'Heatwave', 'HeavyRain'].
            """)
            progress_multi = st.progress(0)
            with st.spinner("Training Multi-Class Classifier Model..."):
                st.markdown(f"**Features (X):** {list(X_train_multi.columns)}")
                st.markdown(f"**Target (y):** {list(y_train_multi.columns)}")
                progress_multi.progress(10)
                model_multi, scaler_multi = train_model(X_train_multi, y_train_multi, model_type="Multi-Class Classifier")
                progress_multi.progress(60)
                metrics_multi = evaluate_model(model_multi, scaler_multi, X_train_multi, y_train_multi, X_test_multi, y_test_multi, "Multi-Class Classifier")
                progress_multi.progress(90)
                st.session_state['model_multi'] = model_multi
                st.session_state['scaler_multi'] = scaler_multi
                progress_multi.progress(100)
                st.success("✅ Multi-Class Classifier Model Trained Successfully!")
                st.subheader("Multi-Class Classifier Metrics")
                show_metrics(metrics_multi, class_names = st.session_state.fe_obj.event_type_classes)
                save_model(model_multi, scaler_multi, 'multi_class_classifier')
                               
            ### 3. Binary Classifier Training
            st.subheader("Training Binary Classifier Model...")
            st.markdown("""
            **Binary Classifier** is used to predict one of two possible outcomes. 
            In this case, the model will be predicting whether an extreme event occurs (yes/no).
            """)
            progress_binary = st.progress(0)
            with st.spinner("Training Binary Classifier Model..."):
                st.markdown(f"**Features (X):** {list(X_train_binary.columns)}")
                st.markdown(f"**Target (y):** {list(y_train_binary.columns)}")
                progress_binary.progress(10)
                model_binary, scaler_binary = train_model(X_train_binary, y_train_binary, model_type='Binary Classifier')
                progress_binary.progress(60)
                metrics_binary = evaluate_model(model_binary, scaler_binary, X_train_binary, y_train_binary, X_test_binary, y_test_binary, "Binary Classifier")
                progress_binary.progress(90)
                st.session_state['model_binary'] = model_binary
                st.session_state['scaler_binary'] = scaler_binary
                progress_binary.progress(100)
                st.success("✅ Binary Classifier Model Trained Successfully!")
                st.subheader("Binary Classifier Metrics")
                show_metrics(metrics_binary, class_names = None)
                save_model(model_binary, scaler_binary, 'binary_classifier')

            st.success("🎯 All Models Trained Successfully!")
       
def show_metrics(metrics, class_names = None):
    """
    This method use two columns for displaying train and test metrics side by side
    metrics contains 'model_type' field which can be either "Regression" or "Classification"
    """
    col1, col2 = st.columns(2)

    if metrics['model_type'] == 'Regression':
        with col1:
            st.subheader("Training Metrics")
            st.write(f"RMSE: {metrics['train_rmse']}")
            st.write(f"R2: {metrics['train_r2']}")

        with col2:
            st.subheader("Testing Metrics")
            st.write(f"RMSE: {metrics['test_rmse']}")
            st.write(f"R2: {metrics['test_r2']}")
    
    elif metrics['model_type'] == 'Classification':
        with col1:
            st.subheader("Training Metrics")
            st.write(f"Accuracy: {metrics['train_accuracy']}")
            st.write("Confusion Matrix:")
            # st.write(metrics['train_confusion_matrix'])
            conf_plt_multi_test = plot_confusion_matrix(metrics['train_confusion_matrix'],
                                                        fig_size = (4,3),
                                                        class_names = class_names)
            st.pyplot(conf_plt_multi_test)
            st.write("Classification Report:")
            # st.text(metrics['train_classification_report'])
            report_df = pd.DataFrame(metrics['train_classification_report']).transpose()
            st.dataframe(report_df)

        with col2:
            st.subheader("Testing Metrics")       
            st.write(f"Accuracy: {metrics['test_accuracy']}")
            st.write("Confusion Matrix:")
            # st.write(metrics['test_confusion_matrix'])
            conf_plt_multi_test = plot_confusion_matrix(metrics['test_confusion_matrix'],
                                                        fig_size = (4,3),
                                                        class_names = class_names)
            st.pyplot(conf_plt_multi_test)
            st.write("Classification Report:")
            # st.text(metrics['test_classification_report'])
            report_df = pd.DataFrame(metrics['test_classification_report']).transpose()
            st.dataframe(report_df)

    else:
        st.warning(f"Argument '{metrics['model_type']}' to metrics['model_type'] is not allowed. It must be one of ['Regression', 'Classification']")