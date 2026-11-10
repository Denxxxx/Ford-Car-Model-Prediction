import streamlit as st
import pandas as pd
import joblib
import os
import sklearn

# Set page configuration
st.set_page_config(
    page_title="Ford Used Car Price Predictor",
    page_icon="🚗",
    layout="centered"
)

# Title and Header Description
st.title("🚗 Ford Used Car Price Predictor")
st.markdown("""
Predict the market value of a used Ford car using our machine learning models trained on real dealership sales data.
""")

# Helper function to load the model with verification checks
@st.cache_resource
def load_model(filename):
    if os.path.exists(filename):
        try:
            return joblib.load(filename)
        except Exception as e:
            st.error(f"""⚠️ **Deserialization Error loading {filename}:** 
            This usually happens because of a mismatch in library versions between the training environment (Colab) and the deployment server. 
            
            * **Deployment Environment scikit-learn version:** `{sklearn.__version__}`
            * **Error Details:** `{str(e)}`
            
            **To fix this:** Make sure to deploy using the `requirements.txt` file created in your notebook, or retrain and export the models on the same system run environment.""")
    return None

# Try to load both models
rf_model = load_model('random_forest_model.joblib')
lr_model = load_model('linear_regression_model.joblib')

# Check if at least one model exists
if rf_model is None and lr_model is None:
    st.warning("⚠️ Waiting for model pipelines to be loaded successfully to run the inference application.")
else:
    # Define features based on clean unique values
    models_list = [' Fiesta', ' Focus', ' Kuga', ' EcoSport', ' Mondeo', ' S-MAX', ' B-MAX', 
                   ' Edge', ' KA', ' Galaxy', ' Tourneo Custom', ' Grand C-MAX', ' Ka+', 
                   ' Mustang', ' Tourneo Connect', ' Grand Tourneo Connect', ' Puma']
    
    transmissions_list = ['Manual', 'Automatic', 'Semi-Auto']
    fuels_list = ['Petrol', 'Diesel', 'Hybrid']

    # Setup Input Form
    st.subheader("🔧 Customize Car Specifications")
    
    col1, col2 = st.columns(2)
    
    with col1:
        car_model = st.selectbox("Select Car Model", sorted(models_list), index=models_list.index(' Fiesta') if ' Fiesta' in models_list else 0)
        year = st.slider("Year of Registration", 2000, 2020, 2018)
        transmission = st.selectbox("Transmission Type", transmissions_list)

    with col2:
        fuel_type = st.selectbox("Fuel Type", fuels_list)
        tax = st.slider("Annual Road Tax (£)", 0, 175, 145, step=5)
        engine_size = st.slider("Engine Displacement (Litres)", 0.0, 5.0, 1.2, step=0.1)

    # Prepare inputs for inference
    input_data = pd.DataFrame([{
        'model': car_model,
        'year': year,
        'transmission': transmission,
        'fuelType': fuel_type,
        'tax': tax,
        'engineSize': engine_size
    }])

    st.markdown("--- ")

    # Prediction trigger
    if st.button("🔮 Calculate Estimated Price", use_container_width=True):
        # Display a spinner during inference
        with st.spinner("Running pricing algorithms..."):
            predictions = {}
            if rf_model is not None:
                try:
                    predictions['Random Forest'] = rf_model.predict(input_data)[0]
                except Exception as e:
                    st.error(f"Inference error (Random Forest): {e}")
            if lr_model is not None:
                try:
                    predictions['Linear Regression'] = lr_model.predict(input_data)[0]
                except Exception as e:
                    st.error(f"Inference error (Linear Regression): {e}")

            if predictions:
                # Highlight predictions in boxes
                st.subheader("🏁 Estimated Valuation Result")
                
                for model_name, val in predictions.items():
                    bg_color = "#e3f2fd" if "Recommended" in model_name else "#f5f5f5"
                    border_color = "#1976d2" if "Recommended" in model_name else "#9e9e9e"
                    
                    st.markdown(f"""
                    <div style="padding: 20px; border-radius: 8px; background-color: {bg_color}; border: 1px solid {border_color}; margin-bottom: 12px;">
                        <span style="font-size: 14px; text-transform: uppercase; letter-spacing: 1px; color: #555;"><b>{model_name}</b></span>
                        <h2 style="margin: 5px 0 0 0; color: #111;">£{val:,.2f}</h2>
                    </div>
                    """, unsafe_allow_html=True)
