import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Configure the page
st.set_page_config(page_title="Real Estate Price Predictor", layout="wide")
st.title("🏡 Real Estate Price Prediction App")

# Function to load data from uploaded file and clean columns
@st.cache_data(show_spinner=False)
def load_data(file_obj):
    data = pd.read_csv(file_obj)
    data.columns = data.columns.str.strip().str.lower()  # ensure lowercase columns
    return data

# Function to train a model using a pipeline
def train_model(df):
    # Ensure the target column 'price' exists
    if 'price' not in df.columns:
        raise KeyError(f"'price' column not found. Available columns: {list(df.columns)}")
    
    # Features (all columns except target) and target
    X = df.drop(columns=['price'])
    y = df['price']

    # Identify numeric and categorical columns based on the data types
    numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_cols = X.select_dtypes(include='object').columns.tolist()

    # Define a preprocessor that passes numeric features and one-hot encodes categorical features
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', 'passthrough', numeric_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols)
        ]
    )

    # Build the pipeline: preprocessing + regressor
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
    ])

    # Split into training and testing subsets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    pipeline.fit(X_train, y_train)

    # Evaluate on the test set (we use RMSE for reporting)
    y_pred = pipeline.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    return pipeline, rmse

# Section 1: Upload CSV and train model
uploaded_file = st.file_uploader("Upload your real_estate.csv file", type=["csv"])
if uploaded_file:
    try:
        df = load_data(uploaded_file)
        st.subheader("📊 Preview of Uploaded Data")
        st.write(df.head())

        if st.button("Train Model"):
            pipeline, rmse = train_model(df)
            st.session_state.pipeline = pipeline  # store the trained pipeline in session state
            st.success("✅ Model trained successfully!")
            st.metric(label="Root Mean Squared Error (RMSE)", value=f"{rmse:.2f}")
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
else:
    st.info("Please upload your CSV file.")

# Section 2: Prediction form (only visible if the model is trained)
if "pipeline" in st.session_state:
    st.write("---")
    st.subheader("🔮 Predict a Property Price")
    
    with st.form("prediction_form"):
        # Get numerical values; adjust default values as needed
        year_sold    = st.number_input("Year Sold", value=2020, step=1)
        property_tax = st.number_input("Property Tax", value=0.0, format="%.2f")
        insurance    = st.number_input("Insurance", value=0.0, format="%.2f")
        beds         = st.number_input("Beds", value=3, step=1)
        baths        = st.number_input("Baths", value=2, step=1)
        sqft         = st.number_input("Square Footage", value=1500, step=1)
        year_built   = st.number_input("Year Built", value=2000, step=1)
        lot_size     = st.number_input("Lot Size", value=1000, step=1)
        basement     = st.number_input("Basement (0 if none)", value=0, step=1)
        property_type= st.text_input("Property Type", value="Bungalow")

        submitted = st.form_submit_button("Predict Price")

    if submitted:
        # Create a DataFrame using the same column names as the training features
        input_data = pd.DataFrame({
            "year_sold": [year_sold],
            "property_tax": [property_tax],
            "insurance": [insurance],
            "beds": [beds],
            "baths": [baths],
            "sqft": [sqft],
            "year_built": [year_built],
            "lot_size": [lot_size],
            "basement": [basement],
            "property_type": [property_type]
        })

        try:
            # Generate prediction using the trained pipeline stored in session state
            prediction = st.session_state.pipeline.predict(input_data)
            st.success(f"### Predicted Price: ${prediction[0]:,.2f}")
        except Exception as e:
            st.error(f"Prediction failed: {e}")
