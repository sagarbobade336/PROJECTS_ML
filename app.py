import streamlit as st
import pandas as pd
import pickle
import datetime
import warnings
warnings.filterwarnings("ignore")

# --- Page Config ---
st.set_page_config(
    page_title="BigMart Sales Prediction App",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Load the Model ---
try:
    with open("bigmart_best_model.pkl", "rb") as f:
        pipeline, sklearn_version = pickle.load(f)
    st.success(f"✅ Model loaded successfully! (Sklearn version: {sklearn_version})")
except FileNotFoundError:
    st.error("❌ The 'bigmart_best_model.pkl' file was not found. Please ensure it is in the same directory as this script.")
    st.stop()
except Exception as e:
    st.error(f"❌ An error occurred while loading the model: {e}")
    st.stop()

# --- App Title and Description ---
st.title("🛒 BigMart Sales Predictor")
st.markdown("Enter the item and outlet details to predict the *Item Outlet Sales*.")
st.markdown("---")

# --- Sidebar Inputs ---
st.sidebar.header("📝 Input Features")

# Item Features
st.sidebar.subheader("Item Information")
item_identifier = st.sidebar.text_input("Item Identifier", value="FDN15")
item_weight = st.sidebar.number_input("Item Weight", value=17.5, min_value=0.0)
item_fat_content = st.sidebar.selectbox("Item Fat Content", ("Low Fat", "Regular"))
item_visibility = st.sidebar.number_input("Item Visibility", value=0.0167, min_value=0.0)
item_type = st.sidebar.selectbox("Item Type", ("Dairy", "Soft Drinks", "Meat", "Fruits and Vegetables", "Household", "Baking Goods", "Snack Foods", "Frozen Foods", "Breakfast", "Health and Hygiene", "Canned", "Breads", "Starchy Foods", "Others", "Hard Drinks", "Seafood"))
item_mrp = st.sidebar.number_input("Item MRP", value=141.6, min_value=0.0)

# Outlet Features
st.sidebar.subheader("Outlet Information")
outlet_identifier = st.sidebar.selectbox("Outlet Identifier", ("OUT049", "OUT018", "OUT046", "OUT045", "OUT013", "OUT017", "OUT010", "OUT027", "OUT035", "OUT019"))
outlet_establishment_year = st.sidebar.number_input("Outlet Establishment Year", value=1999, min_value=1900, max_value=datetime.date.today().year)
outlet_size = st.sidebar.selectbox("Outlet Size", ("Medium", "Small", "High", "Tiny"))
outlet_location_type = st.sidebar.selectbox("Outlet Location Type", ("Tier 1", "Tier 2", "Tier 3"))
outlet_type = st.sidebar.selectbox("Outlet Type", ("Supermarket Type1", "Supermarket Type2", "Grocery Store", "Supermarket Type3"))

# --- Main Page ---
if st.button("Predict Sales"):
    with st.spinner("Calculating..."):
        # Create a DataFrame from user input
        input_data = pd.DataFrame([{
            'Item_Identifier': item_identifier,
            'Item_Weight': item_weight,
            'Item_Fat_Content': item_fat_content,
            'Item_Visibility': item_visibility,
            'Item_Type': item_type,
            'Item_MRP': item_mrp,
            'Outlet_Identifier': outlet_identifier,
            'Outlet_Establishment_Year': outlet_establishment_year,
            'Outlet_Size': outlet_size,
            'Outlet_Location_Type': outlet_location_type,
            'Outlet_Type': outlet_type
        }])
        
        # Apply the same transformations as the training script
        # 1. Feature Engineering
        input_data['Outlet_Age'] = datetime.date.today().year - input_data['Outlet_Establishment_Year']
        input_data.drop('Outlet_Establishment_Year', axis=1, inplace=True)
        
        # 2. Correcting Item_Visibility
        input_data['Item_Visibility'] = input_data['Item_Visibility'].apply(lambda x: min(x, 0.3))
        
        # 3. Correcting Item_Fat_Content (already handled by selectbox, but good practice to have)
        input_data['Item_Fat_Content'] = input_data['Item_Fat_Content'].replace({
            'low fat': 'Low Fat',
            'LF': 'Low Fat',
            'reg': 'Regular'
        })
        
        # Make a prediction
        prediction = pipeline.predict(input_data)
        predicted_sales = prediction[0]

    # --- Display Results ---
    st.markdown("---")
    st.subheader("🔮 Predicted Item Outlet Sales")
    
    st.metric("Predicted Sales", f"${predicted_sales:.2f}")

    st.markdown("""
        *Note:* This is a predicted value based on a machine learning model.
        The actual sales may vary.
    """)

# --- Footer ---
st.markdown("---")
st.markdown("Created by S.B.A")