import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor


@st.cache_resource
def load_model():
    df = pd.read_csv('dubai_properties.csv')

    keep_types = ['Apartment', 'Villa', 'Townhouse', 'Penthouse']
    df = df[df['Type'].isin(keep_types)]
    df = df[df['Purpose'] == 'For Rent']
    df = df[df['Rent'] > 0]
    df = df[df['Rent'] <= 2000000]

    le_city       = LabelEncoder()
    le_type       = LabelEncoder()
    le_furnishing = LabelEncoder()
    le_location   = LabelEncoder()

    df['City_encoded']       = le_city.fit_transform(df['City'])
    df['Type_encoded']       = le_type.fit_transform(df['Type'])
    df['Furnishing_encoded'] = le_furnishing.fit_transform(df['Furnishing'])
    df['Location_encoded']   = le_location.fit_transform(df['Location'])

    features = ['City_encoded', 'Type_encoded', 'Furnishing_encoded',
                'Location_encoded', 'Beds', 'Baths', 'Area_in_sqft']
    X = df[features]
    y = df['Rent']

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)

    encoders = {
        'city': le_city,
        'type': le_type,
        'furnishing': le_furnishing,
        'location': le_location
    }
    return model, encoders


model, encoders = load_model()

st.title("UAE Rent Price Predictor")

# USER INPUTS
city       = st.selectbox("City",        encoders['city'].classes_)
prop_type  = st.selectbox("Property Type", encoders['type'].classes_)
furnishing = st.selectbox("Furnishing",  encoders['furnishing'].classes_)
location   = st.selectbox("Location",    encoders['location'].classes_)
beds       = st.slider("Bedrooms", 0, 6, 2)
baths      = st.slider("Bathrooms", 1, 7, 2)
area       = st.number_input("Area (sqft)", min_value=100, max_value=20000, value=1000)

# PREDICT
if st.button("Predict Rent"):
    city_enc = encoders['city'].transform([city])[0]
    type_enc = encoders['type'].transform([prop_type])[0]
    furn_enc = encoders['furnishing'].transform([furnishing])[0]
    loc_enc  = encoders['location'].transform([location])[0]

    input_df = pd.DataFrame([[city_enc, type_enc, furn_enc, loc_enc, beds, baths, area]],
                             columns=['City_encoded', 'Type_encoded', 'Furnishing_encoded',
                                      'Location_encoded', 'Beds', 'Baths', 'Area_in_sqft'])

    prediction = model.predict(input_df)[0]
    st.success(f"Estimated Annual Rent: AED {prediction:,.0f}")
