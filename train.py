import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import joblib


#1. LOAD THE DATA 
df = pd.read_csv('C:\\Users\\ADMIN\\Desktop\\Dataset\\dubai_properties.csv')

#2. CLEAN AND FILTER
keep_types = ['Apartment', 'Villa', 'Townhouse', 'Penthouse']
df = df[df['Type'].isin(keep_types)]
df = df[df['Purpose'] == 'For Rent']
df = df[df['Rent'] > 0]
df = df[df['Rent'] <= 2000000]
df = df.drop(columns=['Address', 'Rent_category', 'Rent_per_sqft',
                       'Posted_date', 'Age_of_listing_in_days',
                       'Latitude', 'Longitude'])

print("Clean shape:", df.shape)

# 3. ENCODE
le_city      = LabelEncoder()
le_type      = LabelEncoder()
le_furnishing = LabelEncoder()
le_location  = LabelEncoder()

df['City_encoded']       = le_city.fit_transform(df['City'])
df['Type_encoded']       = le_type.fit_transform(df['Type'])
df['Furnishing_encoded'] = le_furnishing.fit_transform(df['Furnishing'])
df['Location_encoded']   = le_location.fit_transform(df['Location'])

#4. TRAIN 
features = ['City_encoded', 'Type_encoded', 'Furnishing_encoded',
            'Location_encoded', 'Beds', 'Baths', 'Area_in_sqft']

X = df[features]
y = df['Rent']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

score = model.score(X_test, y_test)
print(f"Model R²: {score:.2f}")

#5. SAVE 
joblib.dump(model, 'rent_model.pkl')
joblib.dump({
    'city':       le_city,
    'type':       le_type,
    'furnishing': le_furnishing,
    'location':   le_location
}, 'encoders.pkl')

print("Model and encoders saved!")


