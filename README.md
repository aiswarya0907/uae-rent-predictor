# uae-rent-predictor

A machine learning web app that predicts annual rent prices across UAE emirates based on property features.

Live Demo: https://uae-rent-predictor.streamlit.app/

Project Overview

Rent prices in the UAE vary drastically — a 2-bedroom apartment in Dubai Marina can cost 5x more than the same unit in Sharjah. This app helps users estimate annual rent based on location, property type, size, and furnishing status.

Built as a portfolio project using a real dataset of 73,000+ UAE property listings sourced from Kaggle.

---

Features: 

- Rent Predictor — input property details and get an instant annual rent estimate
- Market Analysis — interactive charts comparing rent across cities, property types, and furnishing
- Full UAE Coverage — Dubai, Abu Dhabi, Sharjah, Ajman, Ras Al Khaimah, Al Ain, and Umm Al Quwain

---

How it works:

1. Data Cleaning— filtered 73k listings, removed outliers, kept 4 core property types
2. Feature Engineering — label encoded categorical variables (city, type, furnishing, location)
3. Model Training — Random Forest Regressor trained on 80% of data
4. Evaluation — R² score of 0.86 on test set

---

Tech Stack
Python- Core language 
Pandas- Data cleaning & EDA 
Scikit-learn - ML model 
Plotly- Interactive charts
Streamlit- Web app & deployment


## 📊 Dataset

UAE property listings sourced from Kaggle covering all 7 emirates. Cleaned to remove zero-rent entries, non-residential types, and outliers above 2M AED.

Kaggle link: https://www.kaggle.com/datasets/azharsaleem/real-estate-goldmine-dubai-uae-rental-market?resource=download
