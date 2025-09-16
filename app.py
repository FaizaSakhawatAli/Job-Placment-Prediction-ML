import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

# Load dataset
df = pd.read_csv("Job_Placement_Data.csv")

# Encode categorical columns
le = LabelEncoder()
for col in df.columns:
    if df[col].dtype == 'object':
        df[col] = le.fit_transform(df[col])

# Features (X) and Target (y)
X = df.drop('status', axis=1)
y = df['status']

# Train model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
lg = LogisticRegression(max_iter=1000)
lg.fit(X_train, y_train)

# Web app
img = Image.open('Job-Placement-Agency.jpg')
st.image(img, width=650)
st.title("Job Placement Prediction Model")

st.write("⚠️ Enter the following features in order (comma-separated):")
st.write(", ".join(X.columns))

input_text = st.text_input("Enter all features:")

if input_text:
    try:
        input_list = input_text.split(',')
        np_df = np.asarray(input_list, dtype=float)
        prediction = lg.predict(np_df.reshape(1, -1))

        if prediction[0] == 1:
            st.success("✅ This Person is Placed")
        else:
            st.error("❌ This Person is Not Placed")

    except Exception as e:
        st.error(f"⚠️ Invalid input format: {e}")
