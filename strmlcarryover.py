import streamlit as st
import pandas as pd
from sklearn.linear_model import LogisticRegression

st.title("Course Failure Risk Detector")

# Train model once
df = pd.read_excel("STP_213_2026.xlsx")

# 1. Temporary debug line (check your Streamlit logs to see the output)
print("Available columns:", df.columns.tolist())

# 2. Clean up trailing/leading spaces automatically just in case
df.columns = df.columns.str.strip()

# 3. Your original selection
X = df[['Pract', 'CA', 'Exam']]

df = pd.read_excel("STP_213_2026.xlsx")
X = df[['Pract', 'CA', 'Exam']]
y = df['Total']
model = LogisticRegression().fit(X, y)

# User input
st.sidebar.header("Enter Student Data")
pct = st.sidebar.slider("Pract", 0, 40, 20)
test = st.sidebar.slider("CA", 0, 20, 10)
exm = st.sidebar.slider("Exam", 0, 40, 20)
#courses = st.sidebar.slider("Courses Registered", 5, 10, 6)

if st.button("Predict Risk"):
    pred = model.predict([[pct, test, exm]])[0]
    prob = model.predict_proba([[pct, test, exm]])[0][1]
    st.subheader(f"Prediction: {pred}")
    st.write(f"Probability of Carryover: {prob:.2%}")
    if pred == "Carryover":
        st.error("High Risk. Recommend academic intervention.")
    else:
        st.success("Low Risk. On track to pass.")

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# 1. Load data
df = pd.read_excel("STP_213_2026.xlsx")

# 2. Split features and target
X = df[['Pract', 'CA', 'Exam']]
y = df['Total']

# 3. Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Train model
model = LogisticRegression()
model.fit(X_train, y_train)

# 5. Evaluate
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Report:\n", classification_report(y_test, y_pred))

# 6. Feature importance
print("Feature importance:", dict(zip(X.columns, model.coef_[0])))

import streamlit as st
import joblib
import numpy as np

model = joblib.load('risk_model.pkl') # Load your trained model
